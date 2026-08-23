"""Shared AI Deck protocol and Windows environment helpers.

Protocol constants and framing are ported unchanged from
``real_flight/aideck_packet_test.py`` and ``real_flight/aideck_reconnect_test.py``.
Only the platform-specific shell commands differ.

Standard library only. No pip installs required.
"""

from __future__ import annotations

import json
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# --- protocol constants (must match the GAP8/ESP CPX streamer) ---------------

IMG_HEADER_MAGIC = 0xBC
JPEG_ENCODING = 1
CPX_HEADER_STRUCT = "<HBB"       # length, routing, function
CPX_HEADER_BYTES = 4
IMG_HEADER_STRUCT = "<BHHBBI"    # magic, width, height, depth, type, size
IMG_HEADER_BYTES = 11

DEFAULT_PORT = 5000
DEFAULT_IPS = ("192.168.4.1", "192.168.7.201")
DEFAULT_SSID = "aideck-stream"

TESTBED_DIR = Path(__file__).resolve().parent
LAB_DIR = TESTBED_DIR.parent.parent
DEFAULT_OUT_DIR = LAB_DIR / "real_flight" / "aideck_logs"

# Keep GUI-launched subprocesses from flashing a console window.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


class Cancelled(Exception):
    """Raised inside a test when the operator asks it to stop."""


class CancelToken:
    """Cooperative stop flag shared between the GUI and a worker thread."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("stopped by operator")


class Reporter:
    """Writes a run log to disk and mirrors every line to an optional callback."""

    def __init__(self, run_dir: Path, name: str, on_line=None) -> None:
        self.run_dir = run_dir
        self.log_path = run_dir / f"{name}.log"
        self._fh = self.log_path.open("w", encoding="utf-8")
        self._on_line = on_line

    def info(self, message: str = "") -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"{stamp} {message}\n" if message else "\n")
        self._fh.flush()
        if self._on_line is not None:
            self._on_line(message)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self) -> "Reporter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- Windows shell helpers ---------------------------------------------------

def _decode(raw: bytes) -> str:
    """Decode console output without guessing wrong and crashing."""
    for encoding in ("utf-8", "cp1252", "cp850"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def run_text(cmd: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW
        )
        return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)
    except Exception as exc:  # noqa: BLE001 - the failure text is the useful part
        return 999, "", repr(exc)


def ping(ip: str, count: int = 2, timeout_ms: int = 2000) -> dict:
    """Windows ping. Note ``-n``/``-w`` where Linux uses ``-c``/``-W``."""
    rc, out, err = run_text(
        ["ping", "-n", str(count), "-w", str(timeout_ms), ip], timeout=count * 4 + 5
    )
    return {"returncode": rc, "stdout": out, "stderr": err, "ok": rc == 0}


def wlan_interfaces() -> dict:
    """SSID / signal / rate for the active wireless adapter."""
    rc, out, err = run_text(["netsh", "wlan", "show", "interfaces"])
    parsed: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in {
            "name", "state", "ssid", "bssid", "signal",
            "receive rate (mbps)", "transmit rate (mbps)", "channel", "radio type",
        }:
            parsed[key] = value
    return {"returncode": rc, "stdout": out, "stderr": err, "parsed": parsed}


def collect_env() -> dict:
    """Snapshot of adapter, routing and wireless state at run time."""
    env: dict = {}
    for name, cmd in {
        "ipconfig_all": ["ipconfig", "/all"],
        "route_print": ["route", "print", "-4"],
    }.items():
        rc, out, err = run_text(cmd)
        env[name] = {"returncode": rc, "stdout": out, "stderr": err}
    env["netsh_wlan"] = wlan_interfaces()
    env["hostname"] = socket.gethostname()
    return env


def local_ipv4_addresses() -> list[str]:
    addrs: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if addr not in addrs:
                addrs.append(addr)
    except OSError:
        pass
    return addrs


# --- stream helpers ----------------------------------------------------------

def recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def connect_to_deck(ip: str, port: int, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def read_packet(sock: socket.socket) -> tuple[int, int, int, bytes]:
    """Read one CPX packet. Payload is ``length - 2`` bytes."""
    header = recv_exact(sock, CPX_HEADER_BYTES)
    length, routing, function = struct.unpack(CPX_HEADER_STRUCT, header)
    payload = recv_exact(sock, max(0, length - 2))
    return length, routing, function, payload


def parse_image_header(payload: bytes) -> dict | None:
    if len(payload) < IMG_HEADER_BYTES or payload[0] != IMG_HEADER_MAGIC:
        return None
    magic, width, height, depth, img_type, size = struct.unpack(
        IMG_HEADER_STRUCT, payload[:IMG_HEADER_BYTES]
    )
    return {
        "magic": magic,
        "width": width,
        "height": height,
        "depth": depth,
        "type": img_type,
        "encoding": "JPEG" if img_type == JPEG_ENCODING else "RAW",
        "size": size,
    }


def frame_suffix(img_type: int) -> str:
    return ".jpg" if img_type == JPEG_ENCODING else ".bin"


# --- run folders -------------------------------------------------------------

def new_run_dir(out_dir: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(run_dir: Path, summary: dict) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def zip_run_dir(run_dir: Path) -> Path:
    """Zip the run folder next to itself so it can be moved off this box."""
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=str(run_dir))
    return Path(archive)


def stamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def monotonic() -> float:
    return time.monotonic()
