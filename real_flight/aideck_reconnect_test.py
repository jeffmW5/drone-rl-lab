#!/usr/bin/env python3.11
"""AI Deck reconnect diagnostic.

This test checks whether the deck can deliver one good frame per fresh TCP
connection. If it can, the desktop viewer can be made usable with timeout +
reconnect logic. If only the first connection works, the firmware/ESP32 path is
wedging and probably needs a power-cycle or firmware patch.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = LAB_DIR / "real_flight" / "aideck_logs"
IMG_HEADER_MAGIC = 0xBC
JPEG_ENCODING = 1


def run_text(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 999, "", repr(exc)


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("aideck_reconnect_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(run_dir / "reconnect_test.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def connect(ip: str, port: int, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def read_one_frame(sock: socket.socket, read_timeout: float, max_packets: int = 100) -> dict:
    sock.settimeout(read_timeout)
    started = time.monotonic()
    packets = []

    packet_info = recv_exact(sock, 4)
    length, routing, function = struct.unpack("<HBB", packet_info)
    payload = recv_exact(sock, max(0, length - 2))
    packets.append(
        {
            "length": length,
            "payload_len": len(payload),
            "routing": routing,
            "function": function,
            "first16_hex": payload[:16].hex(),
            "t_rel_s": round(time.monotonic() - started, 6),
        }
    )

    if len(payload) < 11 or payload[0] != IMG_HEADER_MAGIC:
        raise ValueError(f"first packet is not an image header: {payload[:16].hex()}")

    magic, width, height, depth, img_type, size = struct.unpack("<BHHBBI", payload[:11])
    img = bytearray()
    while len(img) < size:
        if len(packets) >= max_packets:
            raise TimeoutError(f"max packet count reached with {len(img)}/{size} frame bytes")
        packet_info = recv_exact(sock, 4)
        length, routing, function = struct.unpack("<HBB", packet_info)
        payload = recv_exact(sock, max(0, length - 2))
        img.extend(payload)
        packets.append(
            {
                "length": length,
                "payload_len": len(payload),
                "routing": routing,
                "function": function,
                "first16_hex": payload[:16].hex(),
                "frame_bytes": len(img),
                "t_rel_s": round(time.monotonic() - started, 6),
            }
        )

    return {
        "width": width,
        "height": height,
        "depth": depth,
        "type": img_type,
        "encoding": "JPEG" if img_type == JPEG_ENCODING else "RAW",
        "size": size,
        "packets": packets,
        "frame_bytes": bytes(img[:size]),
        "elapsed_s": round(time.monotonic() - started, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Deck reconnect diagnostic")
    parser.add_argument("--ip", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=4.0)
    parser.add_argument("--read-timeout", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"reconnect_test_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)

    logger.info("AI Deck reconnect test")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Target: {args.ip}:{args.port}")
    logger.info(f"Attempts: {args.attempts}, delay={args.delay}s")

    env_info = {}
    for name, cmd in {
        "hostname_I": ["hostname", "-I"],
        "ip_addr": ["ip", "-br", "addr"],
        "ip_route": ["ip", "route"],
    }.items():
        rc, out, err = run_text(cmd)
        env_info[name] = {"returncode": rc, "stdout": out, "stderr": err}
        logger.info("")
        logger.info(f"$ {' '.join(cmd)}")
        logger.info(out or err or f"returncode={rc}")

    results = []
    for attempt in range(1, args.attempts + 1):
        logger.info("")
        logger.info(f"Attempt {attempt}/{args.attempts}: connecting...")
        result = {"attempt": attempt, "ok": False, "error": None}
        sock = None
        try:
            sock = connect(args.ip, args.port, args.connect_timeout)
            logger.info("  connected; reading one complete frame...")
            frame = read_one_frame(sock, args.read_timeout)
            suffix = ".jpg" if frame["type"] == JPEG_ENCODING else ".bin"
            frame_path = run_dir / f"attempt_{attempt:02d}{suffix}"
            frame_path.write_bytes(frame["frame_bytes"])
            result.update(
                {
                    "ok": True,
                    "width": frame["width"],
                    "height": frame["height"],
                    "encoding": frame["encoding"],
                    "size": frame["size"],
                    "elapsed_s": frame["elapsed_s"],
                    "packet_count": len(frame["packets"]),
                    "frame_path": frame_path.name,
                    "packets": frame["packets"],
                }
            )
            logger.info(
                f"  OK: {frame['width']}x{frame['height']} {frame['encoding']} "
                f"{frame['size']} bytes in {len(frame['packets'])} packets, "
                f"{frame['elapsed_s']:.3f}s"
            )
        except Exception as exc:
            result["error"] = repr(exc)
            logger.info(f"  FAIL: {exc}")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

        results.append(result)
        if attempt < args.attempts:
            time.sleep(args.delay)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "target": {"ip": args.ip, "port": args.port},
        "env": env_info,
        "results": results,
        "ok_count": sum(1 for r in results if r["ok"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("")
    logger.info("=== Summary ===")
    logger.info(f"Successful one-frame connections: {summary['ok_count']}/{args.attempts}")
    logger.info(f"Saved logs in: {run_dir}")
    logger.info("When you reconnect to normal WiFi, tell Codex the reconnect test is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
