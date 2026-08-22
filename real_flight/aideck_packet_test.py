#!/usr/bin/env python3.11
"""Offline AI Deck packet diagnostic.

This is meant for the WiFi pivot workflow:
1. Connect the VM/host WiFi to the AI Deck AP.
2. Run this script from the desktop launcher.
3. Reconnect to normal internet and inspect real_flight/aideck_logs/.

The test records CPX packet headers, payload sizes, frame-header metadata, and
the exact point where the stream stalls. It intentionally does not need any
internet connection.
"""

from __future__ import annotations

import argparse
import csv
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
    log_path = run_dir / "packet_test.log"
    logger = logging.getLogger("aideck_packet_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def parse_image_header(payload: bytes) -> dict | None:
    if len(payload) < 11 or payload[0] != IMG_HEADER_MAGIC:
        return None
    magic, width, height, depth, img_type, size = struct.unpack("<BHHBBI", payload[:11])
    return {
        "magic": magic,
        "width": width,
        "height": height,
        "depth": depth,
        "type": img_type,
        "encoding": "JPEG" if img_type == JPEG_ENCODING else "RAW",
        "size": size,
    }


def packet_rows_path(run_dir: Path) -> Path:
    return run_dir / "packets.csv"


def connect_to_deck(ip: str, port: int, timeout: float, logger: logging.Logger) -> socket.socket:
    logger.info(f"Connecting TCP {ip}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    logger.info("TCP connected.")
    return sock


def test_one_ip(args: argparse.Namespace, run_dir: Path, ip: str, logger: logging.Logger) -> dict:
    summary: dict = {
        "ip": ip,
        "port": args.port,
        "connected": False,
        "packets": 0,
        "bytes_total": 0,
        "image_headers": [],
        "completed_frames": 0,
        "stall": None,
        "error": None,
    }

    logger.info("")
    logger.info(f"=== Candidate {ip}:{args.port} ===")

    ping_cmd = ["ping", "-c", "2", "-W", "2", ip]
    rc, out, err = run_text(ping_cmd, timeout=6)
    summary["ping_returncode"] = rc
    summary["ping_stdout"] = out
    summary["ping_stderr"] = err
    logger.info(f"Ping return code: {rc}")
    if out:
        logger.info(out)

    raw_prefix = bytearray()
    current_frame = None
    frame_bytes = bytearray()
    frame_started_at = None
    first_header_saved = False

    try:
        sock = connect_to_deck(ip, args.port, args.connect_timeout, logger)
    except Exception as exc:
        summary["error"] = f"connect failed: {exc}"
        logger.info(f"Connect failed: {exc}")
        return summary

    summary["connected"] = True
    sock.settimeout(args.read_timeout)
    started = time.monotonic()
    last_packet_at = started

    with packet_rows_path(run_dir).open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "candidate_ip",
                "packet_index",
                "t_rel_s",
                "gap_s",
                "length_field",
                "payload_len",
                "routing",
                "function",
                "first16_hex",
                "is_image_header",
                "frame_bytes_after_packet",
                "frame_expected_bytes",
            ],
        )
        if csv_file.tell() == 0:
            writer.writeheader()

        try:
            while time.monotonic() - started < args.duration:
                before = time.monotonic()
                try:
                    packet_info = recv_exact(sock, 4)
                except socket.timeout:
                    waited = time.monotonic() - before
                    summary["stall"] = {
                        "reason": "timeout waiting for next CPX packet header",
                        "waited_s": round(waited, 3),
                        "after_packets": summary["packets"],
                        "active_frame_expected_bytes": current_frame["size"] if current_frame else None,
                        "active_frame_received_bytes": len(frame_bytes) if current_frame else None,
                    }
                    logger.info("")
                    logger.info("STALL: timed out waiting for the next CPX packet header.")
                    break

                length_field, routing, function = struct.unpack("<HBB", packet_info)
                payload_len = max(0, length_field - 2)
                payload = recv_exact(sock, payload_len) if payload_len else b""

                now = time.monotonic()
                summary["packets"] += 1
                summary["bytes_total"] += len(packet_info) + len(payload)
                gap_s = now - last_packet_at
                last_packet_at = now

                if len(raw_prefix) < args.raw_prefix_bytes:
                    remaining = args.raw_prefix_bytes - len(raw_prefix)
                    raw_prefix.extend((packet_info + payload)[:remaining])

                header = parse_image_header(payload)
                is_header = header is not None

                if is_header:
                    summary["image_headers"].append(
                        {
                            **header,
                            "packet_index": summary["packets"],
                            "t_rel_s": round(now - started, 3),
                        }
                    )
                    logger.info(
                        "Image header: "
                        f"{header['width']}x{header['height']} {header['encoding']} "
                        f"{header['size']} bytes at packet {summary['packets']}"
                    )
                    current_frame = header
                    frame_bytes = bytearray()
                    frame_started_at = now
                    if not first_header_saved:
                        (run_dir / "first_image_header_payload.bin").write_bytes(payload)
                        first_header_saved = True
                elif current_frame is not None:
                    frame_bytes.extend(payload)
                    if len(frame_bytes) >= current_frame["size"]:
                        summary["completed_frames"] += 1
                        frame_path = run_dir / f"frame_{summary['completed_frames']:03d}"
                        if current_frame["type"] == JPEG_ENCODING:
                            frame_path = frame_path.with_suffix(".jpg")
                        else:
                            frame_path = frame_path.with_suffix(".bin")
                        frame_path.write_bytes(bytes(frame_bytes[: current_frame["size"]]))
                        logger.info(
                            "Frame complete: "
                            f"{len(frame_bytes[: current_frame['size']])} bytes saved to {frame_path.name}"
                        )
                        current_frame = None
                        frame_bytes = bytearray()
                        frame_started_at = None
                        if summary["completed_frames"] >= args.max_frames:
                            logger.info(f"Reached max_frames={args.max_frames}; stopping.")
                            break

                if summary["packets"] <= args.print_packets or is_header:
                    logger.info(
                        f"Packet {summary['packets']:04d}: len={length_field} "
                        f"payload={payload_len} route=0x{routing:02x} func=0x{function:02x} "
                        f"gap={gap_s:.3f}s first={payload[:16].hex()}"
                    )

                writer.writerow(
                    {
                        "candidate_ip": ip,
                        "packet_index": summary["packets"],
                        "t_rel_s": f"{now - started:.6f}",
                        "gap_s": f"{gap_s:.6f}",
                        "length_field": length_field,
                        "payload_len": payload_len,
                        "routing": f"0x{routing:02x}",
                        "function": f"0x{function:02x}",
                        "first16_hex": payload[:16].hex(),
                        "is_image_header": int(is_header),
                        "frame_bytes_after_packet": len(frame_bytes) if current_frame else "",
                        "frame_expected_bytes": current_frame["size"] if current_frame else "",
                    }
                )

        except Exception as exc:
            summary["error"] = repr(exc)
            logger.exception(f"Error while reading stream: {exc}")
        finally:
            try:
                sock.close()
            except OSError:
                pass

    if raw_prefix:
        (run_dir / f"raw_prefix_{ip.replace('.', '_')}.bin").write_bytes(bytes(raw_prefix))

    if current_frame is not None:
        partial_path = run_dir / "partial_frame_payload.bin"
        partial_path.write_bytes(bytes(frame_bytes))
        elapsed = time.monotonic() - frame_started_at if frame_started_at else None
        summary["partial_frame"] = {
            "expected_bytes": current_frame["size"],
            "received_bytes": len(frame_bytes),
            "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            "path": partial_path.name,
        }
        logger.info(
            "Partial frame saved: "
            f"{len(frame_bytes)}/{current_frame['size']} bytes to {partial_path.name}"
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Deck packet-level stream diagnostic")
    parser.add_argument("--ip", action="append", default=None, help="AI Deck IP candidate")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--print-packets", type=int, default=25)
    parser.add_argument("--raw-prefix-bytes", type=int, default=262144)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    candidates = args.ip or ["192.168.4.1", "192.168.7.201"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"packet_test_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)

    logger.info("AI Deck packet test")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Candidates: {', '.join(candidates)}")
    logger.info(f"Duration per connected candidate: {args.duration:.1f}s")

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

    summaries = []
    for ip in candidates:
        result = test_one_ip(args, run_dir, ip, logger)
        summaries.append(result)
        if result.get("connected"):
            break

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "env": env_info,
        "results": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("")
    logger.info("=== Summary ===")
    for result in summaries:
        logger.info(
            f"{result['ip']}: connected={result['connected']} "
            f"packets={result['packets']} bytes={result['bytes_total']} "
            f"headers={len(result['image_headers'])} frames={result['completed_frames']} "
            f"stall={result['stall'] is not None} error={result['error']}"
        )
    logger.info("")
    logger.info(f"Saved logs in: {run_dir}")
    logger.info("When you reconnect to normal WiFi, tell Codex the test is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
