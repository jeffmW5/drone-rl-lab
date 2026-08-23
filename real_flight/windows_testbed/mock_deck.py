"""Mock AI Deck CPX streamer, for validating the test bed without hardware.

Serves CPX-framed JPEG frames on a TCP port using the same wire format as the
GAP8 wifi-img-streamer. By default it replays a real captured frame from
``real_flight/aideck_logs``.

It can also reproduce the observed failure so the tests can be checked against
a known-bad stream:

  py -3 mock_deck.py --port 5555                      # healthy stream
  py -3 mock_deck.py --port 5555 --stall-after 1      # header then silence
  py -3 mock_deck.py --port 5555 --stall-mid-frame 2  # partial second frame

This is a test fixture. It is not part of the diagnostic path.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aideck_core import (
    IMG_HEADER_MAGIC,
    IMG_HEADER_STRUCT,
    JPEG_ENCODING,
    LAB_DIR,
)

CPX_ROUTING = 0x0B
CPX_FUNCTION = 0x09
MAX_PAYLOAD = 1022  # CPX length field is 16-bit; the streamer chunks well below it


def find_captured_frame() -> tuple[bytes, int, int]:
    """Prefer a real captured deck frame; fall back to a generated JPEG."""
    logs = LAB_DIR / "real_flight" / "aideck_logs"
    for path in sorted(logs.glob("*/frame_*.jpg")) + sorted(logs.glob("*/attempt_*.jpg")):
        data = path.read_bytes()
        if data[:2] == b"\xff\xd8":
            print(f"Using captured frame: {path.relative_to(LAB_DIR)} ({len(data)} bytes)")
            return data, 324, 244
    try:
        import io

        from PIL import Image

        image = Image.new("L", (324, 244), color=128)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
        print(f"Using generated JPEG ({len(data)} bytes)")
        return data, 324, 244
    except ImportError:
        raise SystemExit("No captured frame found and Pillow is not installed.")


def cpx_packet(payload: bytes) -> bytes:
    return struct.pack("<HBB", len(payload) + 2, CPX_ROUTING, CPX_FUNCTION) + payload


def image_header(width: int, height: int, size: int) -> bytes:
    return struct.pack(IMG_HEADER_STRUCT, IMG_HEADER_MAGIC, width, height, 1,
                       JPEG_ENCODING, size)


def serve_one(conn: socket.socket, args, frame: bytes, width: int, height: int) -> None:
    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    sent_frames = 0
    while args.frames == 0 or sent_frames < args.frames:
        if args.stall_after and sent_frames >= args.stall_after:
            conn.sendall(cpx_packet(image_header(width, height, len(frame))))
            print(f"  sent header {sent_frames + 1} then stalling (holding socket open)")
            while True:
                time.sleep(1.0)

        conn.sendall(cpx_packet(image_header(width, height, len(frame))))

        stall_mid = args.stall_mid_frame and (sent_frames + 1) == args.stall_mid_frame
        sent_bytes = 0
        for offset in range(0, len(frame), args.chunk):
            chunk = frame[offset:offset + args.chunk]
            if stall_mid and sent_bytes >= len(frame) // 3:
                print(f"  frame {sent_frames + 1}: stalling at "
                      f"{sent_bytes}/{len(frame)} bytes")
                while True:
                    time.sleep(1.0)
            conn.sendall(cpx_packet(chunk))
            sent_bytes += len(chunk)

        sent_frames += 1
        print(f"  sent frame {sent_frames} ({len(frame)} bytes)")
        if interval:
            time.sleep(interval)

    print("  frame budget reached; closing connection")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock AI Deck CPX streamer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--chunk", type=int, default=MAX_PAYLOAD)
    parser.add_argument("--frames", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--stall-after", type=int, default=0,
                        help="send N frames, then a header and nothing else")
    parser.add_argument("--stall-mid-frame", type=int, default=0,
                        help="stall partway through frame N")
    parser.add_argument("--once", action="store_true", help="serve one connection then exit")
    args = parser.parse_args()

    frame, width, height = find_captured_frame()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Mock deck listening on {args.host}:{args.port}")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Connection from {addr[0]}:{addr[1]}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                serve_one(conn, args, frame, width, height)
            except (ConnectionError, OSError) as exc:
                print(f"  connection ended: {exc!r}")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
