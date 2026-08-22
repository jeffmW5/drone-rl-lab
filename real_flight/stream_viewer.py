#!/usr/bin/env python3
"""AI Deck camera stream viewer.

Connects to the AI Deck ESP32 over WiFi (TCP port 5000) and displays
JPEG or raw grayscale frames from the GAP8 wifi-img-streamer firmware.

Auto-reconnects on stalls or connection drops.

Usage:
    python stream_viewer.py [--ip IP] [--port PORT] [--save-dir DIR]

Press Ctrl+C to stop.
"""
import argparse
import socket
import struct
import time
import io
import os
from PIL import Image

IMG_HEADER_MAGIC = 0xBC
RAW_ENCODING = 0
JPEG_ENCODING = 1

CAM_WIDTH = 324
CAM_HEIGHT = 244

PACKET_TIMEOUT = 3
CONNECT_TIMEOUT = 5
MAX_RETRIES = 0  # 0 = unlimited


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_cpx_packet(sock):
    length_bytes = recv_exact(sock, 2)
    length = struct.unpack("<H", length_bytes)[0]
    if length == 0:
        return b""
    payload = recv_exact(sock, length)
    return payload[2:]


def parse_img_header(data):
    if len(data) < 11 or data[0] != IMG_HEADER_MAGIC:
        return None
    magic, width, height, depth, img_type, size = struct.unpack("<BHHBBI", data[:11])
    return {
        "width": width,
        "height": height,
        "depth": depth,
        "type": img_type,
        "size": size,
    }


def connect(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(CONNECT_TIMEOUT)
    sock.connect((ip, port))
    sock.settimeout(PACKET_TIMEOUT)
    return sock


def receive_frame(sock):
    """Receive one complete frame. Returns (header, image_bytes) or raises on stall."""
    data = recv_cpx_packet(sock)
    while data == b"":
        data = recv_cpx_packet(sock)

    header = parse_img_header(data)
    if header is None:
        raise ValueError(f"Bad header: {data[:16].hex()}")

    img_size = header["size"]
    img_data = bytearray()

    while len(img_data) < img_size:
        chunk = recv_cpx_packet(sock)
        img_data.extend(chunk)

    return header, bytes(img_data[:img_size])


def main():
    parser = argparse.ArgumentParser(description="AI Deck camera viewer")
    parser.add_argument("--ip", default="192.168.7.201")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    frame_count = 0
    drop_count = 0
    start_time = time.time()
    retries = 0
    sock = None

    try:
        while True:
            # Connect / reconnect
            if sock is None:
                try:
                    print(f"Connecting to {args.ip}:{args.port}...", flush=True)
                    sock = connect(args.ip, args.port)
                    print("Connected.", flush=True)
                    retries = 0
                except (OSError, ConnectionError) as e:
                    retries += 1
                    if MAX_RETRIES and retries > MAX_RETRIES:
                        print(f"\nGave up after {retries - 1} retries.")
                        break
                    wait = min(2 ** retries, 10)
                    print(f"Connect failed: {e} — retry in {wait}s", flush=True)
                    time.sleep(wait)
                    continue

            # Try to receive a frame
            try:
                header, img_data = receive_frame(sock)
            except (socket.timeout, ConnectionError, OSError, ValueError) as e:
                drop_count += 1
                print(f"\nDrop #{drop_count}: {e} — reconnecting", flush=True)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                time.sleep(0.5)
                continue

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            enc = "JPEG" if header["type"] == JPEG_ENCODING else "RAW"
            print(f"\rFrame {frame_count} | {header['width']}x{header['height']} | "
                  f"{enc} | {len(img_data)} bytes | {fps:.1f} FPS | drops={drop_count}", end="", flush=True)

            if header["type"] == JPEG_ENCODING:
                try:
                    img = Image.open(io.BytesIO(img_data))
                except Exception as e:
                    print(f"\nJPEG decode error: {e}")
                    continue
            else:
                if len(img_data) >= header["width"] * header["height"]:
                    img = Image.frombytes("L", (header["width"], header["height"]), img_data)
                else:
                    print(f"\nShort frame: got {len(img_data)}, expected {header['width'] * header['height']}")
                    continue

            if args.save_dir:
                img.save(os.path.join(args.save_dir, f"frame_{frame_count:04d}.png"))

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nStopped. {frame_count} frames, {drop_count} drops in {elapsed:.1f}s ({fps:.1f} FPS)")
    finally:
        if sock:
            sock.close()


if __name__ == "__main__":
    main()
