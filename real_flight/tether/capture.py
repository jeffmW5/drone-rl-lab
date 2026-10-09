"""Record CPX-streamed AI-deck grayscale frames; no control packets sent.

Protocol fields follow Bitcraze's wifi-img-streamer example. Host timestamps
measure receipt, not sensor exposure. Raw color Bayer cameras require a
separately verified demosaic path; this recorder targets the grayscale deck.
"""
import argparse
import io
import json
from pathlib import Path
import socket
import struct
import time
from PIL import Image


def exact(sock, count):
    result = bytearray()
    while len(result) < count:
        chunk = sock.recv(count - len(result))
        if not chunk:
            raise EOFError('AI-deck disconnected')
        result.extend(chunk)
    return bytes(result)


def packet(sock):
    length, route, function = struct.unpack('<HBB', exact(sock, 4))
    if not 2 <= length <= 65535:
        raise ValueError('Invalid CPX length')
    return exact(sock, length - 2)


def frame(sock):
    header = packet(sock)
    if len(header) != 11:
        raise ValueError('Expected 11-byte image header')
    magic, width, height, depth, encoding, size = struct.unpack('<BHHBBI', header)
    if magic != 0xBC or not (1 <= width <= 1024 and 1 <= height <= 1024) or not (0 < size <= 2097152):
        raise ValueError('Invalid image metadata')
    data = bytearray()
    while len(data) < size:
        chunk = packet(sock)
        if not chunk or len(data) + len(chunk) > size:
            raise ValueError('Invalid image chunk length')
        data.extend(chunk)
    if encoding == 0:
        if depth != 1 or size != width * height:
            raise ValueError('Only 8-bit raw grayscale is supported')
        return Image.frombytes('L', (width, height), bytes(data))
    if encoding == 1:
        image = Image.open(io.BytesIO(data))
        image.load()
        if image.size != (width, height):
            raise ValueError('JPEG size differs from header')
        return image.convert('L')
    raise ValueError('Unknown encoding')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='192.168.4.1')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        with (args.output / 'frames.jsonl').open('w') as metadata:
            for i in range(args.frames):
                image = frame(sock)
                filename = f'{i:06d}.png'
                image.save(args.output / filename)
                metadata.write(json.dumps({'file': filename, 'host_received_ns': time.time_ns(), 'size': image.size}) + '\n')
                metadata.flush()
                print(filename, flush=True)


if __name__ == '__main__':
    main()
