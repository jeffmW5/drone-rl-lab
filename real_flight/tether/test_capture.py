"""Protocol checks without a drone or network connection."""
import socket
import struct
import unittest
from capture import frame


def wire(payload):
    return struct.pack('<HBB', len(payload) + 2, 0, 0) + payload


class CaptureTest(unittest.TestCase):
    def decode(self, message):
        sender, receiver = socket.socketpair()
        try:
            sender.sendall(message)
            sender.shutdown(socket.SHUT_WR)
            receiver.settimeout(1)
            return frame(receiver)
        finally:
            sender.close(); receiver.close()

    def test_raw(self):
        header = struct.pack('<BHHBBI', 0xBC, 2, 2, 1, 0, 4)
        image = self.decode(wire(header) + wire(bytes([0, 64])) + wire(bytes([128, 255])))
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(list(image.getdata()), [0, 64, 128, 255])

    def test_disconnect(self):
        header = struct.pack('<BHHBBI', 0xBC, 2, 2, 1, 0, 4)
        with self.assertRaises(EOFError):
            self.decode(wire(header))

    def test_oversize_chunk(self):
        header = struct.pack('<BHHBBI', 0xBC, 2, 2, 1, 0, 4)
        with self.assertRaises(ValueError):
            self.decode(wire(header) + wire(bytes(5)))


if __name__ == '__main__':
    unittest.main()
