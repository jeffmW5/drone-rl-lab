"""The four AI Deck test bed tests.

Ported from the Linux diagnostics. The protocol logic is unchanged; the shell
commands are the Windows equivalents and every test is cancellable and
callback-driven so a GUI can drive it without freezing.

Each test writes its artifacts into ``run_dir`` and returns a summary dict.
"""

from __future__ import annotations

import csv
import socket
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from aideck_core import (
    JPEG_ENCODING,
    CancelToken,
    Cancelled,
    Reporter,
    collect_env,
    connect_to_deck,
    frame_suffix,
    monotonic,
    parse_image_header,
    ping,
    read_packet,
    stamp_now,
    write_summary,
)

PACKET_CSV_FIELDS = [
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
]


@dataclass
class TestConfig:
    """Everything the four tests can be tuned with."""

    ips: list[str] = field(default_factory=lambda: ["192.168.4.1"])
    port: int = 5000
    connect_timeout: float = 5.0
    read_timeout: float = 6.0
    # packet test
    duration: float = 45.0
    max_frames: int = 2          # 0 = unlimited
    print_packets: int = 25
    raw_prefix_bytes: int = 262144
    # reconnect test
    attempts: int = 8
    delay: float = 1.0
    # throughput test
    throughput_duration: float = 120.0
    save_every: int = 25
    stall_gap_s: float = 3.0


def _log_env(reporter: Reporter) -> dict:
    env = collect_env()
    wlan = env.get("netsh_wlan", {}).get("parsed", {})
    if wlan:
        reporter.info(
            f"WLAN: ssid={wlan.get('ssid', '?')} state={wlan.get('state', '?')} "
            f"signal={wlan.get('signal', '?')} "
            f"rx={wlan.get('receive rate (mbps)', '?')}Mbps "
            f"tx={wlan.get('transmit rate (mbps)', '?')}Mbps"
        )
    else:
        reporter.info("WLAN: no wireless interface reported by netsh.")
    return env


# --- 1. link check -----------------------------------------------------------

def link_check(
    cfg: TestConfig,
    run_dir: Path,
    reporter: Reporter,
    cancel: CancelToken | None = None,
) -> dict:
    """Adapter state, IP, route and ping. Sanity gate before a real run."""
    cancel = cancel or CancelToken()
    reporter.info("=== Link check ===")
    env = _log_env(reporter)

    results = []
    for ip in cfg.ips:
        cancel.check()
        reporter.info("")
        reporter.info(f"Pinging {ip} ...")
        ping_result = ping(ip)
        reporter.info(ping_result["stdout"] or ping_result["stderr"] or "(no output)")

        tcp: dict = {"ok": False, "error": None, "connect_s": None}
        started = monotonic()
        try:
            sock = connect_to_deck(ip, cfg.port, cfg.connect_timeout)
            tcp["ok"] = True
            tcp["connect_s"] = round(monotonic() - started, 3)
            sock.close()
            reporter.info(f"TCP {ip}:{cfg.port} OK in {tcp['connect_s']}s")
        except Exception as exc:  # noqa: BLE001
            tcp["error"] = repr(exc)
            reporter.info(f"TCP {ip}:{cfg.port} failed: {exc}")

        results.append({"ip": ip, "ping": ping_result, "tcp": tcp})

    reachable = [r["ip"] for r in results if r["tcp"]["ok"]]
    reporter.info("")
    reporter.info(f"Reachable on TCP {cfg.port}: {reachable or 'none'}")

    summary = {
        "test": "link_check",
        "created_at": stamp_now(),
        "run_dir": str(run_dir),
        "env": env,
        "results": results,
        "reachable": reachable,
        "ok": bool(reachable),
    }
    write_summary(run_dir, summary)
    return summary


# --- 2. packet test ----------------------------------------------------------

def _packet_test_one_ip(
    cfg: TestConfig,
    run_dir: Path,
    ip: str,
    reporter: Reporter,
    cancel: CancelToken,
    on_frame=None,
) -> dict:
    summary: dict = {
        "ip": ip,
        "port": cfg.port,
        "connected": False,
        "packets": 0,
        "bytes_total": 0,
        "image_headers": [],
        "completed_frames": 0,
        "stall": None,
        "error": None,
    }

    reporter.info("")
    reporter.info(f"=== Candidate {ip}:{cfg.port} ===")

    ping_result = ping(ip)
    summary["ping_returncode"] = ping_result["returncode"]
    summary["ping_stdout"] = ping_result["stdout"]
    summary["ping_stderr"] = ping_result["stderr"]
    reporter.info(f"Ping return code: {ping_result['returncode']}")

    raw_prefix = bytearray()
    current_frame = None
    frame_bytes = bytearray()
    frame_started_at = None
    first_header_saved = False

    try:
        reporter.info(f"Connecting TCP {ip}:{cfg.port} ...")
        sock = connect_to_deck(ip, cfg.port, cfg.connect_timeout)
        reporter.info("TCP connected.")
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"connect failed: {exc}"
        reporter.info(f"Connect failed: {exc}")
        return summary

    summary["connected"] = True
    sock.settimeout(cfg.read_timeout)
    started = monotonic()
    last_packet_at = started

    with (run_dir / "packets.csv").open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PACKET_CSV_FIELDS)
        if csv_file.tell() == 0:
            writer.writeheader()

        try:
            while monotonic() - started < cfg.duration:
                cancel.check()
                before = monotonic()
                try:
                    length_field, routing, function, payload = read_packet(sock)
                except socket.timeout:
                    waited = monotonic() - before
                    summary["stall"] = {
                        "reason": "timeout waiting for next CPX packet header",
                        "waited_s": round(waited, 3),
                        "after_packets": summary["packets"],
                        "active_frame_expected_bytes": (
                            current_frame["size"] if current_frame else None
                        ),
                        "active_frame_received_bytes": (
                            len(frame_bytes) if current_frame else None
                        ),
                    }
                    reporter.info("")
                    reporter.info("STALL: timed out waiting for the next CPX packet.")
                    break

                now = monotonic()
                payload_len = len(payload)
                summary["packets"] += 1
                summary["bytes_total"] += 4 + payload_len
                gap_s = now - last_packet_at
                last_packet_at = now

                if len(raw_prefix) < cfg.raw_prefix_bytes:
                    remaining = cfg.raw_prefix_bytes - len(raw_prefix)
                    raw_prefix.extend(payload[:remaining])

                header = parse_image_header(payload)
                is_header = header is not None

                # Printed before the header/frame handling below so the log
                # reads chronologically.
                if summary["packets"] <= cfg.print_packets or is_header:
                    reporter.info(
                        f"Packet {summary['packets']:04d}: len={length_field} "
                        f"payload={payload_len} route=0x{routing:02x} "
                        f"func=0x{function:02x} gap={gap_s:.3f}s "
                        f"first={payload[:16].hex()}"
                    )

                if is_header:
                    summary["image_headers"].append(
                        {**header, "packet_index": summary["packets"],
                         "t_rel_s": round(now - started, 3)}
                    )
                    reporter.info(
                        f"Image header: {header['width']}x{header['height']} "
                        f"{header['encoding']} {header['size']} bytes "
                        f"at packet {summary['packets']}"
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
                        data = bytes(frame_bytes[: current_frame["size"]])
                        frame_path = (
                            run_dir
                            / f"frame_{summary['completed_frames']:03d}"
                            f"{frame_suffix(current_frame['type'])}"
                        )
                        frame_path.write_bytes(data)
                        reporter.info(
                            f"Frame complete: {len(data)} bytes saved to {frame_path.name}"
                        )
                        if on_frame is not None and current_frame["type"] == JPEG_ENCODING:
                            on_frame(data, dict(current_frame))
                        current_frame = None
                        frame_bytes = bytearray()
                        frame_started_at = None
                        if cfg.max_frames and summary["completed_frames"] >= cfg.max_frames:
                            reporter.info(f"Reached max_frames={cfg.max_frames}; stopping.")
                            break

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
        except Cancelled:
            summary["error"] = "cancelled by operator"
            reporter.info("Cancelled by operator.")
        except Exception as exc:  # noqa: BLE001
            summary["error"] = repr(exc)
            reporter.info(f"Error while reading stream: {exc!r}")
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
        elapsed = monotonic() - frame_started_at if frame_started_at else None
        summary["partial_frame"] = {
            "expected_bytes": current_frame["size"],
            "received_bytes": len(frame_bytes),
            "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            "path": partial_path.name,
        }
        reporter.info(
            f"Partial frame saved: {len(frame_bytes)}/{current_frame['size']} "
            f"bytes to {partial_path.name}"
        )

    return summary


def packet_test(
    cfg: TestConfig,
    run_dir: Path,
    reporter: Reporter,
    cancel: CancelToken | None = None,
    on_frame=None,
) -> dict:
    """CPX packet headers, payload sizes, gaps, and the exact stall point."""
    cancel = cancel or CancelToken()
    reporter.info("=== Packet test ===")
    reporter.info(f"Run directory: {run_dir}")
    reporter.info(f"Candidates: {', '.join(cfg.ips)}")
    reporter.info(f"Duration per connected candidate: {cfg.duration:.1f}s")
    env = _log_env(reporter)

    results = []
    for ip in cfg.ips:
        result = _packet_test_one_ip(cfg, run_dir, ip, reporter, cancel, on_frame)
        results.append(result)
        if result.get("connected"):
            break

    summary = {
        "test": "packet_test",
        "created_at": stamp_now(),
        "run_dir": str(run_dir),
        "env": env,
        "results": results,
    }
    write_summary(run_dir, summary)

    reporter.info("")
    reporter.info("=== Summary ===")
    for result in results:
        reporter.info(
            f"{result['ip']}: connected={result['connected']} "
            f"packets={result['packets']} bytes={result['bytes_total']} "
            f"headers={len(result['image_headers'])} "
            f"frames={result['completed_frames']} "
            f"stall={result['stall'] is not None} error={result['error']}"
        )
    return summary


# --- 3. reconnect test -------------------------------------------------------

def _read_one_frame(sock: socket.socket, read_timeout: float, max_packets: int = 100) -> dict:
    sock.settimeout(read_timeout)
    started = monotonic()
    packets = []

    length, routing, function, payload = read_packet(sock)
    packets.append(
        {
            "length": length,
            "payload_len": len(payload),
            "routing": routing,
            "function": function,
            "first16_hex": payload[:16].hex(),
            "t_rel_s": round(monotonic() - started, 6),
        }
    )

    header = parse_image_header(payload)
    if header is None:
        raise ValueError(f"first packet is not an image header: {payload[:16].hex()}")

    img = bytearray()
    while len(img) < header["size"]:
        if len(packets) >= max_packets:
            raise TimeoutError(
                f"max packet count reached with {len(img)}/{header['size']} frame bytes"
            )
        length, routing, function, payload = read_packet(sock)
        img.extend(payload)
        packets.append(
            {
                "length": length,
                "payload_len": len(payload),
                "routing": routing,
                "function": function,
                "first16_hex": payload[:16].hex(),
                "frame_bytes": len(img),
                "t_rel_s": round(monotonic() - started, 6),
            }
        )

    return {
        **header,
        "packets": packets,
        "frame_bytes": bytes(img[: header["size"]]),
        "elapsed_s": round(monotonic() - started, 6),
    }


def reconnect_test(
    cfg: TestConfig,
    run_dir: Path,
    reporter: Reporter,
    cancel: CancelToken | None = None,
    on_frame=None,
) -> dict:
    """N fresh TCP connections, one complete frame each, pass/fail per attempt."""
    cancel = cancel or CancelToken()
    ip = cfg.ips[0]
    reporter.info("=== Reconnect test ===")
    reporter.info(f"Run directory: {run_dir}")
    reporter.info(f"Target: {ip}:{cfg.port}")
    reporter.info(f"Attempts: {cfg.attempts}, delay={cfg.delay}s")
    env = _log_env(reporter)

    results = []
    for attempt in range(1, cfg.attempts + 1):
        try:
            cancel.check()
        except Cancelled:
            reporter.info("Cancelled by operator.")
            break

        reporter.info("")
        reporter.info(f"Attempt {attempt}/{cfg.attempts}: connecting...")
        result: dict = {"attempt": attempt, "ok": False, "error": None}
        sock = None
        try:
            sock = connect_to_deck(ip, cfg.port, cfg.connect_timeout)
            reporter.info("  connected; reading one complete frame...")
            frame = _read_one_frame(sock, cfg.read_timeout)
            frame_path = run_dir / f"attempt_{attempt:02d}{frame_suffix(frame['type'])}"
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
            reporter.info(
                f"  OK: {frame['width']}x{frame['height']} {frame['encoding']} "
                f"{frame['size']} bytes in {len(frame['packets'])} packets, "
                f"{frame['elapsed_s']:.3f}s"
            )
            if on_frame is not None and frame["type"] == JPEG_ENCODING:
                on_frame(frame["frame_bytes"], {k: frame[k] for k in
                                                ("width", "height", "size", "encoding", "type")})
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
            reporter.info(f"  FAIL: {exc}")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

        results.append(result)
        if attempt < cfg.attempts:
            _wait(cfg.delay, cancel)

    ok_count = sum(1 for r in results if r["ok"])
    summary = {
        "test": "reconnect_test",
        "created_at": stamp_now(),
        "run_dir": str(run_dir),
        "target": {"ip": ip, "port": cfg.port},
        "env": env,
        "results": results,
        "ok_count": ok_count,
        "attempts_run": len(results),
    }
    write_summary(run_dir, summary)

    reporter.info("")
    reporter.info("=== Summary ===")
    reporter.info(f"Successful one-frame connections: {ok_count}/{len(results)}")
    return summary


def _wait(seconds: float, cancel: CancelToken) -> None:
    """Sleep in short slices so a cancel is noticed promptly."""
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        if cancel.cancelled:
            return
        time.sleep(0.05)


# --- 4. sustained throughput -------------------------------------------------

def throughput_test(
    cfg: TestConfig,
    run_dir: Path,
    reporter: Reporter,
    cancel: CancelToken | None = None,
    on_frame=None,
    on_progress=None,
) -> dict:
    """Long continuous read. This is the 'is it actually fixed' test.

    The ported scripts stop at ``max_frames=2`` and cannot show a stall that
    first appears at frame 40.
    """
    cancel = cancel or CancelToken()
    ip = cfg.ips[0]
    reporter.info("=== Sustained throughput test ===")
    reporter.info(f"Run directory: {run_dir}")
    reporter.info(f"Target: {ip}:{cfg.port}")
    reporter.info(f"Duration: {cfg.throughput_duration:.0f}s, "
                  f"saving every {cfg.save_every} frame(s)")
    env = _log_env(reporter)

    summary: dict = {
        "test": "throughput_test",
        "created_at": stamp_now(),
        "run_dir": str(run_dir),
        "target": {"ip": ip, "port": cfg.port},
        "env": env,
        "connected": False,
        "frames": 0,
        "bytes_total": 0,
        "packets": 0,
        "headers": 0,
        "stall": None,
        "error": None,
    }

    try:
        reporter.info(f"Connecting TCP {ip}:{cfg.port} ...")
        sock = connect_to_deck(ip, cfg.port, cfg.connect_timeout)
        reporter.info("TCP connected.")
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"connect failed: {exc}"
        reporter.info(f"Connect failed: {exc}")
        write_summary(run_dir, summary)
        return summary

    summary["connected"] = True
    sock.settimeout(cfg.read_timeout)

    frame_records: list[dict] = []
    current_frame = None
    frame_bytes = bytearray()
    started = monotonic()
    last_frame_at = started
    last_report_at = started

    frames_csv = (run_dir / "frames.csv").open("w", newline="", encoding="utf-8")
    frame_writer = csv.DictWriter(
        frames_csv,
        fieldnames=["frame_index", "t_rel_s", "interval_s", "size_bytes",
                    "packets_in_frame", "width", "height", "encoding"],
    )
    frame_writer.writeheader()
    packets_in_frame = 0

    try:
        while monotonic() - started < cfg.throughput_duration:
            cancel.check()
            before = monotonic()
            try:
                _length, _routing, _function, payload = read_packet(sock)
            except socket.timeout:
                waited = monotonic() - before
                summary["stall"] = {
                    "reason": "timeout waiting for next CPX packet",
                    "waited_s": round(waited, 3),
                    "after_frames": summary["frames"],
                    "after_packets": summary["packets"],
                    "active_frame_expected_bytes": (
                        current_frame["size"] if current_frame else None
                    ),
                    "active_frame_received_bytes": (
                        len(frame_bytes) if current_frame else None
                    ),
                    "t_rel_s": round(monotonic() - started, 3),
                }
                reporter.info("")
                reporter.info(
                    f"STALL after {summary['frames']} frames "
                    f"({round(monotonic() - started, 1)}s in)."
                )
                break

            now = monotonic()
            summary["packets"] += 1
            summary["bytes_total"] += 4 + len(payload)
            packets_in_frame += 1

            header = parse_image_header(payload)
            if header is not None:
                summary["headers"] += 1
                current_frame = header
                frame_bytes = bytearray()
                packets_in_frame = 1
                continue

            if current_frame is None:
                continue

            frame_bytes.extend(payload)
            if len(frame_bytes) < current_frame["size"]:
                continue

            summary["frames"] += 1
            interval = now - last_frame_at
            last_frame_at = now
            data = bytes(frame_bytes[: current_frame["size"]])

            frame_records.append(
                {
                    "frame_index": summary["frames"],
                    "t_rel_s": round(now - started, 6),
                    "interval_s": round(interval, 6),
                    "size_bytes": len(data),
                    "packets_in_frame": packets_in_frame,
                }
            )
            frame_writer.writerow(
                {
                    "frame_index": summary["frames"],
                    "t_rel_s": f"{now - started:.6f}",
                    "interval_s": f"{interval:.6f}",
                    "size_bytes": len(data),
                    "packets_in_frame": packets_in_frame,
                    "width": current_frame["width"],
                    "height": current_frame["height"],
                    "encoding": current_frame["encoding"],
                }
            )
            frames_csv.flush()

            if cfg.save_every and (summary["frames"] - 1) % cfg.save_every == 0:
                path = (
                    run_dir
                    / f"frame_{summary['frames']:05d}{frame_suffix(current_frame['type'])}"
                )
                path.write_bytes(data)

            if on_frame is not None and current_frame["type"] == JPEG_ENCODING:
                on_frame(data, dict(current_frame))

            if interval >= cfg.stall_gap_s:
                reporter.info(
                    f"Long gap: {interval:.2f}s before frame {summary['frames']}"
                )

            if now - last_report_at >= 5.0:
                elapsed = now - started
                fps = summary["frames"] / elapsed if elapsed > 0 else 0.0
                rate = summary["bytes_total"] / elapsed / 1024 if elapsed > 0 else 0.0
                reporter.info(
                    f"t={elapsed:6.1f}s frames={summary['frames']:5d} "
                    f"fps={fps:5.2f} rate={rate:7.1f} KiB/s"
                )
                last_report_at = now
                if on_progress is not None:
                    on_progress(elapsed, summary["frames"], fps, rate)

            current_frame = None
            frame_bytes = bytearray()
            packets_in_frame = 0

    except Cancelled:
        summary["error"] = "cancelled by operator"
        reporter.info("Cancelled by operator.")
    except Exception as exc:  # noqa: BLE001
        summary["error"] = repr(exc)
        reporter.info(f"Error while reading stream: {exc!r}")
    finally:
        frames_csv.close()
        try:
            sock.close()
        except OSError:
            pass

    elapsed = monotonic() - started
    intervals = [r["interval_s"] for r in frame_records[1:]]
    summary["elapsed_s"] = round(elapsed, 3)
    summary["fps"] = round(summary["frames"] / elapsed, 3) if elapsed > 0 else 0.0
    summary["bytes_per_s"] = round(summary["bytes_total"] / elapsed, 1) if elapsed > 0 else 0.0
    summary["kib_per_s"] = round(summary["bytes_per_s"] / 1024, 2)
    summary["frame_intervals"] = {
        "count": len(intervals),
        "mean_s": round(statistics.fmean(intervals), 4) if intervals else None,
        "median_s": round(statistics.median(intervals), 4) if intervals else None,
        "max_s": round(max(intervals), 4) if intervals else None,
        "min_s": round(min(intervals), 4) if intervals else None,
        "over_stall_gap": sum(1 for i in intervals if i >= cfg.stall_gap_s),
        "stall_gap_s": cfg.stall_gap_s,
    }
    summary["frame_records"] = frame_records
    write_summary(run_dir, summary)

    reporter.info("")
    reporter.info("=== Summary ===")
    reporter.info(
        f"frames={summary['frames']} in {summary['elapsed_s']}s -> "
        f"{summary['fps']} fps, {summary['kib_per_s']} KiB/s"
    )
    if intervals:
        reporter.info(
            f"frame interval mean={summary['frame_intervals']['mean_s']}s "
            f"median={summary['frame_intervals']['median_s']}s "
            f"max={summary['frame_intervals']['max_s']}s "
            f"gaps>={cfg.stall_gap_s}s: {summary['frame_intervals']['over_stall_gap']}"
        )
    reporter.info(f"stalled={summary['stall'] is not None} error={summary['error']}")
    return summary


TESTS = {
    "link": ("Link check", link_check, "link_check"),
    "packet": ("Packet test", packet_test, "packet_test"),
    "reconnect": ("Reconnect test", reconnect_test, "reconnect_test"),
    "throughput": ("Sustained throughput", throughput_test, "throughput_test"),
}
