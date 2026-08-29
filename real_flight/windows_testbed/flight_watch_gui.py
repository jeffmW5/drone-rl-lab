"""Flight watch -- watch and log a Crazyflie test flight from Windows.

Four steps:

    1. Radio      find the Crazyradio and the drone
    2. Pre-flight read battery, decks and estimator, then RELEASE the radio
    3. Watch      record the AI Deck camera while you fly
    4. Result     session folder, REPORT.md, zip

Safety property this tool is built around: **it never commands the drone.**
It reads telemetry in step 2, then hands the radio back before step 3, so
``fly.py`` owns the radio for the whole flight. Two programs cannot share one
Crazyradio, and a viewer that held it would block the flight script.

The camera arrives over the AI Deck's own WiFi, which is a separate link from
the Crazyradio, so recording video does not compete with flight control.

cflib is optional. Without it steps 1 and 2 degrade to "skipped" and the camera
recording still works.
"""

from __future__ import annotations

import csv
import queue
import socket
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

import aideck_core as core

try:
    import io

    from PIL import Image, ImageTk

    PIL_OK = True
except ImportError:  # pragma: no cover - depends on host install
    PIL_OK = False

NAVY = "#2C3E5C"
INK = "#2B2B2B"
MIST = "#D5E2EC"
PAPER = "#F5F5F0"
GOOD = "#1E6B3A"
BAD = "#8C2F2F"
WARN = "#8A6D1F"
MUTED = "#6B7280"

STEPS = ("Radio", "Pre-flight", "Watch", "Result")

# The config cutoff is 3.2V, but starting a flight below 3.7V leaves almost no
# margin once the motors load the pack.
VBAT_GOOD = 3.9
VBAT_LOW = 3.7


def cflib_available() -> tuple[bool, str]:
    try:
        import cflib.crtp  # noqa: F401

        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def scan_drones() -> list[str]:
    import cflib.crtp

    cflib.crtp.init_drivers()
    return [uri for uri, _ in cflib.crtp.scan_interfaces()]


def preflight(uri: str, cache: str) -> dict:
    """Connect, read state, disconnect. Never sends a setpoint."""
    import time

    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cflib.crtp.init_drivers()
    out: dict = {"uri": uri, "decks": {}, "telemetry": {}, "params": {}}
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache)) as scf:
        cf = scf.cf
        time.sleep(1.0)
        for name in cf.param.toc.toc.get("deck", {}):
            try:
                out["decks"][name] = int(cf.param.get_value("deck." + name))
            except Exception:  # noqa: BLE001
                pass
        for p in ("stabilizer.estimator", "stabilizer.controller"):
            try:
                out["params"][p] = cf.param.get_value(p)
            except Exception:  # noqa: BLE001
                pass
        lg = LogConfig(name="pre", period_in_ms=100)
        for v in ("pm.vbat", "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                  "range.zrange", "kalman.varPX", "kalman.varPY"):
            try:
                lg.add_variable(v, "float")
            except Exception:  # noqa: BLE001
                pass
        cf.log.add_config(lg)
        vals: dict = {}
        lg.data_received_cb.add_callback(lambda ts, d, c: vals.update(d))
        lg.start()
        time.sleep(2.5)
        lg.stop()
        out["telemetry"] = {k: round(float(v), 4) for k, v in vals.items()}
    return out


class WatchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Flight Watch -- Crazyflie test recorder")
        root.geometry("1020x700")
        root.configure(bg=PAPER)

        self.step = 0
        self.events: queue.Queue = queue.Queue()
        self.cancel = core.CancelToken()
        self.session_dir: Path | None = None
        self.uris: list[str] = []
        self.pre: dict | None = None
        self.watch_summary: dict | None = None
        self.photo = None

        self._fonts()
        self._style()
        self._vars()
        self._build()
        self._goto(0)
        self._pump()
        self._refresh_wifi()

    # -- chrome ------------------------------------------------------------

    def _fonts(self) -> None:
        self.f_h1 = tkfont.Font(family="Segoe UI Semibold", size=17)
        self.f_h2 = tkfont.Font(family="Segoe UI Semibold", size=12)
        self.f_body = tkfont.Font(family="Segoe UI", size=10)
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_stat = tkfont.Font(family="Consolas", size=13)

    def _style(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TFrame", background=PAPER)
        s.configure("Card.TFrame", background="white", relief="flat")
        s.configure("Rail.TFrame", background=MIST)
        s.configure("Banner.TFrame", background=MIST)
        s.configure("TLabel", background=PAPER, foreground=INK, font=self.f_body)
        s.configure("Card.TLabel", background="white", foreground=INK, font=self.f_body)
        s.configure("H1.TLabel", background=PAPER, foreground=NAVY, font=self.f_h1)
        s.configure("H2.TLabel", background="white", foreground=NAVY, font=self.f_h2)
        s.configure("Muted.TLabel", background=PAPER, foreground=MUTED, font=self.f_small)
        s.configure("CardMuted.TLabel", background="white", foreground=MUTED, font=self.f_small)
        s.configure("Good.TLabel", background="white", foreground=GOOD, font=self.f_h2)
        s.configure("Bad.TLabel", background="white", foreground=BAD, font=self.f_h2)
        s.configure("Warn.TLabel", background="white", foreground=WARN, font=self.f_h2)
        s.configure("Banner.TLabel", background=MIST, foreground=NAVY, font=self.f_h2)
        s.configure("Rail.TLabel", background=MIST, foreground=INK, font=self.f_body)
        s.configure("RailOn.TLabel", background=MIST, foreground=NAVY, font=self.f_h2)
        s.configure("RailOff.TLabel", background=MIST, foreground=MUTED, font=self.f_body)
        s.configure("TButton", font=self.f_body, padding=(14, 7))
        s.configure("Go.TButton", font=self.f_h2, padding=(22, 11))

    def _vars(self) -> None:
        self.v_wifi = tk.StringVar(value="checking WiFi...")
        self.v_status = tk.StringVar(value="")
        self.v_uri = tk.StringVar(value="")
        self.v_stat = tk.StringVar(value="")
        self.v_ip = tk.StringVar(value=", ".join(core.DEFAULT_IPS))
        self.v_note = tk.StringVar(value="Press Scan to look for the drone.")

    def _build(self) -> None:
        head = ttk.Frame(self.root, padding=(18, 14, 18, 10))
        head.pack(fill="x")
        ttk.Label(head, text="Flight Watch", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text="Checks the drone, then records the camera while you fly. Never commands the drone.",
            style="Muted.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(self.root, padding=(18, 0, 18, 0))
        body.pack(fill="both", expand=True)

        rail = ttk.Frame(body, style="Rail.TFrame", padding=(16, 16))
        rail.pack(side="left", fill="y")
        self.rail_labels = []
        for i, name in enumerate(STEPS):
            lab = ttk.Label(rail, text=f"{i + 1}.  {name}", style="RailOff.TLabel")
            lab.pack(anchor="w", pady=(0, 12))
            self.rail_labels.append(lab)
        ttk.Label(rail, text="", style="Rail.TLabel").pack(expand=True, fill="y")
        ttk.Label(rail, textvariable=self.v_wifi, style="Rail.TLabel",
                  wraplength=190, justify="left").pack(anchor="w")

        self.stage = ttk.Frame(body, style="Card.TFrame", padding=22)
        self.stage.pack(side="left", fill="both", expand=True, padx=(14, 0))

        foot = ttk.Frame(self.root, padding=(18, 12, 18, 16))
        foot.pack(fill="x")
        ttk.Label(foot, textvariable=self.v_status, style="Muted.TLabel").pack(side="left")
        self.b_next = ttk.Button(foot, text="Next", style="Go.TButton", command=self._next)
        self.b_next.pack(side="right")
        self.b_back = ttk.Button(foot, text="Back", command=self._back)
        self.b_back.pack(side="right", padx=(0, 8))
        self.b_stop = ttk.Button(foot, text="Stop", command=self._stop, state="disabled")
        self.b_stop.pack(side="right", padx=(0, 8))

    def _clear(self) -> None:
        for child in self.stage.winfo_children():
            child.destroy()

    def _goto(self, index: int) -> None:
        self.step = index
        for i, lab in enumerate(self.rail_labels):
            lab.configure(style="RailOn.TLabel" if i == index else "RailOff.TLabel")
        self._clear()
        (self._step_radio, self._step_pre, self._step_watch, self._step_result)[index]()
        self.b_back.configure(state="normal" if index > 0 else "disabled")

    # -- step 1: radio -----------------------------------------------------

    def _step_radio(self) -> None:
        ttk.Label(self.stage, text="Step 1  ·  Find the drone", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            self.stage,
            text=("Plug in the Crazyradio and switch the drone on. Nothing spins.\n"
                  "This step only listens for which drones are answering."),
            style="Card.TLabel", justify="left",
        ).pack(anchor="w", pady=(6, 14))

        ok, err = cflib_available()
        if not ok:
            ttk.Label(self.stage, text="cflib is not installed", style="Bad.TLabel").pack(anchor="w")
            ttk.Label(
                self.stage,
                text=(f"{err}\n\nInstall it with:    py -3 -m pip install cflib\n\n"
                      "You can still press Next -- the camera recording works without it, "
                      "you just get no battery or deck check."),
                style="CardMuted.TLabel", justify="left",
            ).pack(anchor="w", pady=(4, 0))
            return

        self.list_box = tk.Listbox(self.stage, height=5, font=self.f_mono,
                                   highlightthickness=1, relief="flat")
        self.list_box.pack(anchor="w", fill="x", pady=(0, 8))
        for u in self.uris:
            self.list_box.insert("end", u)
        if self.uris:
            self.list_box.selection_set(0)

        ttk.Button(self.stage, text="Scan for drones", command=self._begin_scan).pack(anchor="w")
        ttk.Label(self.stage, textvariable=self.v_note, style="CardMuted.TLabel",
                  wraplength=580, justify="left").pack(anchor="w", pady=(10, 0))

    def _begin_scan(self) -> None:
        self.v_note.set("Scanning all channels, a few seconds...")
        self._set_busy(True, "scanning")

        def work() -> None:
            try:
                self.events.put({"type": "scan", "uris": scan_drones()})
            except Exception as exc:  # noqa: BLE001
                self.events.put({"type": "scan_fail", "error": f"{type(exc).__name__}: {exc}"})

        self._spawn(work)

    # -- step 2: pre-flight ------------------------------------------------

    def _step_pre(self) -> None:
        ttk.Label(self.stage, text="Step 2  ·  Pre-flight check", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            self.stage,
            text=("Reads battery, fitted decks and the position estimator.\n"
                  "No motors. The radio is handed back before the next step."),
            style="Card.TLabel", justify="left",
        ).pack(anchor="w", pady=(6, 12))

        self.pre_box = tk.Text(self.stage, height=15, font=self.f_mono, wrap="word",
                               relief="flat", background="#FAFAF7")
        self.pre_box.pack(fill="both", expand=True)
        self.pre_box.configure(state="disabled")
        if self.pre:
            self._render_pre(self.pre)

        row = ttk.Frame(self.stage, style="Card.TFrame")
        row.pack(anchor="w", pady=(10, 0))
        ok, _ = cflib_available()
        state = "normal" if (ok and self.v_uri.get()) else "disabled"
        ttk.Button(row, text="Run pre-flight check", command=self._begin_pre,
                   state=state).pack(side="left")
        if not self.v_uri.get():
            ttk.Label(row, text="   no drone selected -- go back and scan",
                      style="CardMuted.TLabel").pack(side="left")

    def _begin_pre(self) -> None:
        self._set_busy(True, "reading drone")
        uri = self.v_uri.get()
        cache = str(core.LAB_DIR / ".cfcache")

        def work() -> None:
            try:
                self.events.put({"type": "pre", "data": preflight(uri, cache)})
            except Exception as exc:  # noqa: BLE001
                self.events.put({"type": "pre_fail", "error": f"{type(exc).__name__}: {exc}"})

        self._spawn(work)

    def _render_pre(self, d: dict) -> None:
        decks = {k: v for k, v in d.get("decks", {}).items() if v}
        tel = d.get("telemetry", {})
        vbat = tel.get("pm.vbat")
        lines = [f"Drone      {d.get('uri')}", "", "Decks fitted"]
        for k in sorted(decks):
            lines.append(f"    {k}")
        if not decks:
            lines.append("    (none reported)")
        lines += ["", "Battery"]
        if vbat is None:
            lines.append("    not read")
        elif vbat >= VBAT_GOOD:
            lines.append(f"    {vbat:.2f} V   good")
        elif vbat >= VBAT_LOW:
            lines.append(f"    {vbat:.2f} V   getting low, expect a short flight")
        else:
            lines.append(f"    {vbat:.2f} V   TOO LOW -- charge before flying")
        lines += ["", "On the ground"]
        for k, label in (("stateEstimate.z", "height estimate"),
                         ("range.zrange", "laser (mm)"),
                         ("kalman.varPX", "position spread X"),
                         ("kalman.varPY", "position spread Y")):
            if k in tel:
                lines.append(f"    {label:<20} {tel[k]}")
        lines.append("")
        if not decks.get("bcFlow2"):
            lines.append("WARNING: no flow deck detected. The drone cannot hold position.")
        if not decks.get("bcAI"):
            lines.append("WARNING: no AI deck detected. There will be no camera to record.")
        lines += [
            "Position spread is normally large on the ground -- the flow deck cannot",
            "focus on the floor from a few millimetres up. Judge it in the air.",
        ]
        self.pre_box.configure(state="normal")
        self.pre_box.delete("1.0", "end")
        self.pre_box.insert("end", "\n".join(lines))
        self.pre_box.configure(state="disabled")

    # -- step 3: watch -----------------------------------------------------

    def _step_watch(self) -> None:
        banner = ttk.Frame(self.stage, style="Banner.TFrame", padding=(12, 9))
        banner.pack(fill="x")
        ttk.Label(banner, text="Radio released  --  fly.py can connect now",
                  style="Banner.TLabel").pack(anchor="w")

        ttk.Label(self.stage, text="Step 3  ·  Record the flight",
                  style="H2.TLabel").pack(anchor="w", pady=(14, 0))
        ttk.Label(
            self.stage,
            text=("Join the drone's WiFi, press Start, then fly in your other window.\n"
                  "Every frame is saved with its timestamp. Press Stop when it lands."),
            style="Card.TLabel", justify="left",
        ).pack(anchor="w", pady=(6, 12))

        split = ttk.Frame(self.stage, style="Card.TFrame")
        split.pack(fill="both", expand=True)

        left = ttk.Frame(split, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(left, background="#111418", width=52, height=17)
        self.preview.pack(anchor="nw")
        ttk.Label(left, textvariable=self.v_stat, style="Card.TLabel",
                  font=self.f_stat).pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(split, style="Card.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.log_box = tk.Text(right, height=16, font=self.f_mono, wrap="word",
                               relief="flat", background="#FAFAF7")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        row = ttk.Frame(self.stage, style="Card.TFrame")
        row.pack(anchor="w", pady=(10, 0))
        self.b_rec = ttk.Button(row, text="Start recording", command=self._begin_watch)
        self.b_rec.pack(side="left")
        if not PIL_OK:
            ttk.Label(row, text="   (no live picture -- py -3 -m pip install pillow)",
                      style="CardMuted.TLabel").pack(side="left")

    def _begin_watch(self) -> None:
        self.cancel = core.CancelToken()
        self._set_busy(True, "recording")
        self.b_rec.configure(state="disabled")
        session = self._ensure_session()
        run_dir = core.new_run_dir(session, "flight")
        ips = [s.strip() for s in self.v_ip.get().split(",") if s.strip()]
        cancel = self.cancel

        def work() -> None:
            summary = self._record(ips, run_dir, cancel)
            self.events.put({"type": "watch_done", "summary": summary})

        self._spawn(work)

    def _record(self, ips: list[str], run_dir: Path, cancel: core.CancelToken) -> dict:
        """Read frames until stopped. Camera only -- never touches the radio."""
        summary: dict = {
            "test": "flight_watch",
            "created_at": core.stamp_now(),
            "run_dir": str(run_dir),
            "connected": False,
            "frames": 0,
            "packets": 0,
            "bytes_total": 0,
            "stall": None,
            "error": None,
        }
        sock = None
        for entry in ips:
            # "ip" or "ip:port" -- the port form is only needed to point this at
            # mock_deck.py for a dry run without hardware.
            ip, _, port_text = entry.partition(":")
            port = int(port_text) if port_text.isdigit() else core.DEFAULT_PORT
            try:
                self.events.put({"type": "line", "text": f"connecting {ip}:{port} ..."})
                sock = core.connect_to_deck(ip, port, 6.0)
                summary["target"] = {"ip": ip, "port": port}
                break
            except Exception as exc:  # noqa: BLE001
                self.events.put({"type": "line", "text": f"  {ip}:{port} failed: {exc}"})
        if sock is None:
            summary["error"] = "could not reach the deck on any address"
            self.events.put({"type": "line",
                             "text": "No camera. Are you on the aideck-stream WiFi?"})
            core.write_summary(run_dir, summary)
            return summary

        summary["connected"] = True
        self.events.put({"type": "line", "text": "camera connected -- recording"})
        sock.settimeout(4.0)

        fh = (run_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=[
            "frame_index", "t_rel_s", "interval_s", "size_bytes", "width", "height", "file"])
        writer.writeheader()

        current = None
        buf = bytearray()
        started = core.monotonic()
        last = started
        last_ui = started
        try:
            while not cancel.cancelled:  # `cancelled` is a property, not a call
                try:
                    _l, _r, _f, payload = core.read_packet(sock)
                except socket.timeout:
                    summary["stall"] = {
                        "after_frames": summary["frames"],
                        "t_rel_s": round(core.monotonic() - started, 3),
                    }
                    self.events.put({"type": "line", "text":
                                     f"STREAM STALLED after {summary['frames']} frames"})
                    break
                except OSError as exc:
                    summary["error"] = str(exc)
                    break

                now = core.monotonic()
                summary["packets"] += 1
                summary["bytes_total"] += 4 + len(payload)

                header = core.parse_image_header(payload)
                if header is not None:
                    current = header
                    buf = bytearray()
                    continue
                if current is None:
                    continue
                buf.extend(payload)
                if len(buf) < current["size"]:
                    continue

                summary["frames"] += 1
                idx = summary["frames"]
                name = f"frame_{idx:05d}{core.frame_suffix(current.get('type', 1))}"
                (run_dir / name).write_bytes(bytes(buf))
                writer.writerow({
                    "frame_index": idx,
                    "t_rel_s": round(now - started, 3),
                    "interval_s": round(now - last, 3),
                    "size_bytes": len(buf),
                    "width": current.get("width"),
                    "height": current.get("height"),
                    "file": name,
                })
                fh.flush()
                last = now
                if now - last_ui > 0.25:
                    last_ui = now
                    el = now - started
                    self.events.put({
                        "type": "progress",
                        "elapsed": el,
                        "frames": idx,
                        "fps": idx / el if el > 0 else 0.0,
                        "frame": bytes(buf),
                    })
                current = None
                buf = bytearray()
        finally:
            fh.close()
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass

        el = core.monotonic() - started
        summary["elapsed_s"] = round(el, 1)
        summary["fps"] = round(summary["frames"] / el, 3) if el > 0 else 0.0
        core.write_summary(run_dir, summary)
        return summary

    # -- step 4: result ----------------------------------------------------

    def _step_result(self) -> None:
        ttk.Label(self.stage, text="Step 4  ·  What was recorded",
                  style="H2.TLabel").pack(anchor="w")
        s = self.watch_summary or {}
        frames = s.get("frames", 0)
        if not s:
            ttk.Label(self.stage, text="Nothing recorded yet.",
                      style="Card.TLabel").pack(anchor="w", pady=(10, 0))
        elif s.get("error"):
            ttk.Label(self.stage, text="No recording", style="Bad.TLabel").pack(
                anchor="w", pady=(10, 0))
            ttk.Label(self.stage, text=str(s["error"]), style="Card.TLabel").pack(anchor="w")
        elif s.get("stall"):
            ttk.Label(self.stage, text="Recorded, but the stream stalled",
                      style="Warn.TLabel").pack(anchor="w", pady=(10, 0))
            ttk.Label(self.stage, text=(
                f"{frames} frames before it stopped, {s['stall'].get('t_rel_s')}s in. "
                "The flight data up to that point is still good."
            ), style="Card.TLabel", wraplength=620, justify="left").pack(anchor="w")
        else:
            ttk.Label(self.stage, text="Recorded cleanly", style="Good.TLabel").pack(
                anchor="w", pady=(10, 0))
            ttk.Label(self.stage, text=(
                f"{frames} frames over {s.get('elapsed_s')}s at {s.get('fps')} fps."
            ), style="Card.TLabel").pack(anchor="w")

        if self.session_dir:
            self._write_report()
            ttk.Label(self.stage, text=str(self.session_dir), style="CardMuted.TLabel",
                      wraplength=620, justify="left").pack(anchor="w", pady=(12, 6))
            row = ttk.Frame(self.stage, style="Card.TFrame")
            row.pack(anchor="w")
            ttk.Button(row, text="Open folder", command=self._open_session).pack(side="left")
            ttk.Button(row, text="Record another", command=self._again).pack(side="left", padx=(8, 0))

    def _again(self) -> None:
        self.watch_summary = None
        self._goto(2)

    def _ensure_session(self) -> Path:
        if self.session_dir is None:
            self.session_dir = core.new_run_dir(core.DEFAULT_OUT_DIR, "flightwatch")
        return self.session_dir

    def _write_report(self) -> None:
        if not self.session_dir:
            return
        s = self.watch_summary or {}
        pre = self.pre or {}
        tel = pre.get("telemetry", {})
        decks = [k for k, v in pre.get("decks", {}).items() if v]
        lines = [
            "# Flight watch session",
            "",
            f"Run: {core.stamp_now()}",
            f"Drone: {pre.get('uri', 'not checked')}",
            "",
            "## Pre-flight",
            "",
            f"- Decks: {', '.join(decks) if decks else 'not read'}",
            f"- Battery: {tel.get('pm.vbat', 'not read')} V",
            f"- Ground height estimate: {tel.get('stateEstimate.z', 'n/a')}",
            f"- Laser (mm): {tel.get('range.zrange', 'n/a')}",
            "",
            "## Recording",
            "",
            f"- Frames: {s.get('frames', 0)}",
            f"- Duration: {s.get('elapsed_s', 'n/a')} s",
            f"- Rate: {s.get('fps', 'n/a')} fps",
            f"- Stalled: {'yes' if s.get('stall') else 'no'}",
            f"- Error: {s.get('error') or 'none'}",
            "",
            "Frames and `frames.csv` are in the sub-folder. Times in `frames.csv` are",
            "seconds from the start of recording, so they line up with the flight log",
            "written by `fly.py` by wall-clock offset.",
            "",
            "This tool never commanded the drone.",
        ]
        (self.session_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        try:
            core.zip_run_dir(self.session_dir)
        except Exception:  # noqa: BLE001
            pass

    def _open_session(self) -> None:
        import subprocess

        if self.session_dir:
            subprocess.Popen(["explorer", str(self.session_dir)])

    # -- plumbing ----------------------------------------------------------

    def _spawn(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self.v_status.set(status)
        self.b_next.configure(state="disabled" if busy else "normal")
        self.b_stop.configure(state="normal" if busy else "disabled")

    def _stop(self) -> None:
        self.cancel.cancel()
        self.v_status.set("stopping...")

    def _line(self, text: str) -> None:
        box = getattr(self, "log_box", None)
        if box is None:
            return
        box.configure(state="normal")
        box.insert("end", text + "\n")
        box.see("end")
        box.configure(state="disabled")

    def _show(self, data: bytes) -> None:
        if not PIL_OK:
            return
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((420, 320))
            self.photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.photo, width=img.width, height=img.height)
        except Exception:  # noqa: BLE001
            pass

    def _refresh_wifi(self) -> None:
        def probe() -> None:
            try:
                info = core.wlan_interfaces()
                ssid = ""
                for line in info.get("stdout", "").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("SSID") and ":" in stripped and "BSSID" not in stripped:
                        ssid = stripped.split(":", 1)[1].strip()
                        break
                self.events.put({"type": "wifi", "ssid": ssid})
            except Exception:  # noqa: BLE001
                pass

        self._spawn(probe)
        self.root.after(6000, self._refresh_wifi)

    def _pump(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                t = ev["type"]
                if t == "scan":
                    self.uris = ev["uris"]
                    self._set_busy(False)
                    if self.uris:
                        self.v_uri.set(self.uris[0])
                        note = f"Found {len(self.uris)}: " + ", ".join(self.uris)
                        if len(self.uris) > 1:
                            note += ("\n\nMore than one drone is answering. Make sure the one you "
                                     "pick is the one in front of you before you fly.")
                    else:
                        note = "No drone answered. Is it switched on and charged?"
                    self.v_note.set(note)
                    if hasattr(self, "list_box"):
                        self.list_box.delete(0, "end")
                        for u in self.uris:
                            self.list_box.insert("end", u)
                        if self.uris:
                            self.list_box.selection_set(0)
                elif t == "scan_fail":
                    self._set_busy(False)
                    self.v_note.set("Scan failed: " + ev["error"])
                elif t == "pre":
                    self.pre = ev["data"]
                    self._set_busy(False)
                    self._render_pre(ev["data"])
                elif t == "pre_fail":
                    self._set_busy(False)
                    self.pre_box.configure(state="normal")
                    self.pre_box.delete("1.0", "end")
                    self.pre_box.insert("end", "Could not read the drone.\n\n" + ev["error"])
                    self.pre_box.configure(state="disabled")
                elif t == "line":
                    self._line(ev["text"])
                elif t == "progress":
                    self.v_stat.set(f"{ev['frames']:5d} frames   {ev['fps']:.1f} fps   "
                                    f"{ev['elapsed']:.0f}s")
                    self._show(ev["frame"])
                elif t == "watch_done":
                    self.watch_summary = ev["summary"]
                    self._set_busy(False)
                    if hasattr(self, "b_rec"):
                        self.b_rec.configure(state="normal")
                    self._goto(3)
                elif t == "wifi":
                    ssid = ev.get("ssid") or "not connected"
                    if ssid == core.DEFAULT_SSID:
                        self.v_wifi.set(f"WiFi: {ssid}\ncamera reachable")
                    else:
                        self.v_wifi.set(f"WiFi: {ssid}\njoin {core.DEFAULT_SSID} to record")
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _next(self) -> None:
        if self.step == 0 and hasattr(self, "list_box"):
            sel = self.list_box.curselection()
            if sel:
                self.v_uri.set(self.list_box.get(sel[0]))
        if self.step < 3:
            self._goto(self.step + 1)

    def _back(self) -> None:
        if self.step > 0:
            self._goto(self.step - 1)


def main() -> None:
    root = tk.Tk()
    WatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
