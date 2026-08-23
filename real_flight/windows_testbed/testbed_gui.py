"""AI Deck WiFi Test -- guided Tkinter front end.

Four steps, one button at a time:

    1. Connect   join the deck's WiFi (or start the practice deck)
    2. Find      locate the deck on the network
    3. Test      run all three diagnostics back to back
    4. Result    plain-language verdict and where the evidence went

Everything is logged automatically. There is nothing to configure to get a
valid run; the Advanced panel exists only to override the defaults.

The diagnostics themselves live in aideck_tests.py. This file is the operator
interface and must not contain protocol logic.
"""

from __future__ import annotations

import io
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aideck_core as core
import aideck_tests as tests

try:
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
MUTED = "#6B7280"

MOCK_PORT = 5555
STEPS = ("Connect", "Find deck", "Run tests", "Result")

# The sequence step 3 runs. Kept short on purpose: a full pass should be a few
# minutes, not a decision.
SEQUENCE = (
    ("packet", "Packet test", "Finds the exact packet where the stream breaks."),
    ("reconnect", "Reconnect test", "Checks if a fresh connection gets one clean frame."),
    ("throughput", "Sustained throughput", "Checks if it holds up over minutes."),
)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("AI Deck WiFi Test")
        root.geometry("1060x730")
        root.minsize(940, 660)
        root.configure(bg=PAPER)

        self.events: queue.Queue[dict] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel: core.CancelToken | None = None
        self.mock_proc: subprocess.Popen | None = None

        self.step = 0
        self.target_ip: str | None = None
        self.target_port = core.DEFAULT_PORT
        self.practice = False
        self.last_ssid = ""
        self.session_dir: Path | None = None
        self.results: dict[str, dict] = {}
        self._preview_ref = None

        self._fonts()
        self._style()
        self._vars()
        self._build()

        self._goto(0)
        self.root.after(80, self._pump)
        self.root.after(300, self._refresh_wifi)
        root.protocol("WM_DELETE_WINDOW", self._close)

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
        s.configure("TLabel", background=PAPER, foreground=INK, font=self.f_body)
        s.configure("Card.TLabel", background="white", foreground=INK, font=self.f_body)
        s.configure("H1.TLabel", background=PAPER, foreground=NAVY, font=self.f_h1)
        s.configure("H2.TLabel", background="white", foreground=NAVY, font=self.f_h2)
        s.configure("Muted.TLabel", background=PAPER, foreground=MUTED, font=self.f_small)
        s.configure("CardMuted.TLabel", background="white", foreground=MUTED, font=self.f_small)
        s.configure("Good.TLabel", background="white", foreground=GOOD, font=self.f_h2)
        s.configure("Bad.TLabel", background="white", foreground=BAD, font=self.f_h2)
        s.configure("Rail.TLabel", background=MIST, foreground=INK, font=self.f_body)
        s.configure("RailOn.TLabel", background=MIST, foreground=NAVY, font=self.f_h2)
        s.configure("RailOff.TLabel", background=MIST, foreground=MUTED, font=self.f_body)
        s.configure("TButton", font=self.f_body, padding=(14, 7))
        s.configure("Go.TButton", font=self.f_h2, padding=(22, 11))
        s.configure("TCheckbutton", background="white", font=self.f_body)
        s.configure("TRadiobutton", background="white", font=self.f_body)

    def _vars(self) -> None:
        self.v_wifi = tk.StringVar(value="checking WiFi...")
        self.v_status = tk.StringVar(value="")
        self.v_mode = tk.StringVar(value="real")
        self.v_stat = tk.StringVar(value="")
        self.v_ip = tk.StringVar(value=", ".join(core.DEFAULT_IPS))
        self.v_port = tk.StringVar(value=str(core.DEFAULT_PORT))
        self.v_tp_secs = tk.StringVar(value="120")
        self.v_attempts = tk.StringVar(value="6")

    def _build(self) -> None:
        head = ttk.Frame(self.root, padding=(18, 14, 18, 10))
        head.pack(fill="x")
        ttk.Label(head, text="AI Deck WiFi Test", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text="Measures where the AI Deck camera stream breaks, and saves the evidence.",
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
        self.rail_wifi = ttk.Label(
            rail, textvariable=self.v_wifi, style="Rail.TLabel", wraplength=190, justify="left"
        )
        self.rail_wifi.pack(anchor="w")

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

    def _clear_stage(self) -> None:
        for child in self.stage.winfo_children():
            child.destroy()

    def _goto(self, index: int) -> None:
        self.step = index
        for i, lab in enumerate(self.rail_labels):
            lab.configure(style="RailOn.TLabel" if i == index else "RailOff.TLabel")
        self._clear_stage()
        (self._step_connect, self._step_find, self._step_run, self._step_result)[index]()
        self.b_back.configure(state="normal" if index > 0 else "disabled")

    # -- step 1: connect ---------------------------------------------------

    def _step_connect(self) -> None:
        ttk.Label(self.stage, text="Step 1  ·  Get connected", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            self.stage,
            text=(
                "The AI Deck broadcasts its own WiFi network. Your PC has to join it before\n"
                "the deck can be tested. You will lose internet while you are on it."
            ),
            style="Card.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(8, 16))

        box = ttk.Frame(self.stage, style="Card.TFrame")
        box.pack(fill="x")

        ttk.Radiobutton(
            box,
            text="  Test the real AI Deck",
            value="real",
            variable=self.v_mode,
            command=self._mode_changed,
        ).pack(anchor="w")
        ttk.Label(
            box,
            text=(
                f"      Open the Windows WiFi menu and join  {core.DEFAULT_SSID}\n"
                "      Propellers off. Crazyflie powered on."
            ),
            style="CardMuted.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(2, 14))

        ttk.Radiobutton(
            box,
            text="  Practice run (no hardware needed)",
            value="mock",
            variable=self.v_mode,
            command=self._mode_changed,
        ).pack(anchor="w")
        ttk.Label(
            box,
            text=(
                "      Starts a simulated deck on this PC so you can learn the tool.\n"
                "      Stays on your normal WiFi. Results are not real measurements."
            ),
            style="CardMuted.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(2, 14))

        self.wifi_banner = ttk.Label(self.stage, text="", style="Card.TLabel", justify="left")
        self.wifi_banner.pack(anchor="w", pady=(6, 0))
        self._paint_wifi_banner()
        self.b_next.configure(text="Next")

    def _mode_changed(self) -> None:
        self.practice = self.v_mode.get() == "mock"
        self._paint_wifi_banner()

    def _paint_wifi_banner(self) -> None:
        if not hasattr(self, "wifi_banner") or not self.wifi_banner.winfo_exists():
            return
        if self.v_mode.get() == "mock":
            self.wifi_banner.configure(
                text="Practice deck will start automatically when you press Next.",
                style="Card.TLabel",
            )
            return
        # Uses the cached value from the background probe; netsh takes about a
        # second and must never run on the GUI thread.
        ssid = self.last_ssid
        if ssid == core.DEFAULT_SSID:
            self.wifi_banner.configure(
                text=f"Connected to {ssid}  —  ready.", style="Good.TLabel"
            )
        elif ssid:
            self.wifi_banner.configure(
                text=f"Currently on {ssid}. Join {core.DEFAULT_SSID} to continue.",
                style="Bad.TLabel",
            )
        else:
            self.wifi_banner.configure(text="No wireless network detected.", style="Bad.TLabel")

    def _ssid(self) -> str:
        try:
            return core.wlan_interfaces()["parsed"].get("ssid", "")
        except Exception:  # noqa: BLE001
            return ""

    def _refresh_wifi(self) -> None:
        def probe() -> None:
            ssid = self._ssid()
            self.events.put({"kind": "wifi", "ssid": ssid})

        threading.Thread(target=probe, daemon=True).start()
        self.root.after(4000, self._refresh_wifi)

    # -- step 2: find ------------------------------------------------------

    def _step_find(self) -> None:
        ttk.Label(self.stage, text="Step 2  ·  Find the deck", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            self.stage,
            text="Checks the network and locates the deck. Nothing is changed on the drone.",
            style="Card.TLabel",
        ).pack(anchor="w", pady=(8, 14))

        self.find_state = ttk.Label(self.stage, text="Press Find deck.", style="Card.TLabel")
        self.find_state.pack(anchor="w")

        self.find_log = tk.Text(
            self.stage,
            height=14,
            wrap="word",
            font=self.f_mono,
            relief="flat",
            background="#FAFAF7",
            foreground=INK,
            state="disabled",
        )
        self.find_log.pack(fill="both", expand=True, pady=(14, 0))

        self._advanced(self.stage)
        self.b_next.configure(text="Find deck")

    def _advanced(self, parent) -> None:
        wrap = ttk.Frame(parent, style="Card.TFrame")
        wrap.pack(fill="x", pady=(12, 0))
        shown = tk.BooleanVar(value=False)
        inner = ttk.Frame(wrap, style="Card.TFrame")

        def toggle() -> None:
            if shown.get():
                inner.pack_forget()
                shown.set(False)
                btn.configure(text="Advanced settings  ▸")
            else:
                inner.pack(fill="x", pady=(8, 0))
                shown.set(True)
                btn.configure(text="Advanced settings  ▾")

        btn = ttk.Button(wrap, text="Advanced settings  ▸", command=toggle)
        btn.pack(anchor="w")

        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(anchor="w")
        rows = (
            ("Deck IP(s)", self.v_ip, 30),
            ("Port", self.v_port, 8),
            ("Throughput seconds", self.v_tp_secs, 8),
            ("Reconnect attempts", self.v_attempts, 8),
        )
        for r, (label, var, width) in enumerate(rows):
            ttk.Label(grid, text=label, style="Card.TLabel").grid(
                row=r, column=0, sticky="e", padx=(0, 8), pady=3
            )
            ttk.Entry(grid, textvariable=var, width=width, font=self.f_body).grid(
                row=r, column=1, sticky="w", pady=3
            )

    # -- step 3: run -------------------------------------------------------

    def _step_run(self) -> None:
        ttk.Label(self.stage, text="Step 3  ·  Run the tests", style="H2.TLabel").pack(anchor="w")
        target = f"{self.target_ip}:{self.target_port}" if self.target_ip else "not set"
        ttk.Label(
            self.stage,
            text=f"Three tests, back to back, against {target}. Everything is saved as it runs.",
            style="Card.TLabel",
        ).pack(anchor="w", pady=(8, 14))

        cols = ttk.Frame(self.stage, style="Card.TFrame")
        cols.pack(fill="both", expand=True)

        left = ttk.Frame(cols, style="Card.TFrame")
        left.pack(side="left", fill="both", expand=True)

        self.seq_labels: dict[str, ttk.Label] = {}
        for key, name, why in SEQUENCE:
            row = ttk.Frame(left, style="Card.TFrame")
            row.pack(fill="x", pady=(0, 10))
            lab = ttk.Label(row, text=f"○  {name}", style="Card.TLabel")
            lab.pack(anchor="w")
            ttk.Label(row, text=f"     {why}", style="CardMuted.TLabel").pack(anchor="w")
            self.seq_labels[key] = lab

        ttk.Label(left, textvariable=self.v_stat, style="Card.TLabel", font=self.f_stat).pack(
            anchor="w", pady=(6, 8)
        )

        self.run_log = tk.Text(
            left,
            height=11,
            wrap="none",
            font=self.f_mono,
            relief="flat",
            background="#FAFAF7",
            foreground=INK,
            state="disabled",
        )
        self.run_log.pack(fill="both", expand=True)

        right = ttk.Frame(cols, style="Card.TFrame")
        right.pack(side="left", fill="y", padx=(18, 0))
        ttk.Label(right, text="Live camera frame", style="CardMuted.TLabel").pack(anchor="w")
        self.preview = ttk.Label(
            right,
            text="waiting..." if PIL_OK else "(install Pillow for preview)",
            style="CardMuted.TLabel",
            width=34,
            anchor="center",
        )
        self.preview.pack(pady=(6, 0))

        self.b_next.configure(text="Run all tests")

    # -- step 4: result ----------------------------------------------------

    def _step_result(self) -> None:
        ttk.Label(self.stage, text="Step 4  ·  Result", style="H2.TLabel").pack(anchor="w")

        verdict, detail, healthy = self._verdict()
        ttk.Label(
            self.stage, text=verdict, style="Good.TLabel" if healthy else "Bad.TLabel",
            wraplength=680, justify="left",
        ).pack(anchor="w", pady=(10, 6))
        ttk.Label(
            self.stage, text=detail, style="Card.TLabel", wraplength=680, justify="left"
        ).pack(anchor="w")

        if self.practice:
            ttk.Label(
                self.stage,
                text="Practice run against the simulated deck. These are not real measurements.",
                style="CardMuted.TLabel",
            ).pack(anchor="w", pady=(12, 0))

        where = ttk.Frame(self.stage, style="Card.TFrame")
        where.pack(fill="x", pady=(18, 0))
        ttk.Label(where, text="Saved to", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Label(
            where,
            text=str(self.session_dir) if self.session_dir else "(nothing saved)",
            style="Card.TLabel",
            font=self.f_mono,
            wraplength=680,
            justify="left",
        ).pack(anchor="w")

        row = ttk.Frame(self.stage, style="Card.TFrame")
        row.pack(anchor="w", pady=(14, 0))
        ttk.Button(row, text="Open folder", command=self._open_session).pack(side="left")
        ttk.Button(row, text="Start over", command=self._restart).pack(side="left", padx=(8, 0))

        self.b_next.configure(text="Done")

    def _verdict(self) -> tuple[str, str, bool]:
        tp = self.results.get("throughput") or {}
        pk = self.results.get("packet") or {}
        rc = self.results.get("reconnect") or {}

        pk_first = (pk.get("results") or [{}])[0] if pk.get("results") else {}

        pk_frames = pk_first.get("completed_frames", 0)
        pk_stalled = pk_first.get("stall") is not None
        pk_connected = bool(pk_first.get("connected"))

        rc_ok = rc.get("ok_count", 0)
        rc_tries = len(rc.get("results") or [])

        tp_frames = tp.get("frames", 0)
        tp_stalled = tp.get("stall") is not None
        tp_secs = tp.get("elapsed_s", 0) or 0
        tp_connected = bool(tp.get("connected"))

        # A run that never opened a socket is a different fact from a run that
        # opened one and got nothing; do not report the first as "0 frames".
        if tp_connected:
            tp_line = f"Throughput: {tp_frames} frames in {tp_secs:.0f}s ({tp.get('fps', 0)} fps)."
        elif tp:
            tp_line = "Throughput: could not connect."
        else:
            tp_line = "Throughput: not run."

        stats = (
            f"Packet test: {pk_frames} frame(s)"
            f"{', then stalled' if pk_stalled else ''}. "
            f"Reconnect: {rc_ok}/{rc_tries} clean. "
            + tp_line
        )

        no_socket = not pk_connected and not tp_connected and rc_ok == 0
        if no_socket:
            return (
                "Could not reach the deck at all.",
                "Nothing accepted a connection. Check that the Crazyflie is powered and that "
                "this PC is still on the deck's WiFi network, then run again.",
                False,
            )

        healthy = tp_connected and not tp_stalled and tp_frames > 50
        if healthy:
            return (
                "Stream held up. No stall.",
                stats
                + "\n\nThis PC does not use the VirtualBox NAT path. A clean run here against "
                "the same firmware that stalls in the VM means the VM's network stack was the "
                "problem, not the deck. Do not flash the TXQ16 image on the old VM logs alone.",
                True,
            )

        # Any of these is the stream breaking. Reconnect failures count: on
        # 2026-05-16 the deck recovered only 1 of 8 fresh connections, so a deck
        # that stops accepting connections is showing the same fault.
        wedged = bool(tp) and not tp_connected and pk_connected
        broke = tp_stalled or pk_stalled or wedged or (rc_tries and rc_ok < rc_tries)

        if broke:
            if tp_stalled:
                where = f"Throughput stalled after {tp.get('stall', {}).get('after_frames', tp_frames)} frames."
            elif wedged:
                where = (
                    "The deck stopped accepting connections after the earlier tests, so the "
                    "throughput test could not start."
                )
            elif pk_stalled:
                where = f"The packet test got {pk_frames} frame(s), then the stream stopped."
            else:
                where = f"Only {rc_ok} of {rc_tries} fresh connections returned a clean frame."
            return (
                "The stream breaks.",
                f"{where}\n\n{stats}\n\nThis happened without the VirtualBox NAT layer in the "
                "path, so the VM is not the cause. That points at the deck. The queue-depth "
                "(TXQ16) image is the next thing to try.",
                False,
            )

        return (
            "Ran, but not enough to call it.",
            stats + "\n\nNo stall was seen, but too few frames to be confident. Increase "
            "Throughput seconds in Advanced settings and run again.",
            False,
        )

    # -- navigation --------------------------------------------------------

    def _next(self) -> None:
        if self.busy:
            return
        if self.step == 0:
            self._begin_find()
        elif self.step == 1:
            if self.target_ip:
                self._goto(2)
            else:
                self._begin_find()
        elif self.step == 2:
            self._begin_run()
        else:
            self._close()

    def _back(self) -> None:
        if self.busy:
            return
        self._goto(max(0, self.step - 1))

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _restart(self) -> None:
        self.results.clear()
        self.session_dir = None
        self.target_ip = None
        self._goto(0)

    # -- work: find --------------------------------------------------------

    def _begin_find(self) -> None:
        if self.practice and not self._start_mock():
            return
        if self.step != 1:
            self._goto(1)

        if self.practice:
            ips = ["127.0.0.1"]
            port = MOCK_PORT
        else:
            ips = [p.strip() for p in self.v_ip.get().split(",") if p.strip()]
            try:
                port = int(self.v_port.get())
            except ValueError:
                messagebox.showerror("Port", "Port must be a whole number.")
                return
        self.target_port = port

        self._set_busy(True, "Looking for the deck...")
        self.find_state.configure(text="Checking the network...", style="Card.TLabel")

        cfg = tests.TestConfig(ips=ips, port=port, connect_timeout=4.0)
        session = self._ensure_session()

        def work() -> None:
            try:
                run_dir = core.new_run_dir(session, "link_check")
                with core.Reporter(run_dir, "link_check", on_line=self._line) as rep:
                    summary = tests.link_check(cfg, run_dir, rep, self.cancel)
                self.events.put({"kind": "found", "summary": summary})
            except Exception:  # noqa: BLE001
                self.events.put({"kind": "line", "text": traceback.format_exc()})
                self.events.put({"kind": "found", "summary": {"reachable": []}})

        self.cancel = core.CancelToken()
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # -- work: run all -----------------------------------------------------

    def _begin_run(self) -> None:
        if not self.target_ip:
            messagebox.showinfo("No deck", "Find the deck first.")
            return
        try:
            tp_secs = float(self.v_tp_secs.get())
            attempts = int(self.v_attempts.get())
        except ValueError:
            messagebox.showerror("Settings", "Advanced settings must be numbers.")
            return

        cfg = tests.TestConfig(
            ips=[self.target_ip],
            port=self.target_port,
            duration=45.0,
            max_frames=3,
            attempts=attempts,
            delay=1.0,
            throughput_duration=tp_secs,
            save_every=25,
        )
        session = self._ensure_session()
        self._set_busy(True, "Running...")
        for key in self.seq_labels:
            self._mark(key, "○", MUTED)

        def work() -> None:
            try:
                for key, name, _why in SEQUENCE:
                    if self.cancel.cancelled:
                        break
                    self.events.put({"kind": "seq", "key": key, "state": "run"})
                    label, func, prefix = tests.TESTS[key]
                    run_dir = core.new_run_dir(session, prefix)
                    with core.Reporter(run_dir, prefix, on_line=self._line) as rep:
                        kwargs = {"on_frame": self._frame}
                        if key == "throughput":
                            kwargs["on_progress"] = self._progress
                        summary = func(cfg, run_dir, rep, self.cancel, **kwargs)
                    self.events.put(
                        {"kind": "seq", "key": key, "state": "done", "summary": summary}
                    )
            except Exception:  # noqa: BLE001
                self.events.put({"kind": "line", "text": traceback.format_exc()})
            finally:
                self.events.put({"kind": "all_done"})

        self.cancel = core.CancelToken()
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # -- worker callbacks (called off-thread) ------------------------------

    def _line(self, text: str) -> None:
        self.events.put({"kind": "line", "text": text})

    def _frame(self, data: bytes, header: dict) -> None:
        self.events.put({"kind": "frame", "data": data})

    def _progress(self, elapsed: float, frames: int, fps: float, rate: float) -> None:
        self.events.put(
            {"kind": "prog", "elapsed": elapsed, "frames": frames, "fps": fps, "rate": rate}
        )

    # -- event pump --------------------------------------------------------

    def _pump(self) -> None:
        newest_frame = None
        try:
            for _ in range(300):
                ev = self.events.get_nowait()
                kind = ev["kind"]
                if kind == "line":
                    self._append(ev["text"])
                elif kind == "frame":
                    newest_frame = ev["data"]
                elif kind == "wifi":
                    ssid = ev["ssid"]
                    self.last_ssid = ssid
                    self.v_wifi.set(f"WiFi\n{ssid or 'not connected'}")
                    self._paint_wifi_banner()
                elif kind == "prog":
                    self.v_stat.set(
                        f"{ev['frames']} frames   {ev['fps']:.1f} fps   {human(ev['rate'])}/s"
                    )
                elif kind == "seq":
                    self._on_seq(ev)
                elif kind == "found":
                    self._on_found(ev["summary"])
                elif kind == "all_done":
                    self._on_all_done()
        except queue.Empty:
            pass
        if newest_frame is not None:
            self._show(newest_frame)
        self.root.after(80, self._pump)

    def _on_seq(self, ev: dict) -> None:
        key = ev["key"]
        if ev["state"] == "run":
            self._mark(key, "▶", NAVY)
            self.v_status.set(f"Running {dict((k, n) for k, n, _ in SEQUENCE)[key]}...")
            return
        summary = ev.get("summary") or {}
        self.results[key] = summary
        bad = bool(summary.get("stall")) or bool(summary.get("error"))
        if key == "reconnect":
            bad = summary.get("ok_count", 0) < len(summary.get("results", []) or [1])
        self._mark(key, "✕" if bad else "✓", BAD if bad else GOOD)

    def _on_found(self, summary: dict) -> None:
        self._set_busy(False, "")
        reachable = summary.get("reachable") or []
        if reachable:
            self.target_ip = reachable[0]
            self.find_state.configure(
                text=f"Found the deck at {self.target_ip}:{self.target_port}.", style="Good.TLabel"
            )
            self.v_status.set("Deck found.")
            self.b_next.configure(text="Continue")
        else:
            self.find_state.configure(
                text="No deck answered. Check the WiFi network, then press Find deck again.",
                style="Bad.TLabel",
            )
            self.b_next.configure(text="Find deck")

    def _on_all_done(self) -> None:
        self._set_busy(False, "")
        self._write_report()
        self._stop_mock()
        try:
            if self.session_dir:
                core.zip_run_dir(self.session_dir)
        except OSError:
            pass
        self._goto(3)

    def _mark(self, key: str, glyph: str, colour: str) -> None:
        lab = self.seq_labels.get(key)
        if lab is None or not lab.winfo_exists():
            return
        name = dict((k, n) for k, n, _ in SEQUENCE)[key]
        lab.configure(text=f"{glyph}  {name}", foreground=colour)

    # -- output ------------------------------------------------------------

    def _target_log(self) -> tk.Text | None:
        for attr in ("run_log", "find_log"):
            widget = getattr(self, attr, None)
            if widget is not None and widget.winfo_exists():
                return widget
        return None

    def _append(self, text: str) -> None:
        widget = self._target_log()
        if widget is None:
            return
        widget.configure(state="normal")
        widget.insert("end", text + "\n")
        lines = int(widget.index("end-1c").split(".")[0])
        if lines > 800:
            widget.delete("1.0", f"{lines - 800}.0")
        widget.see("end")
        widget.configure(state="disabled")

    def _show(self, data: bytes) -> None:
        if not PIL_OK or not hasattr(self, "preview") or not self.preview.winfo_exists():
            return
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
            self._preview_ref = ImageTk.PhotoImage(img)
            self.preview.configure(image=self._preview_ref, text="")
        except Exception:  # noqa: BLE001
            pass

    def _set_busy(self, busy: bool, status: str) -> None:
        self.v_status.set(status)
        self.b_next.configure(state="disabled" if busy else "normal")
        self.b_back.configure(state="disabled" if busy else "normal")
        self.b_stop.configure(state="normal" if busy else "disabled")

    def _stop(self) -> None:
        if self.cancel:
            self.cancel.cancel()
        self.v_status.set("Stopping...")

    # -- session / report --------------------------------------------------

    def _ensure_session(self) -> Path:
        if self.session_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "practice" if self.practice else "session"
            self.session_dir = core.DEFAULT_OUT_DIR / f"{tag}_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir

    def _write_report(self) -> None:
        if self.session_dir is None:
            return
        verdict, detail, _ok = self._verdict()
        lines = [
            "# AI Deck WiFi test session",
            "",
            f"Run: {datetime.now().isoformat(timespec='seconds')}",
            f"Target: {self.target_ip}:{self.target_port}",
            f"Mode: {'practice (simulated deck)' if self.practice else 'real hardware'}",
            "",
            "## Verdict",
            "",
            verdict,
            "",
            detail,
            "",
            "## Numbers",
            "",
        ]
        for key, name, _why in SEQUENCE:
            s = self.results.get(key)
            if not s:
                lines.append(f"- {name}: not run")
                continue
            if key == "throughput":
                lines.append(
                    f"- {name}: {s.get('frames', 0)} frames in {s.get('elapsed_s', 0)}s, "
                    f"{s.get('fps', 0)} fps, {s.get('kib_per_s', 0)} KiB/s, "
                    f"stalled={s.get('stall') is not None}"
                )
            elif key == "reconnect":
                lines.append(
                    f"- {name}: {s.get('ok_count', 0)}/{len(s.get('results', []) or [])} clean"
                )
            else:
                first = (s.get("results") or [{}])[0]
                lines.append(
                    f"- {name}: {first.get('completed_frames', 0)} frames, "
                    f"{first.get('packets', 0)} packets, "
                    f"stalled={first.get('stall') is not None}"
                )
        if self.practice:
            lines += ["", "Practice run against a simulated deck. Not real measurements."]
        (self.session_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _open_session(self) -> None:
        target = self.session_dir or core.DEFAULT_OUT_DIR
        try:
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(target))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder", f"{target}\n\n{exc}")

    # -- practice deck -----------------------------------------------------

    def _start_mock(self) -> bool:
        if self.mock_proc and self.mock_proc.poll() is None:
            return True
        script = Path(__file__).resolve().parent / "mock_deck.py"
        if not script.exists():
            messagebox.showerror("Practice deck", "mock_deck.py not found next to this app.")
            return False
        try:
            self.mock_proc = subprocess.Popen(
                [sys.executable, str(script), "--port", str(MOCK_PORT), "--fps", "20"],
                cwd=str(script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=core._NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Practice deck", f"Could not start it.\n\n{exc}")
            return False
        self.root.after(600)
        return True

    def _stop_mock(self) -> None:
        if self.mock_proc and self.mock_proc.poll() is None:
            try:
                self.mock_proc.terminate()
                self.mock_proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        self.mock_proc = None

    def _close(self) -> None:
        if self.busy:
            if not messagebox.askokcancel("Quit", "A test is running. Stop it and quit?"):
                return
            if self.cancel:
                self.cancel.cancel()
            if self.worker:
                self.worker.join(timeout=3)
        self._stop_mock()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
