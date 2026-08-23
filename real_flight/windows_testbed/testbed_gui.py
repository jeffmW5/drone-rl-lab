"""AI Deck WiFi Test Bed -- Tkinter front end.

Runs the four diagnostics against the AI Deck AP, streams the log live, shows
the most recent JPEG frame when Pillow is available, and writes a timestamped
run folder plus a zip for every run.

Launch with run_testbed.bat, or:  py -3 testbed_gui.py
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import aideck_core as core
import aideck_tests as tests

try:
    from PIL import Image, ImageTk

    HAVE_PIL = True
except ImportError:  # preview degrades to "frame saved" text
    HAVE_PIL = False

NAVY = "#2C3E5C"
CHARCOAL = "#2B2B2B"
LIGHT = "#D5E2EC"
OFFWHITE = "#F5F5F0"
WHITE = "#FFFFFF"

PREVIEW_W, PREVIEW_H = 324, 244


class TestBedApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Deck WiFi Test Bed")
        self.root.geometry("1080x720")
        self.root.configure(bg=OFFWHITE)

        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel: core.CancelToken | None = None
        self.current_run_dir: Path | None = None
        self.last_run_dir: Path | None = None
        self._preview_image = None
        self._frames_seen = 0

        self._build_vars()
        self._build_style()
        self._build_layout()
        self.root.after(100, self._drain_events)
        self._log_line(f"Python {sys.version.split()[0]}  |  Pillow preview: "
                       f"{'enabled' if HAVE_PIL else 'not installed'}")
        self._log_line(f"Output root: {self.out_dir_var.get()}")
        self._log_line("Connect this machine to the 'aideck-stream' AP, then run Link check.")

    # --- setup ---------------------------------------------------------------

    def _build_vars(self) -> None:
        self.ip_var = tk.StringVar(value=core.DEFAULT_IPS[0])
        self.fallback_var = tk.BooleanVar(value=True)
        self.port_var = tk.IntVar(value=core.DEFAULT_PORT)
        self.duration_var = tk.DoubleVar(value=45.0)
        self.max_frames_var = tk.IntVar(value=2)
        self.attempts_var = tk.IntVar(value=8)
        self.delay_var = tk.DoubleVar(value=1.0)
        self.tp_duration_var = tk.DoubleVar(value=120.0)
        self.save_every_var = tk.IntVar(value=25)
        self.stall_gap_var = tk.DoubleVar(value=3.0)
        self.read_timeout_var = tk.DoubleVar(value=6.0)
        self.connect_timeout_var = tk.DoubleVar(value=5.0)
        self.zip_var = tk.BooleanVar(value=True)
        self.out_dir_var = tk.StringVar(value=str(core.DEFAULT_OUT_DIR))
        self.status_var = tk.StringVar(value="Idle")
        self.note_var = tk.StringVar(value="")

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=OFFWHITE)
        style.configure("TLabel", background=OFFWHITE, foreground=CHARCOAL)
        style.configure("Header.TLabel", background=NAVY, foreground=WHITE,
                        font=("Segoe UI", 13, "bold"), padding=10)
        style.configure("Section.TLabelframe", background=OFFWHITE)
        style.configure("Section.TLabelframe.Label", background=OFFWHITE,
                        foreground=NAVY, font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Run.TButton", font=("Segoe UI", 9, "bold"))
        style.configure("Status.TLabel", background=LIGHT, foreground=CHARCOAL,
                        padding=6, font=("Segoe UI", 9))

    def _build_layout(self) -> None:
        ttk.Label(self.root, text="AI Deck WiFi Test Bed", style="Header.TLabel",
                  anchor="w").pack(fill="x")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_target(left)
        self._build_params(left)
        self._build_actions(left)
        self._build_preview(left)
        self._build_log(right)

        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel",
                  anchor="w").pack(fill="x", side="bottom")

    def _build_target(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Target", style="Section.TLabelframe", padding=8)
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Deck IP").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.ip_var, width=18).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="Port").grid(row=1, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.port_var, width=18).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(box, text=f"also try {core.DEFAULT_IPS[1]}",
                        variable=self.fallback_var).grid(row=2, column=0, columnspan=2,
                                                         sticky="w", pady=(4, 0))

    def _build_params(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Parameters", style="Section.TLabelframe", padding=8)
        box.pack(fill="x", pady=(0, 8))
        rows = [
            ("Packet: duration s", self.duration_var),
            ("Packet: max frames (0=all)", self.max_frames_var),
            ("Reconnect: attempts", self.attempts_var),
            ("Reconnect: delay s", self.delay_var),
            ("Throughput: duration s", self.tp_duration_var),
            ("Throughput: save every N", self.save_every_var),
            ("Stall gap threshold s", self.stall_gap_var),
            ("Connect timeout s", self.connect_timeout_var),
            ("Read timeout s", self.read_timeout_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(box, textvariable=var, width=10).grid(row=row, column=1, sticky="e")

    def _build_actions(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Run", style="Section.TLabelframe", padding=8)
        box.pack(fill="x", pady=(0, 8))

        self.buttons: dict[str, ttk.Button] = {}
        for key in ("link", "packet", "reconnect", "throughput"):
            label = tests.TESTS[key][0]
            btn = ttk.Button(box, text=label, style="Run.TButton",
                             command=lambda k=key: self.start_test(k))
            btn.pack(fill="x", pady=2)
            self.buttons[key] = btn

        self.stop_button = ttk.Button(box, text="Stop", command=self.stop_test,
                                      state="disabled")
        self.stop_button.pack(fill="x", pady=(8, 2))

        ttk.Checkbutton(box, text="Zip run folder when finished",
                        variable=self.zip_var).pack(anchor="w", pady=(6, 0))

        out = ttk.Frame(box)
        out.pack(fill="x", pady=(6, 0))
        ttk.Button(out, text="Output folder...", command=self.choose_out_dir).pack(
            side="left", fill="x", expand=True)
        ttk.Button(out, text="Open last run", command=self.open_last_run).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

    def _build_preview(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Latest frame", style="Section.TLabelframe",
                             padding=8)
        box.pack(fill="both", expand=True)
        self.preview = tk.Canvas(box, width=PREVIEW_W, height=PREVIEW_H,
                                 bg=LIGHT, highlightthickness=0)
        self.preview.pack()
        self.preview_note = ttk.Label(box, textvariable=self.note_var, anchor="w")
        self.preview_note.pack(fill="x", pady=(4, 0))
        self._clear_preview("no frames yet")

    def _build_log(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Run log", foreground=NAVY,
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.log = tk.Text(frame, wrap="none", bg=WHITE, fg=CHARCOAL,
                           font=("Consolas", 9), relief="flat", borderwidth=1)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=self.log.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.log.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set,
                           state="disabled")

        buttons = ttk.Frame(parent)
        buttons.grid(row=2, column=0, sticky="e", pady=(6, 0))
        ttk.Button(buttons, text="Clear log", command=self.clear_log).pack(side="left")

    # --- test control --------------------------------------------------------

    def _build_config(self) -> tests.TestConfig:
        ips = [self.ip_var.get().strip()]
        if self.fallback_var.get() and core.DEFAULT_IPS[1] not in ips:
            ips.append(core.DEFAULT_IPS[1])
        return tests.TestConfig(
            ips=[ip for ip in ips if ip],
            port=int(self.port_var.get()),
            connect_timeout=float(self.connect_timeout_var.get()),
            read_timeout=float(self.read_timeout_var.get()),
            duration=float(self.duration_var.get()),
            max_frames=int(self.max_frames_var.get()),
            attempts=int(self.attempts_var.get()),
            delay=float(self.delay_var.get()),
            throughput_duration=float(self.tp_duration_var.get()),
            save_every=int(self.save_every_var.get()),
            stall_gap_s=float(self.stall_gap_var.get()),
        )

    def start_test(self, key: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Busy", "A test is already running.")
            return
        try:
            cfg = self._build_config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Bad parameter", f"Check the parameter fields.\n\n{exc}")
            return
        if not cfg.ips:
            messagebox.showerror("Bad parameter", "Enter a deck IP.")
            return

        label, func, prefix = tests.TESTS[key]
        out_dir = Path(self.out_dir_var.get())
        try:
            run_dir = core.new_run_dir(out_dir, prefix)
        except OSError as exc:
            messagebox.showerror("Output folder", f"Cannot create a run folder.\n\n{exc}")
            return

        self.current_run_dir = run_dir
        self._frames_seen = 0
        self.cancel = core.CancelToken()
        self._set_running(True, label)
        self._log_line("")
        self._log_line(f"----- {label} -> {run_dir.name} -----")

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(key, label, func, cfg, run_dir, self.cancel),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, key, label, func, cfg, run_dir, cancel) -> None:
        summary = None
        try:
            reporter = core.Reporter(
                run_dir, tests.TESTS[key][2],
                on_line=lambda line: self.events.put(("log", line)),
            )
            with reporter:
                kwargs = {}
                if key != "link":
                    kwargs["on_frame"] = self._on_frame
                if key == "throughput":
                    kwargs["on_progress"] = lambda *a: self.events.put(("progress", a))
                summary = func(cfg, run_dir, reporter, cancel, **kwargs)
        except Exception:  # noqa: BLE001 - surface the traceback into the log
            self.events.put(("log", "UNHANDLED ERROR:"))
            for line in traceback.format_exc().splitlines():
                self.events.put(("log", line))
        finally:
            if self.zip_var.get():
                try:
                    archive = core.zip_run_dir(run_dir)
                    self.events.put(("log", f"Zipped: {archive.name}"))
                except Exception as exc:  # noqa: BLE001
                    self.events.put(("log", f"Zip failed: {exc!r}"))
            self.events.put(("done", (label, run_dir, summary)))

    def stop_test(self) -> None:
        if self.cancel is not None:
            self.cancel.cancel()
            self._log_line("Stop requested; finishing the current read...")
            self.stop_button.configure(state="disabled")

    def _set_running(self, running: bool, label: str = "") -> None:
        state = "disabled" if running else "normal"
        for btn in self.buttons.values():
            btn.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.status_var.set(f"Running: {label}" if running else "Idle")

    # --- event pump ----------------------------------------------------------

    def _on_frame(self, data: bytes, meta: dict) -> None:
        self._frames_seen += 1
        self.events.put(("frame", (data, meta, self._frames_seen)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log_line(payload)
                elif kind == "frame":
                    self._show_frame(*payload)
                elif kind == "progress":
                    elapsed, frames, fps, rate = payload
                    self.status_var.set(
                        f"Running: t={elapsed:.0f}s frames={frames} "
                        f"fps={fps:.2f} {rate:.0f} KiB/s"
                    )
                elif kind == "done":
                    self._on_done(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _on_done(self, label: str, run_dir: Path, summary: dict | None) -> None:
        self.last_run_dir = run_dir
        self._set_running(False)
        verdict = self._verdict(summary)
        self._log_line(f"----- {label} finished: {verdict} -----")
        self._log_line(f"Artifacts: {run_dir}")
        self.status_var.set(f"Idle | last: {label} - {verdict}")

    @staticmethod
    def _verdict(summary: dict | None) -> str:
        if not summary:
            return "no summary (see log)"
        test = summary.get("test")
        if test == "link_check":
            return f"reachable={summary.get('reachable') or 'none'}"
        if test == "packet_test":
            parts = []
            for r in summary.get("results", []):
                parts.append(
                    f"{r['ip']} frames={r['completed_frames']} "
                    f"packets={r['packets']} stall={r['stall'] is not None}"
                )
            return "; ".join(parts) or "no candidates"
        if test == "reconnect_test":
            return f"{summary.get('ok_count')}/{summary.get('attempts_run')} attempts OK"
        if test == "throughput_test":
            return (
                f"{summary.get('frames')} frames, {summary.get('fps')} fps, "
                f"{summary.get('kib_per_s')} KiB/s, "
                f"stalled={summary.get('stall') is not None}"
            )
        return "done"

    # --- widgets -------------------------------------------------------------

    def _log_line(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _clear_preview(self, note: str) -> None:
        self.preview.delete("all")
        self.preview.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text=note,
                                 fill=CHARCOAL, font=("Segoe UI", 9))
        self.note_var.set("")

    def _show_frame(self, data: bytes, meta: dict, index: int) -> None:
        note = (f"frame {index}: {meta.get('width')}x{meta.get('height')} "
                f"{meta.get('encoding')} {meta.get('size')} bytes")
        if not HAVE_PIL:
            self._clear_preview("frame saved (install Pillow for preview)")
            self.note_var.set(note)
            return
        try:
            import io

            image = Image.open(io.BytesIO(data))
            image.load()
            image.thumbnail((PREVIEW_W, PREVIEW_H))
            self._preview_image = ImageTk.PhotoImage(image)
            self.preview.delete("all")
            self.preview.create_image(PREVIEW_W // 2, PREVIEW_H // 2,
                                      image=self._preview_image)
            self.note_var.set(note)
        except Exception as exc:  # noqa: BLE001 - a corrupt frame is a result, not a crash
            self._clear_preview("frame saved (not decodable)")
            self.note_var.set(f"{note} - decode failed: {exc}")

    def choose_out_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.out_dir_var.get(),
                                         title="Run output folder")
        if chosen:
            self.out_dir_var.set(chosen)
            self._log_line(f"Output root: {chosen}")

    def open_last_run(self) -> None:
        target = self.last_run_dir or Path(self.out_dir_var.get())
        if not Path(target).exists():
            messagebox.showinfo("Not found", f"{target} does not exist yet.")
            return
        core.run_text(["explorer", str(target)], timeout=5)


def main() -> int:
    root = tk.Tk()
    TestBedApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
