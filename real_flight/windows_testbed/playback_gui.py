"""Play back an AI Deck recording made by Flight Watch.

Point it at a `flightwatch_*` run and it replays the frames at the speed they
actually arrived, using the timings in `frames.csv` rather than a fixed rate.
That matters: the deck delivers around 7.5 fps and stutters, and a player that
assumed a constant rate would hide exactly the gaps worth seeing.

If a `fly.py` flight log overlaps the recording in time, its height and drift
are shown alongside each frame. The two are aligned by the timestamps in their
own filenames -- the recording folder is named for when recording started, the
`.npz` for when the flight began -- so the offset between them is known without
either side having to agree on a clock.

Standard library plus Pillow. Pillow is required here; without it there is
nothing to show.
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import aideck_core as core

try:
    from PIL import Image, ImageTk

    PIL_OK = True
except ImportError:  # pragma: no cover
    PIL_OK = False

try:
    import numpy as np

    NUMPY_OK = True
except ImportError:  # pragma: no cover
    NUMPY_OK = False

NAVY = "#2C3E5C"
INK = "#2B2B2B"
MIST = "#D5E2EC"
PAPER = "#F5F5F0"
GOOD = "#1E6B3A"
WARN = "#8A6D1F"
MUTED = "#6B7280"

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


def parse_stamp(name: str) -> datetime | None:
    """Pull a YYYYmmdd_HHMMSS stamp off the end of a file or folder name."""
    tail = name.rsplit("_", 2)
    if len(tail) < 3:
        return None
    try:
        return datetime.strptime(f"{tail[-2]}_{tail[-1]}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


class Recording:
    """One recorded run: its frames, their timings, and any matching flight log."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.frames: list[dict] = []
        csv_path = run_dir / "frames.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    path = run_dir / row["file"]
                    if path.exists():
                        self.frames.append({"t": float(row["t_rel_s"]), "path": path})
        else:
            # No manifest: fall back to file order at the deck's nominal rate.
            for i, path in enumerate(sorted(run_dir.glob("frame_*.jpg"))):
                self.frames.append({"t": i / 7.5, "path": path})
        self.started = parse_stamp(run_dir.name)
        self.flight = None
        self.flight_offset = 0.0
        self._match_flight()

    @property
    def duration(self) -> float:
        return self.frames[-1]["t"] if self.frames else 0.0

    def _match_flight(self) -> None:
        """Find a fly.py log whose flight falls inside this recording."""
        if not (NUMPY_OK and self.started):
            return
        logs_dir = core.LAB_DIR / "real_flight" / "logs"
        if not logs_dir.exists():
            return
        for npz in sorted(logs_dir.glob("flight_*.npz")):
            stamp = parse_stamp(npz.stem)
            if stamp is None:
                continue
            offset = (stamp - self.started).total_seconds()
            if not (0.0 <= offset <= max(self.duration, 1.0)):
                continue
            try:
                data = np.load(npz, allow_pickle=True)
                self.flight = {"t": data["t"], "pos": data["pos"], "name": npz.name}
                self.flight_offset = offset
                return
            except Exception:  # noqa: BLE001
                continue

    def flight_at(self, t_rel: float) -> dict | None:
        """Flight state at recording-time `t_rel`, or None if not airborne then."""
        if self.flight is None:
            return None
        t_flight = t_rel - self.flight_offset
        ts = self.flight["t"]
        if t_flight < ts[0] or t_flight > ts[-1]:
            return None
        i = int(np.searchsorted(ts, t_flight))
        i = max(0, min(i, len(ts) - 1))
        pos = self.flight["pos"]
        drift = float(np.hypot(pos[i][0] - pos[0][0], pos[i][1] - pos[0][1]))
        return {"t": float(t_flight), "z": float(pos[i][2]), "drift": drift,
                "x": float(pos[i][0]), "y": float(pos[i][1])}


def find_recordings(root: Path) -> list[Path]:
    """Every run folder holding frames, newest first."""
    out = []
    for session in root.glob("flightwatch_*"):
        if not session.is_dir():
            continue
        for run in session.iterdir():
            if run.is_dir() and (run / "frames.csv").exists():
                out.append(run)
    return sorted(out, key=lambda p: p.name, reverse=True)


class Player:
    def __init__(self, root: tk.Tk, start_with: Path | None = None) -> None:
        self.root = root
        root.title("Playback -- AI Deck recordings")
        root.configure(bg=PAPER)
        root.minsize(880, 520)

        self.rec: Recording | None = None
        self.index = 0
        self.playing = False
        self.speed = 1.0
        self.zoom = 2
        self.photo = None
        self._after = None

        self._fonts()
        self._style()
        self._build()

        self.runs = find_recordings(core.DEFAULT_OUT_DIR)
        self._fill_list()
        target = start_with or (self.runs[0] if self.runs else None)
        if target:
            self.load(target)
        self._fit_window()

    def _fonts(self) -> None:
        self.f_h1 = tkfont.Font(family="Segoe UI Semibold", size=16)
        self.f_h2 = tkfont.Font(family="Segoe UI Semibold", size=11)
        self.f_body = tkfont.Font(family="Segoe UI", size=10)
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_mono = tkfont.Font(family="Consolas", size=10)
        self.f_big = tkfont.Font(family="Consolas", size=15)

    def _style(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TFrame", background=PAPER)
        s.configure("Card.TFrame", background="white")
        s.configure("TLabel", background=PAPER, foreground=INK, font=self.f_body)
        s.configure("Card.TLabel", background="white", foreground=INK, font=self.f_body)
        s.configure("H1.TLabel", background=PAPER, foreground=NAVY, font=self.f_h1)
        s.configure("H2.TLabel", background="white", foreground=NAVY, font=self.f_h2)
        s.configure("Muted.TLabel", background=PAPER, foreground=MUTED, font=self.f_small)
        s.configure("CardMuted.TLabel", background="white", foreground=MUTED, font=self.f_small)
        s.configure("Good.TLabel", background="white", foreground=GOOD, font=self.f_mono)
        s.configure("Warn.TLabel", background="white", foreground=WARN, font=self.f_small)
        s.configure("TButton", font=self.f_body, padding=(12, 6))
        s.configure("Go.TButton", font=self.f_h2, padding=(18, 8))
        s.configure("Horizontal.TScale", background=PAPER)

    def _build(self) -> None:
        head = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        head.pack(fill="x")
        ttk.Label(head, text="Playback", style="H1.TLabel").pack(anchor="w")
        self.v_sub = tk.StringVar(value="")
        ttk.Label(head, textvariable=self.v_sub, style="Muted.TLabel").pack(anchor="w")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 0))
        body.pack(fill="both", expand=True)

        side = ttk.Frame(body)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Recordings").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=30, height=12, font=self.f_small,
                                  relief="flat", highlightthickness=1)
        self.listbox.pack(fill="y", expand=True, pady=(4, 0))
        self.listbox.bind("<<ListboxSelect>>", self._pick)

        main = ttk.Frame(body, style="Card.TFrame", padding=14)
        main.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.canvas = tk.Label(main, background="#111418")
        self.canvas.pack(anchor="n")

        self.v_read = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.v_read, style="Good.TLabel").pack(anchor="w", pady=(10, 0))
        self.v_flight = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.v_flight, style="Card.TLabel",
                  font=self.f_big).pack(anchor="w", pady=(2, 0))
        self.v_note = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.v_note, style="CardMuted.TLabel",
                  wraplength=560, justify="left").pack(anchor="w", pady=(6, 0))

        self.scrub = ttk.Scale(main, from_=0, to=1, orient="horizontal", command=self._seek)
        self.scrub.pack(fill="x", pady=(12, 6))

        bar = ttk.Frame(main, style="Card.TFrame")
        bar.pack(anchor="w")
        self.b_play = ttk.Button(bar, text="Play", style="Go.TButton", command=self.toggle)
        self.b_play.pack(side="left")
        ttk.Button(bar, text="|<", width=4, command=lambda: self.step(-1e9)).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="<<", width=4, command=lambda: self.step(-10)).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="<", width=4, command=lambda: self.step(-1)).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text=">", width=4, command=lambda: self.step(1)).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text=">>", width=4, command=lambda: self.step(10)).pack(side="left", padx=(4, 0))

        ttk.Label(bar, text="   speed", style="Card.TLabel").pack(side="left")
        self.v_speed = tk.StringVar(value="1.0x")
        speed_box = ttk.Combobox(bar, textvariable=self.v_speed, width=6, state="readonly",
                                 values=[f"{s}x" for s in SPEEDS])
        speed_box.pack(side="left", padx=(6, 0))
        speed_box.bind("<<ComboboxSelected>>", self._set_speed)

        foot = ttk.Frame(self.root, padding=(16, 8, 16, 14))
        foot.pack(fill="x")
        ttk.Label(foot, text="Space plays and pauses. Left and right arrows step one frame.",
                  style="Muted.TLabel").pack(side="left")

        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<Left>", lambda e: self.step(-1))
        self.root.bind("<Right>", lambda e: self.step(1))

    def _fit_window(self) -> None:
        """Pick the largest frame zoom whose layout still fits, then size to it.

        A fixed geometry pushed the transport controls off the bottom of the
        screen on shorter displays, which made the player look broken. Measuring
        the layout instead means the buttons are always reachable.
        """
        max_w = int(self.root.winfo_screenwidth() * 0.92)
        max_h = int(self.root.winfo_screenheight() * 0.88)
        for zoom in (3, 2, 1):
            self.zoom = zoom
            self.show(self.index)
            self.root.update_idletasks()
            w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
            if (w <= max_w and h <= max_h) or zoom == 1:
                self.root.geometry(f"{min(w, max_w)}x{min(h, max_h)}")
                return

    def _fill_list(self) -> None:
        self.listbox.delete(0, "end")
        for run in self.runs:
            n = sum(1 for _ in run.glob("frame_*.jpg"))
            self.listbox.insert("end", f"{run.name}  ({n})")

    def _pick(self, _event) -> None:
        sel = self.listbox.curselection()
        if sel:
            self.load(self.runs[sel[0]])

    def load(self, run_dir: Path) -> None:
        self.pause()
        self.rec = Recording(run_dir)
        self.index = 0
        n = len(self.rec.frames)
        if not n:
            self.v_sub.set(f"{run_dir.name} -- no frames found")
            return
        self.scrub.configure(to=n - 1)
        rate = n / self.rec.duration if self.rec.duration else 0.0
        self.v_sub.set(f"{run_dir.name}   {n} frames   "
                       f"{self.rec.duration:.1f}s   {rate:.2f} fps")
        if self.rec.flight:
            self.v_note.set(
                f"Flight log {self.rec.flight['name']} lines up with this recording, "
                f"starting {self.rec.flight_offset:.0f}s in. Height and drift below come "
                "from the flow deck's own estimate, which has no outside reference."
            )
        else:
            self.v_note.set("No flight log overlaps this recording.")
        self.show(0)

    def show(self, i: int) -> None:
        if not self.rec or not self.rec.frames:
            return
        i = max(0, min(i, len(self.rec.frames) - 1))
        self.index = i
        frame = self.rec.frames[i]
        try:
            img = Image.open(frame["path"])
            img = img.resize((img.width * self.zoom, img.height * self.zoom), Image.NEAREST)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.configure(image=self.photo, width=img.width, height=img.height)
        except Exception as exc:  # noqa: BLE001
            self.canvas.configure(image="", text=f"cannot read frame: {exc}")
        self.v_read.set(f"frame {i + 1}/{len(self.rec.frames)}    t = {frame['t']:6.2f}s")
        state = self.rec.flight_at(frame["t"])
        if state:
            self.v_flight.set(f"height {state['z']:.3f} m     drift {state['drift']:.3f} m"
                              f"     (flight t {state['t']:+.1f}s)")
        else:
            self.v_flight.set("on the ground" if self.rec.flight else "")
        self.scrub.set(i)

    def _seek(self, value: str) -> None:
        i = int(float(value))
        if self.rec and i != self.index:
            self.show(i)

    def _set_speed(self, _event) -> None:
        self.speed = float(self.v_speed.get().rstrip("x"))

    def step(self, delta: float) -> None:
        self.pause()
        if delta <= -1e8:
            self.show(0)
        else:
            self.show(self.index + int(delta))

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def play(self) -> None:
        if not self.rec or not self.rec.frames:
            return
        if self.index >= len(self.rec.frames) - 1:
            self.show(0)
        self.playing = True
        self.b_play.configure(text="Pause")
        self._tick()

    def pause(self) -> None:
        self.playing = False
        if hasattr(self, "b_play"):
            self.b_play.configure(text="Play")
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None

    def _tick(self) -> None:
        if not self.playing or not self.rec:
            return
        frames = self.rec.frames
        if self.index >= len(frames) - 1:
            self.pause()
            return
        # Wait the gap this frame actually had, so stutters replay as stutters.
        gap = frames[self.index + 1]["t"] - frames[self.index]["t"]
        delay = max(1, int(gap * 1000 / self.speed))
        self.show(self.index + 1)
        self._after = self.root.after(delay, self._tick)


def main() -> None:
    if not PIL_OK:
        print("Pillow is required to play frames back.")
        print("Install it with:   py -3 -m pip install pillow")
        return
    start = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    root = tk.Tk()
    Player(root, start_with=start)
    root.mainloop()


if __name__ == "__main__":
    main()
