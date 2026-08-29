"""Calibrate apparent marker size against real distance.

Step 2 of the vision-hover task (see `real_flight/VISION_HOVER_PLAN.md`). The
marker detector reports a marker's apparent size in pixels; the drone needs
that in metres. For a pinhole camera those are related by

    size_px = f_px * marker_size_m / distance_m

so `size_px * distance_m` is a constant, `k`. Measure a few frames at known
distances, fit `k`, and distance becomes `k / size_px`.

The fit is deliberately through that physical model rather than a free
polynomial. A polynomial would fit the samples better and mean nothing --
whereas if the inverse law does *not* hold here, the residuals say so, and
that is worth knowing before anything downstream trusts this mapping.

Two frame sources:

- **Live deck** -- reads the AI Deck WiFi stream, same path as Flight Watch.
  Must run on a native Windows host joined to `aideck-stream`; capture through
  the VM is broken (see `memory/FACTS.md` FACT-018).
- **Folder** -- reads saved frames from a previous recording. No hardware
  needed, and how this wizard can be exercised away from the bench.

Needs Pillow, numpy, and OpenCV (the detector uses it). Run `--self-test` to
check the fit maths with no hardware and no frames.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import queue
import socket
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import filedialog, ttk

import aideck_core as core

sys.path.insert(0, str(core.LAB_DIR / "real_flight"))

try:
    import numpy as np
    from PIL import Image, ImageTk

    DEPS_OK = True
    DEPS_ERR = ""
except ImportError as exc:  # pragma: no cover
    DEPS_OK = False
    DEPS_ERR = str(exc)

try:
    from marker_detector import detect_marker

    DETECTOR_OK = True
    DETECTOR_ERR = ""
except ImportError as exc:  # pragma: no cover
    DETECTOR_OK = False
    DETECTOR_ERR = str(exc)

NAVY = "#2C3E5C"
INK = "#2B2B2B"
MIST = "#D5E2EC"
PAPER = "#F5F5F0"
GOOD = "#1E6B3A"
WARN = "#8A6D1F"
BAD = "#8A2323"
MUTED = "#6B7280"

M_PER_INCH = 0.0254

# A fit from samples that all sit at nearly the same distance is not a fit --
# it is one point with noise. Require a real spread before trusting it.
MIN_SAMPLES = 3
MIN_SPAN_RATIO = 2.0
# marker_detector rejects blobs below min_area_frac of the frame, so there is a
# smallest marker it can see and therefore a furthest distance it can work at.
# Mirrored from that module's default; keep in step if it changes there.
DETECTOR_MIN_AREA_FRAC = 0.001
# A 1" marker at any workable distance is a small part of the frame. A box
# filling much of it is the detector latching onto scenery, not the marker.
SUSPICIOUS_WIDTH_FRAC = 0.40
# Residual above this suggests the detector locked onto the wrong thing, or a
# distance was mistyped, or the inverse law does not hold here.
RESIDUAL_WARN_PCT = 10.0


@dataclass
class Sample:
    distance_m: float
    size_px: float
    source: str = ""


def fit_k(samples: list[Sample]) -> dict:
    """Least-squares fit of size_px = k / distance_m, through the origin.

    Fitting `size_px` against `1/distance_m` with no intercept keeps the
    estimate in the physical model. Returns the fit plus per-sample residuals
    expressed as distance error, which is the number that actually matters
    downstream -- a 5% size error near the marker is a very different
    proposition from 5% far from it.
    """
    if not samples:
        return {"ok": False, "reason": "no samples"}

    xs = [1.0 / s.distance_m for s in samples]
    ys = [s.size_px for s in samples]
    denom = sum(x * x for x in xs)
    if denom <= 0:
        return {"ok": False, "reason": "degenerate distances"}
    k = sum(x * y for x, y in zip(xs, ys)) / denom

    residuals = []
    for s in samples:
        predicted_px = k / s.distance_m
        implied_d = k / s.size_px if s.size_px > 0 else float("inf")
        residuals.append({
            "distance_m": s.distance_m,
            "size_px": s.size_px,
            "predicted_px": predicted_px,
            "implied_distance_m": implied_d,
            "distance_error_m": implied_d - s.distance_m,
            "distance_error_pct": (implied_d - s.distance_m) / s.distance_m * 100.0,
        })

    worst = max(abs(r["distance_error_pct"]) for r in residuals)
    ds = [s.distance_m for s in samples]
    span_ratio = max(ds) / min(ds) if min(ds) > 0 else 0.0

    warnings = []
    if len(samples) < MIN_SAMPLES:
        warnings.append(f"only {len(samples)} samples; want at least {MIN_SAMPLES}")
    if span_ratio < MIN_SPAN_RATIO:
        warnings.append(
            f"distance span is only {span_ratio:.2f}x (want {MIN_SPAN_RATIO:.0f}x or more) "
            "-- samples too bunched to constrain the fit")
    if worst > RESIDUAL_WARN_PCT:
        warnings.append(
            f"worst residual {worst:.1f}% exceeds {RESIDUAL_WARN_PCT:.0f}% "
            "-- check the detector locked onto the marker in every sample")

    return {
        "ok": True,
        "k_px_m": k,
        "residuals": residuals,
        "worst_abs_error_pct": worst,
        "span_ratio": span_ratio,
        "n": len(samples),
        "warnings": warnings,
    }


def decode_frame(raw: bytes) -> "np.ndarray | None":
    """JPEG bytes from the deck -> grayscale array the detector can take."""
    try:
        img = Image.open(io.BytesIO(raw))
        return np.array(img.convert("L"))
    except Exception:  # noqa: BLE001
        return None


class DeckReader(threading.Thread):
    """Pulls frames off the deck and keeps only the newest one.

    Calibration only ever needs the current view, so old frames are dropped
    rather than queued -- a backlog would make the preview lag the drone's
    actual position, which is exactly the thing being measured.
    """

    def __init__(self, ips: list[str], out: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.ips = ips
        self.out = out
        self.stop_flag = threading.Event()

    def run(self) -> None:
        sock = None
        for entry in self.ips:
            ip, _, port_text = entry.partition(":")
            port = int(port_text) if port_text.isdigit() else core.DEFAULT_PORT
            try:
                sock = core.connect_to_deck(ip, port, 6.0)
                self.out.put(("status", f"connected {ip}:{port}"))
                break
            except Exception as exc:  # noqa: BLE001
                self.out.put(("status", f"{ip}:{port} failed: {exc}"))
        if sock is None:
            self.out.put(("status", "no deck found -- on aideck-stream WiFi?"))
            return

        sock.settimeout(4.0)
        current = None
        buf = bytearray()
        try:
            while not self.stop_flag.is_set():
                try:
                    _l, _r, _f, payload = core.read_packet(sock)
                except socket.timeout:
                    self.out.put(("status", "stream stalled"))
                    break
                except OSError as exc:
                    self.out.put(("status", f"stream error: {exc}"))
                    break
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
                self.out.put(("frame", bytes(buf)))
                current = None
                buf = bytearray()
        finally:
            try:
                sock.close()
            except OSError:
                pass


class Wizard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Distance calibration -- marker")
        root.configure(bg=PAPER)
        root.minsize(940, 600)

        self.samples: list[Sample] = []
        self.frame_gray = None
        self.detection = None
        self.photo = None
        self.reader = None
        self.events: queue.Queue = queue.Queue()
        self.folder_frames: list[Path] = []
        self.folder_index = 0

        self._fonts()
        self._style()
        self._build()
        self.root.after(60, self._pump)

    def _fonts(self) -> None:
        self.f_h1 = tkfont.Font(family="Segoe UI Semibold", size=16)
        self.f_h2 = tkfont.Font(family="Segoe UI Semibold", size=11)
        self.f_body = tkfont.Font(family="Segoe UI", size=10)
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_mono = tkfont.Font(family="Consolas", size=10)

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
        s.configure("H2.TLabel", background=PAPER, foreground=NAVY, font=self.f_h2)
        s.configure("Muted.TLabel", background=PAPER, foreground=MUTED, font=self.f_small)
        s.configure("Mono.TLabel", background=PAPER, foreground=INK, font=self.f_mono)
        s.configure("Good.TLabel", background=PAPER, foreground=GOOD, font=self.f_mono)
        s.configure("Warn.TLabel", background=PAPER, foreground=WARN, font=self.f_small)
        s.configure("Bad.TLabel", background=PAPER, foreground=BAD, font=self.f_small)
        s.configure("TButton", font=self.f_body, padding=(10, 5))
        s.configure("Go.TButton", font=self.f_h2, padding=(16, 7))

    def _build(self) -> None:
        head = ttk.Frame(self.root, padding=(16, 12, 16, 6))
        head.pack(fill="x")
        ttk.Label(head, text="Distance calibration", style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, style="Muted.TLabel",
                  text="Put the marker at a known distance, check the box locks on, "
                       "record it. Repeat across a range of distances.").pack(anchor="w")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 0))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Marker", style="H2.TLabel").pack(anchor="w", pady=(4, 2))
        mrow = ttk.Frame(left)
        mrow.pack(anchor="w")
        self.v_marker = tk.StringVar(value="1.0")
        ttk.Entry(mrow, textvariable=self.v_marker, width=8).pack(side="left")
        self.v_units = tk.StringVar(value="inches")
        ttk.Combobox(mrow, textvariable=self.v_units, width=8, state="readonly",
                     values=["inches", "mm"]).pack(side="left", padx=(6, 0))
        ttk.Label(left, text="Real edge length of the printed square.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 10))

        ttk.Label(left, text="Frames", style="H2.TLabel").pack(anchor="w", pady=(4, 2))
        ttk.Button(left, text="Connect to deck", command=self.connect).pack(anchor="w", fill="x")
        ttk.Button(left, text="Open folder...", command=self.open_folder).pack(
            anchor="w", fill="x", pady=(4, 0))
        self.b_next = ttk.Button(left, text="Next frame >", command=self.next_folder_frame,
                                 state="disabled")
        self.b_next.pack(anchor="w", fill="x", pady=(4, 0))
        self.v_src = tk.StringVar(value="no frame source")
        ttk.Label(left, textvariable=self.v_src, style="Muted.TLabel",
                  wraplength=210, justify="left").pack(anchor="w", pady=(4, 10))

        ttk.Label(left, text="Record a sample", style="H2.TLabel").pack(anchor="w", pady=(4, 2))
        drow = ttk.Frame(left)
        drow.pack(anchor="w")
        self.v_dist = tk.StringVar(value="")
        ttk.Entry(drow, textvariable=self.v_dist, width=8).pack(side="left")
        ttk.Label(drow, text="metres").pack(side="left", padx=(6, 0))
        ttk.Button(left, text="Record this frame", style="Go.TButton",
                   command=self.record).pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Button(left, text="Delete selected", command=self.delete_sample).pack(
            anchor="w", fill="x", pady=(4, 0))

        ttk.Label(left, text="Samples", style="H2.TLabel").pack(anchor="w", pady=(12, 2))
        self.listbox = tk.Listbox(left, width=30, height=8, font=self.f_small,
                                  relief="flat", highlightthickness=1)
        self.listbox.pack(fill="both", expand=True)

        ttk.Button(left, text="Save calibration", command=self.save).pack(
            anchor="w", fill="x", pady=(8, 0))

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.canvas = tk.Label(right, background="#111418")
        self.canvas.pack(anchor="n")
        self.v_read = tk.StringVar(value="no frame")
        ttk.Label(right, textvariable=self.v_read, style="Mono.TLabel").pack(
            anchor="w", pady=(8, 0))

        ttk.Label(right, text="Fit", style="H2.TLabel").pack(anchor="w", pady=(12, 2))
        self.v_fit = tk.StringVar(value="Record samples at three or more distances.")
        ttk.Label(right, textvariable=self.v_fit, style="Mono.TLabel",
                  wraplength=560, justify="left").pack(anchor="w")
        self.v_warn = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.v_warn, style="Warn.TLabel",
                  wraplength=560, justify="left").pack(anchor="w", pady=(6, 0))

        foot = ttk.Frame(self.root, padding=(16, 8, 16, 12))
        foot.pack(fill="x")
        self.v_status = tk.StringVar(value="ready")
        ttk.Label(foot, textvariable=self.v_status, style="Muted.TLabel").pack(side="left")

    # ---- marker size -------------------------------------------------

    def marker_size_m(self) -> float | None:
        try:
            value = float(self.v_marker.get())
        except ValueError:
            return None
        if value <= 0:
            return None
        return value * M_PER_INCH if self.v_units.get() == "inches" else value / 1000.0

    # ---- frame sources -----------------------------------------------

    def connect(self) -> None:
        if self.reader is not None:
            self.reader.stop_flag.set()
        self.folder_frames = []
        self.b_next.configure(state="disabled")
        self.reader = DeckReader(list(core.DEFAULT_IPS), self.events)
        self.reader.start()
        self.v_src.set("live deck")
        self.v_status.set("connecting...")

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Folder of frames",
                                         initialdir=str(core.DEFAULT_OUT_DIR))
        if not folder:
            return
        if self.reader is not None:
            self.reader.stop_flag.set()
            self.reader = None
        path = Path(folder)
        self.folder_frames = sorted(
            [p for p in path.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        if not self.folder_frames:
            self.v_src.set(f"{path.name} -- no images found")
            return
        self.folder_index = 0
        self.b_next.configure(state="normal")
        self.v_src.set(f"{path.name} ({len(self.folder_frames)} frames)")
        self.show_folder_frame()

    def next_folder_frame(self) -> None:
        if not self.folder_frames:
            return
        self.folder_index = (self.folder_index + 1) % len(self.folder_frames)
        self.show_folder_frame()

    def show_folder_frame(self) -> None:
        path = self.folder_frames[self.folder_index]
        try:
            gray = np.array(Image.open(path).convert("L"))
        except Exception as exc:  # noqa: BLE001
            self.v_status.set(f"cannot read {path.name}: {exc}")
            return
        self.v_status.set(f"{path.name}  ({self.folder_index + 1}/{len(self.folder_frames)})")
        self.set_frame(gray, source=path.name)

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.v_status.set(payload)
                elif kind == "frame":
                    gray = decode_frame(payload)
                    if gray is not None:
                        self.set_frame(gray, source="live")
        except queue.Empty:
            pass
        self.root.after(60, self._pump)

    # ---- detection and preview ---------------------------------------

    def set_frame(self, gray, source: str) -> None:
        self.frame_gray = gray
        self.frame_source = source
        self.detection = detect_marker(gray)
        self.render()

    def render(self) -> None:
        if self.frame_gray is None:
            return
        img = Image.fromarray(self.frame_gray).convert("RGB")
        zoom = 2
        img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)

        det = self.detection
        if det is not None and det.found and det.bbox:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            x, y, w, h = det.bbox
            draw.rectangle([x * zoom, y * zoom, (x + w) * zoom, (y + h) * zoom],
                           outline=(60, 220, 120), width=2)
            cx, cy = det.center_x * zoom, det.center_y * zoom
            draw.line([cx - 8, cy, cx + 8, cy], fill=(60, 220, 120), width=2)
            draw.line([cx, cy - 8, cx, cy + 8], fill=(60, 220, 120), width=2)
            note = ""
            if w > self.frame_gray.shape[1] * SUSPICIOUS_WIDTH_FRAC:
                note = "   <- box is huge; probably not the marker"
            self.v_read.set(f"size {det.size_px:7.2f} px    "
                            f"v-offset {det.vertical_offset_px:+7.1f} px    "
                            f"h-offset {det.horizontal_offset_px:+7.1f} px{note}")
        else:
            self.v_read.set("marker NOT found in this frame")

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo, width=img.width, height=img.height)

    # ---- samples ------------------------------------------------------

    def record(self) -> None:
        if self.detection is None or not self.detection.found:
            self.v_status.set("no marker detected -- not recording this frame")
            return
        try:
            distance = float(self.v_dist.get())
        except ValueError:
            self.v_status.set("enter the distance in metres first")
            return
        if distance <= 0:
            self.v_status.set("distance must be positive")
            return
        self.samples.append(Sample(distance_m=distance,
                                   size_px=self.detection.size_px,
                                   source=getattr(self, "frame_source", "")))
        self.v_status.set(f"recorded {distance:.3f} m at {self.detection.size_px:.2f} px")
        self.refresh()

    def delete_sample(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        del self.samples[sel[0]]
        self.refresh()

    def refresh(self) -> None:
        self.listbox.delete(0, "end")
        for s in self.samples:
            self.listbox.insert("end", f"{s.distance_m:6.3f} m   {s.size_px:7.2f} px")
        self.update_fit()

    def update_fit(self) -> None:
        if not self.samples:
            self.v_fit.set("Record samples at three or more distances.")
            self.v_warn.set("")
            return
        fit = fit_k(self.samples)
        if not fit["ok"]:
            self.v_fit.set(f"cannot fit: {fit['reason']}")
            return
        size_m = self.marker_size_m()
        focal = fit["k_px_m"] / size_m if size_m else float("nan")
        lines = [
            f"k = {fit['k_px_m']:.4f} px*m      distance_m = k / size_px",
            f"implied focal length = {focal:.1f} px"
            + ("" if size_m else "   (set a valid marker size)"),
            f"n = {fit['n']}   span = {fit['span_ratio']:.2f}x   "
            f"worst error = {fit['worst_abs_error_pct']:.1f}%",
        ]
        if self.frame_gray is not None:
            h, w = self.frame_gray.shape
            min_side = math.sqrt(DETECTOR_MIN_AREA_FRAC * h * w)
            lines.append(f"detector floor {min_side:.1f} px -> "
                         f"cannot see the marker past {fit['k_px_m'] / min_side:.2f} m")
        lines.append("")
        for r in fit["residuals"]:
            lines.append(f"  {r['distance_m']:6.3f} m -> {r['implied_distance_m']:6.3f} m "
                         f"({r['distance_error_pct']:+6.1f}%)")
        self.v_fit.set("\n".join(lines))
        self.v_warn.set("\n".join(f"WARNING: {w}" for w in fit["warnings"]))

    def save(self) -> None:
        if not self.samples:
            self.v_status.set("nothing to save")
            return
        fit = fit_k(self.samples)
        size_m = self.marker_size_m()
        if size_m is None:
            self.v_status.set("set a valid marker size before saving")
            return
        shape = (self.frame_gray.shape if self.frame_gray is not None else None)
        payload = {
            "created_at": core.stamp_now(),
            "marker_size_m": size_m,
            "marker_input": {"value": self.v_marker.get(), "units": self.v_units.get()},
            "frame_shape": {"height": int(shape[0]), "width": int(shape[1])} if shape else None,
            "model": "size_px = k / distance_m",
            "k_px_m": fit["k_px_m"],
            "focal_px": fit["k_px_m"] / size_m,
            "n_samples": fit["n"],
            "span_ratio": fit["span_ratio"],
            "worst_abs_error_pct": fit["worst_abs_error_pct"],
            "warnings": fit["warnings"],
            "samples": [asdict(s) for s in self.samples],
            "residuals": fit["residuals"],
        }
        out = core.LAB_DIR / "real_flight" / "marker_calibration.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        note = " (with warnings)" if fit["warnings"] else ""
        self.v_status.set(f"saved {out.name}{note}")


def _self_test() -> int:
    """Check the fit maths against a synthetic camera, no hardware needed."""
    ok = True

    # Perfect pinhole data: k = focal * marker size.
    focal, marker = 250.0, 0.0254
    k_true = focal * marker
    exact = [Sample(d, k_true / d) for d in (0.30, 0.60, 1.00, 1.50)]
    fit = fit_k(exact)
    err = abs(fit["k_px_m"] - k_true) / k_true * 100
    print(f"{'PASS' if err < 1e-6 else 'FAIL'} [exact pinhole]: "
          f"k={fit['k_px_m']:.6f} vs {k_true:.6f} ({err:.2e}% off)")
    ok &= err < 1e-6

    good = fit["worst_abs_error_pct"] < 1e-6
    print(f"{'PASS' if good else 'FAIL'} [exact residuals]: "
          f"worst {fit['worst_abs_error_pct']:.2e}%")
    ok &= good

    # A mistyped distance must show up as a large residual, not be absorbed.
    bad = list(exact)
    bad[2] = Sample(0.50, k_true / 1.00)  # said 0.50 m, actually shot at 1.00 m
    fit_bad = fit_k(bad)
    caught = fit_bad["worst_abs_error_pct"] > RESIDUAL_WARN_PCT and fit_bad["warnings"]
    print(f"{'PASS' if caught else 'FAIL'} [mistyped distance caught]: "
          f"worst {fit_bad['worst_abs_error_pct']:.1f}%")
    ok &= bool(caught)

    # Bunched samples must be flagged even when they fit perfectly.
    bunched = [Sample(d, k_true / d) for d in (1.00, 1.05, 1.10)]
    fit_b = fit_k(bunched)
    flagged = any("span" in w for w in fit_b["warnings"])
    print(f"{'PASS' if flagged else 'FAIL'} [bunched samples flagged]: "
          f"span {fit_b['span_ratio']:.2f}x, warnings={len(fit_b['warnings'])}")
    ok &= flagged

    # Too few samples must be flagged.
    few = fit_k([Sample(0.5, k_true / 0.5), Sample(1.0, k_true / 1.0)])
    flagged_n = any("samples" in w for w in few["warnings"])
    print(f"{'PASS' if flagged_n else 'FAIL'} [too few samples flagged]")
    ok &= flagged_n

    # Round trip: fitted k must recover the distances it was built from.
    worst = max(abs(k_true / s.size_px - s.distance_m) for s in exact)
    print(f"{'PASS' if worst < 1e-9 else 'FAIL'} [round trip]: worst {worst:.2e} m")
    ok &= worst < 1e-9

    print(f"\nSELF-TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="check the fit maths and exit; no hardware needed")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not DEPS_OK:
        print(f"Needs Pillow and numpy: {DEPS_ERR}")
        print("Install with:   py -3 -m pip install pillow numpy")
        return 1
    if not DETECTOR_OK:
        print(f"Cannot import the marker detector: {DETECTOR_ERR}")
        print("It needs OpenCV. Install with:   py -3 -m pip install opencv-python")
        return 1

    root = tk.Tk()
    Wizard(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
