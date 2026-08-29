"""Generate the printable vision-hover marker as a PDF.

A solid black square of an exact known edge length on US Letter. The square's
real-world size is the whole basis of the distance calibration
(`windows_testbed/marker_calibration_gui.py` fits `size_px * distance_m = k`,
and `k = focal_px * marker_size_m`), so if the print is scaled, every distance
the drone believes is wrong by the same factor and nothing downstream can tell.

Hence: the PDF is written by hand rather than through a drawing library, so the
geometry is exactly what it says it is. PDF user units are 1/72 inch, so a
1.000 in square is literally `72 72 re`. The page also carries a ruler to catch
a scaled print, because "Fit to Page" is on by default in a lot of print
dialogs and silently shrinks by a few percent.

    python3 make_marker.py                  # 1.0 in square -> marker_1in.pdf
    python3 make_marker.py --size-in 2.0    # bigger square, longer usable range
    python3 make_marker.py --self-test      # verify geometry, no printer needed

On marker size -- print the 4 in, not the 1 in. The binding constraint is not
the detector's area floor, it is Otsu. `detect_marker` thresholds globally, and
when the black square is a small fraction of the frame Otsu splits the
histogram between the white page and the wall behind it instead, putting the
square on the same side as the wall where it never becomes its own contour.
Relaxing `min_area_frac` or `min_squareness` does nothing about this: the
square is not being rejected by a filter, it is not being segmented at all.

Simulated at 87 deg FOV and 0.5 m, the crossover sits near 0.5% black pixels:

| square | black in frame | Otsu picks | outcome |
| --- | --- | --- | --- |
| 1 in | 0.10% | 227 | splits wall/page, square invisible |
| 2 in | 0.37% | 226 | splits wall/page, square invisible |
| 3 in | 0.79% | 8 | splits on the square |
| 4 in | 1.54% | 85 | splits on the square |

Bigger also buys precision: relative distance error equals relative size
error, so a 60 px marker measured to +/-0.5 px is +/-0.8%, where a 9 px one is
+/-5.6%. See `VISION_HOVER_PLAN.md` and HYP-MARKER-OTSU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PT_PER_IN = 72.0
PAGE_W = 8.5 * PT_PER_IN   # 612
PAGE_H = 11.0 * PT_PER_IN  # 792

# Everything except the square lives outside this band, so a camera pointed at
# the marker sees clean white around it and the detector has nothing to compete
# with. The detector picks the largest square-ish blob, so competition matters.
QUIET_ZONE_IN = 2.0


def esc(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_content(size_in: float) -> str:
    """Page drawing commands. Origin is bottom-left, units are 1/72 inch."""
    side = size_in * PT_PER_IN
    x0 = (PAGE_W - side) / 2.0
    y0 = (PAGE_H - side) / 2.0

    ops = ["0 0 0 rg"]

    # The marker itself. This rectangle is the only thing that has to be exact.
    ops.append(f"{x0:.4f} {y0:.4f} {side:.4f} {side:.4f} re f")

    def text(x: float, y: float, size: float, body: str) -> None:
        ops.append(f"BT /F1 {size:g} Tf {x:.2f} {y:.2f} Td ({esc(body)}) Tj ET")

    # Header, well clear of the quiet zone.
    text(72, 750, 14, "Vision-hover marker")
    text(72, 732, 10, f"{size_in:.3f} in  ({size_in * 25.4:.1f} mm) solid square")

    # Verification ruler along the bottom: 6 inches of inch ticks. A scaled
    # print shows up here immediately against any real ruler.
    ruler_y = 190.0
    ruler_x = 72.0
    ops.append(f"{ruler_x:.2f} {ruler_y:.2f} {6 * PT_PER_IN:.2f} 1.2 re f")
    for i in range(7):
        x = ruler_x + i * PT_PER_IN
        tick_h = 12.0 if i % 2 == 0 else 7.0
        ops.append(f"{x:.2f} {ruler_y:.2f} 1.2 {tick_h:.2f} re f")
        # Labels go below the baseline: above it they reach into the quiet
        # zone, which the self-test catches.
        text(x - 2, ruler_y - 11, 8, str(i))
    text(ruler_x, ruler_y - 25, 9,
         "Ruler check: these ticks must sit exactly 1 inch apart.")

    lines = [
        "BEFORE USE - print at 100% / Actual Size. Turn OFF Fit to Page and",
        "Shrink to Fit. Then measure the square with a ruler: it must be",
        f"exactly {size_in:.3f} in ({size_in * 25.4:.1f} mm) on each edge. A scaled print",
        "silently corrupts every distance the drone computes.",
        "",
        "Mount flat on a light-coloured wall, square is upright, no tilt.",
        "A page that contrasts strongly with the wall behind it can itself",
        "read as a large square blob and beat the marker - if the calibration",
        "wizard warns the box is huge, that is what happened. Trim the",
        "margins or use a lighter backing.",
    ]
    y = 145.0
    for line in lines:
        if line:
            text(72, y, 9, line)
        y -= 11.5

    text(72, 30, 8, "drone-rl-lab / real_flight/make_marker.py")
    return "\n".join(ops)


def build_pdf(size_in: float) -> bytes:
    content = build_content(size_in).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:g} {PAGE_H:g}] "
         f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode("ascii"),
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
        + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode("ascii")
    return bytes(out)


def _self_test() -> int:
    """Check the geometry is what the calibration depends on."""
    ok = True

    for size_in in (1.0, 2.0):
        side = size_in * PT_PER_IN
        content = build_content(size_in)
        expect = (f"{(PAGE_W - side) / 2.0:.4f} {(PAGE_H - side) / 2.0:.4f} "
                  f"{side:.4f} {side:.4f} re f")
        hit = expect in content
        print(f"{'PASS' if hit else 'FAIL'} [{size_in} in square is exact]: {expect}")
        ok &= hit

    # Nothing but the square may sit inside the quiet zone, or the detector has
    # something to latch onto besides the marker.
    side = PT_PER_IN
    lo = (PAGE_H - side) / 2.0 - QUIET_ZONE_IN * PT_PER_IN
    hi = (PAGE_H + side) / 2.0 + QUIET_ZONE_IN * PT_PER_IN
    intruders = []
    for line in build_content(1.0).splitlines():
        parts = line.split()
        y = None
        if line.endswith("re f") and len(parts) == 6:
            y = float(parts[1])
        elif line.startswith("BT "):
            y = float(parts[5])
        if y is not None and lo < y < hi and "360.0000 72.0000 72.0000 re f" not in line:
            intruders.append(line)
    print(f"{'PASS' if not intruders else 'FAIL'} [quiet zone clear]: "
          f"{len(intruders)} intruders in y {lo:.0f}-{hi:.0f}")
    ok &= not intruders

    pdf = build_pdf(1.0)
    checks = [
        (pdf.startswith(b"%PDF-1.4"), "header"),
        (pdf.rstrip().endswith(b"%%EOF"), "EOF"),
        (b"/MediaBox [0 0 612 792]" in pdf, "US Letter media box"),
    ]
    for good, label in checks:
        print(f"{'PASS' if good else 'FAIL'} [{label}]")
        ok &= good

    # The xref offsets must actually point at their objects, or readers choke.
    start = int(pdf.rsplit(b"startxref", 1)[1].split()[0])
    table = pdf[start:].split(b"\n")
    good_xref = True
    for i, row in enumerate(table[3:8]):
        off = int(row.split()[0])
        if not pdf[off:].startswith(f"{i + 1} 0 obj".encode("ascii")):
            good_xref = False
    print(f"{'PASS' if good_xref else 'FAIL'} [xref offsets resolve]")
    ok &= good_xref

    print(f"\nSELF-TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-in", type=float, default=1.0,
                    help="square edge length in inches (default 1.0)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.size_in <= 0 or args.size_in > 7.0:
        print("size must be between 0 and 7 inches to leave any margin")
        return 1

    label = f"{args.size_in:g}".replace(".", "p")
    out = args.out or Path(__file__).resolve().parent / f"marker_{label}in.pdf"
    out.write_bytes(build_pdf(args.size_in))
    print(f"wrote {out}  ({args.size_in:.3f} in square on 8.5x11)")
    print("Print at 100% / Actual Size. Verify with a ruler before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
