#!/usr/bin/env python3
"""Generate an A4 AprilTag tag36h11 ID 0 sheet at exact print scale."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


# Official AprilRobotics tag36h11 ID 0, including one white quiet-zone
# module on every side. 1 denotes a black module.
TAG = (
    "0000000000",
    "0111111110",
    "0100101010",
    "0110001010",
    "0110011110",
    "0101011110",
    "0110100110",
    "0111101110",
    "0111111110",
    "0000000000",
)

OUTPUT = Path(__file__).with_name("tag36h11_id0_120mm_A4.pdf")


def main() -> None:
    page_w, page_h = A4
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=0)
    pdf.setTitle("AprilTag tag36h11 ID 0 - 120 mm")

    module = 15 * mm
    total = 10 * module
    origin_x = (page_w - total) / 2
    origin_y = (page_h - total) / 2

    pdf.setFillColorRGB(0, 0, 0)
    for row, bits in enumerate(TAG):
        for col, bit in enumerate(bits):
            if bit == "1":
                x = origin_x + col * module
                y = origin_y + (9 - row) * module
                pdf.rect(x, y, module, module, stroke=0, fill=1)

    # 160 x 160 mm square cutting line. The black tag boundary is 120 mm,
    # leaving a 20 mm white margin on every side after cutting.
    cut_size = 160 * mm
    cut_x = (page_w - cut_size) / 2
    cut_y = (page_h - cut_size) / 2
    pdf.saveState()
    pdf.setStrokeColorRGB(0.65, 0.65, 0.65)
    pdf.setLineWidth(0.25)
    pdf.setDash(2 * mm, 2 * mm)
    pdf.rect(cut_x, cut_y, cut_size, cut_size, stroke=1, fill=0)
    pdf.restoreState()

    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
