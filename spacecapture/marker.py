"""Printable ArUco scale marker as an A4 PDF (exact physical size when printed at 100%)."""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2

MM = 72.0 / 25.4  # PDF points per millimeter
A4 = (210.0, 297.0)


def _pdf(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _stream(header: str, data: bytes) -> bytes:
    return f"<< {header} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def make_marker_pdf(out: Path, size_mm: float = 150.0, marker_id: int = 0, dictionary: str = "DICT_4X4_50") -> Path:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    img = cv2.aruco.generateImageMarker(d, marker_id, 600, borderBits=1)  # gray, 600x600
    page_w, page_h = A4[0] * MM, A4[1] * MM
    s = size_mm * MM
    x, y = (page_w - s) / 2, (page_h - s) / 2 + 20 * MM

    ruler_y = y - 25 * MM
    rx = (page_w - 100 * MM) / 2
    ops = [
        "q", f"{s:.3f} 0 0 {s:.3f} {x:.3f} {y:.3f} cm", "/Im0 Do", "Q",
        "BT /F1 16 Tf", f"{20 * MM:.2f} {page_h - 25 * MM:.2f} Td",
        f"(spaceCapture scale marker  -  {dictionary}  id {marker_id}) Tj", "ET",
        "BT /F1 11 Tf", f"{20 * MM:.2f} {page_h - 33 * MM:.2f} Td",
        f"(Black square = {size_mm:g} mm. Print at 100% / Actual size - NOT 'fit to page'.) Tj", "ET",
        "BT /F1 11 Tf", f"{20 * MM:.2f} {page_h - 39 * MM:.2f} Td",
        "(Check with a ruler: the line below must measure exactly 100 mm.) Tj", "ET",
        "0.6 w", f"{rx:.3f} {ruler_y:.3f} m {rx + 100 * MM:.3f} {ruler_y:.3f} l S",
    ]
    for i in range(11):
        tx = rx + i * 10 * MM
        tick = 4 * MM if i % 5 == 0 else 2.5 * MM
        ops.append(f"{tx:.3f} {ruler_y:.3f} m {tx:.3f} {ruler_y + tick:.3f} l S")
    for i, label in ((0, "0"), (5, "5 cm"), (10, "10 cm")):
        ops += ["BT /F1 9 Tf", f"{rx + i * 10 * MM - 2:.2f} {ruler_y - 4 * MM:.2f} Td", f"({label}) Tj", "ET"]
    ops += ["BT /F1 10 Tf", f"{20 * MM:.2f} {25 * MM:.2f} Td",
            "(Tape it flat on a wall at chest height and film it for 2-3 seconds from a few angles.) Tj", "ET"]
    content = "\n".join(ops).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
         "/Resources << /XObject << /Im0 5 0 R >> /Font << /F1 6 0 R >> >> /Contents 4 0 R >>").encode(),
        _stream("", content),
        _stream(f"/Type /XObject /Subtype /Image /Width {img.shape[1]} /Height {img.shape[0]} "
                "/ColorSpace /DeviceGray /BitsPerComponent 8 /Interpolate false /Filter /FlateDecode",
                zlib.compress(img.tobytes(), 9)),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_pdf(objects))
    return out
