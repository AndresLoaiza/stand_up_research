"""
Fails (exit 1) if the PDF shows a number that is not in report_data.json.

Every number token in the PDF text (digits with an optional decimal comma or point) must appear as a
number token somewhere in the JSON. Numbers inside quotes, titles and bios are in the JSON too, so they
pass; a number typed by hand in the template or invented by the renderer does not.

    python report/check_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
NUM = re.compile(r"\d+(?:[.,]\d+)*")


def tokens(text: str) -> set[str]:
    return set(NUM.findall(text))


def main() -> int:
    raw = (HERE / "report_data.json").read_text(encoding="utf-8")
    allowed = tokens(raw) | tokens(json.dumps(json.loads(raw), ensure_ascii=False))
    bad = {}
    with pdfplumber.open(HERE / "informe_comediantes.pdf") as pdf:
        pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            for t in tokens(page.extract_text() or ""):
                if t not in allowed:
                    bad.setdefault(t, []).append(i)
    if bad:
        print(f"FALLA: {len(bad)} números del PDF no están en report_data.json")
        for t, p in sorted(bad.items()):
            print(f"  {t!r} en páginas {p[:10]}")
        return 1
    print(f"OK: todos los números de las {pages} páginas están en report_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
