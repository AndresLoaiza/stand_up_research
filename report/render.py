"""
report_data.json + template.html.j2 -> informe.html -> informe_comediantes.pdf (headless Chrome/Edge).

    python report/render.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).resolve().parent
BROWSERS = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")]


def main():
    data = json.loads((HERE / "report_data.json").read_text(encoding="utf-8"))
    for c in data["comedians"]:
        if c["photo"]:
            c["photo"]["src"] = f"img/{c['photo']['file']}"
    for c in data["members"]:
        if c["photo"]:
            c["photo"]["src"] = Path(c["photo"]["file"]).as_uri()
    env = Environment(loader=FileSystemLoader(HERE), undefined=StrictUndefined, autoescape=True)
    html = env.get_template("template.html.j2").render(
        m=data["meta"], mapa=data["mapa"], comedians=data["comedians"], members=data["members"],
        annex=data["annex"], v=data.get("validacion"))
    out_html = HERE / "informe.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = HERE / "informe_comediantes.pdf"
    browser = next(b for b in BROWSERS if b.exists())
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf}", out_html.as_uri()],
                   check=True, capture_output=True, timeout=300)
    print(pdf)


if __name__ == "__main__":
    main()
