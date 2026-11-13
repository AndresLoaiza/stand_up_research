"""
¿Cuántas risas reales detecta cada umbral? Se compara contra los subtítulos para sordos (SDH),
que marcan "(audience laughing)" con la hora exacta.

Los SDH no marcan todas las risas (el subtitulador anota las notorias), así que sirven para
medir cuántas de esas risas notorias se detectan (sensibilidad), no cuántas detecciones son
falsas. Para eso se mira la otra cara: cuántas detecciones por minuto da cada umbral.

Uso:  python audio/validar.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metricas as M  # noqa: E402

FUENTE = Path(r"D:\ANDRES\Stand up")
SDH = {
    "eddie-izzard-live-from-wembley": "Eddie Izzard/Eddie Izzard Live From Wembley (2009) [1080p] [WEBRip] [YTS.MX]/Subs/SDH.eng.srt",
    "eddie-izzard-sexie": "Eddie Izzard/Eddie Izzard Sexie (2003) [1080p] [WEBRip] [YTS.MX]/Subs/SDH.eng.srt",
    "eddie-izzard-stripped": "Eddie Izzard/Eddie Izzard Stripped (2009) [1080p] [WEBRip] [YTS.MX]/Eddie.Izzard.Stripped.2009.1080p.WEBRip.x264.AAC-[YTS.MX].srt",
    "louis-c-k-dolby": "Louis C.K/2023 Louis.C.K.at.the.Dolby.2023.1080p.WEBRip.x265-LAMA[TGx]/Subs/2_English.srt",
    "patrice-o-neal-killing-is-easy": "Patrice ONeal/Patrice.ONeal.Killing.Is.Easy.2021.720p.WEBRip.x264.AAC-[YTS.MX].srt",
    "sebastian-maniscalco-embarrassed": "Sebastian Maniscalco/Sebastian Maniscalco Arent You Embarrassed (2014) [720p] [WEBRip] [YTS.MX]/Sebastian.Maniscalco.Arent.You.Embarrassed.2014.720p.WEBRip.x264.AAC-[YTS.MX].srt",
}
MARCA = re.compile(r"laugh|applau|cheer", re.I)
HORA = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
TOL = (-2.0, 5.0)   # la marca puede ir un poco antes o hasta 5 s después del arranque de la risa


def marcas(srt: Path) -> np.ndarray:
    texto = srt.read_text(encoding="utf-8", errors="replace")
    out = []
    for bloque in re.split(r"\n\s*\n", texto):
        h = HORA.search(bloque)
        if h and MARCA.search(bloque.split("-->", 1)[1]):
            hh, mm, ss, ms = (int(x) for x in h.groups())
            out.append(hh * 3600 + mm * 60 + ss + ms / 1000)
    return np.array(out)


def main():
    filas = []
    for id_, rel in SDH.items():
        npz = M.FRAMES / f"{id_}.npz"
        if not npz.exists():
            print(f"  {id_}: falta extraer")
            continue
        c = dict(np.load(npz))
        t0, t1 = M.ventana(c["t"], c["habla"])
        mk = marcas(FUENTE / rel)
        mk = mk[(mk >= t0) & (mk <= t1)]
        mins = (t1 - t0) / 60
        paso = float(np.median(np.diff(c["t"])))
        m = (c["t"] >= t0) & (c["t"] <= t1)
        for senal in ("risa", "risa+aplauso"):
            p = c["risa"][m] + (c["aplauso"][m] if senal != "risa" else 0)
            # umbrales fijos y relativos al show (percentil de su propia señal, como usa metricas.py)
            umbrales = [(str(u), u) for u in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25)]
            umbrales += [(f"p{q}", max(M.UMBRAL_PISO, float(np.percentile(p, q)))) for q in (60, 70, 80)]
            for nombre, u in umbrales:
                ev = M.eventos(c["t"][m], p, u, paso)
                ini = np.array([a for a, _ in ev]); fin = np.array([b for _, b in ev])
                hit = [bool(((ini + TOL[0] <= x) & (x <= fin + TOL[1])).any()) for x in mk] if ev else [False] * len(mk)
                filas.append({"id": id_, "senal": senal, "umbral": nombre, "marcas_sdh_min": round(len(mk) / mins, 2),
                              "detectadas_min": round(len(ev) / mins, 2),
                              "sensibilidad": round(float(np.mean(hit)), 2) if len(mk) else np.nan})
    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["senal", "umbral"], values=["sensibilidad", "detectadas_min", "marcas_sdh_min"],
                         aggfunc="median").round(2))
    print()
    print(df[df["umbral"].isin(["0.05", "p70"]) & (df["senal"] == "risa+aplauso")].to_string(index=False))
    df.to_parquet(M.DF / "df_audio_validacion.parquet", index=False)


if __name__ == "__main__":
    main()
