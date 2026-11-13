"""
¿Cuánto cuesta un tramo seco? La idea de TIC-TALK ("una sala caliente se mantiene caliente")
medida en nuestros shows: si el público pasa un rato sin reaccionar, ¿cuánto tarda en volver?

Por show, en ventanas de 15 s, la reacción (risa + aplauso, sin umbral) dividida por el promedio
del propio show: 1 = su ritmo normal. Eso hace comparables shows con grabaciones distintas.

Mide tres cosas:
- memoria: cuánto se parece la reacción de ahora a la de hace 15 s, 30 s... 3 min
- después de un minuto flojo o bueno: cuánta reacción hay en los dos minutos siguientes
- tramos secos (45 s o más por debajo del 25 % del ritmo normal): cuánto tarda en volver al 80 %

Límite: esto describe, no prueba causa. Un tramo seco puede ser material flojo que sigue siendo
flojo después; el público frío y el material flojo se ven igual en el audio.

Uso:  python audio/sala.py   -> data/data_frame/df_audio_sala.parquet y df_audio_secos.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metricas as M  # noqa: E402

VENTANA_S = 15
SECO = 0.25          # por debajo del 25 % del ritmo normal del show
SECO_MIN_S = 45      # al menos tres ventanas seguidas
VUELTA = 0.8         # "volvió" = el promedio del minuto siguiente llega al 80 % del ritmo normal
DESPUES_S = 120


def serie(c: dict) -> np.ndarray:
    """Reacción por ventana de 15 s, relativa al promedio del show."""
    t0, t1 = M.ventana(c["t"], c["habla"])
    m = (c["t"] >= t0) & (c["t"] <= t1)
    p = c["risa"][m] + c["aplauso"][m]
    bordes = np.arange(t0, t1 + VENTANA_S, VENTANA_S)
    masa = np.histogram(c["t"][m], bins=bordes, weights=p)[0]
    return masa / masa.mean() if masa.mean() else masa


def tramos_secos(s: np.ndarray) -> list[tuple[int, int]]:
    """(inicio, fin exclusivo) en índices de ventana, de las rachas por debajo de SECO."""
    out, ini = [], None
    for i, v in enumerate(np.append(s, np.inf)):
        if v < SECO and ini is None:
            ini = i
        elif v >= SECO and ini is not None:
            if (i - ini) * VENTANA_S >= SECO_MIN_S:
                out.append((ini, i))
            ini = None
    return out


def recuperacion_s(s: np.ndarray, fin: int) -> float | None:
    """Segundos desde el final del tramo seco hasta que el promedio móvil de 1 min llega a VUELTA."""
    k = 60 // VENTANA_S
    for j in range(fin, len(s) - k + 1):
        if s[j:j + k].mean() >= VUELTA:
            return float((j - fin) * VENTANA_S)
    return None


def analizar(id_: str, c: dict) -> tuple[dict, list[dict]]:
    s = serie(c)
    k = DESPUES_S // VENTANA_S
    # memoria: correlación de la serie consigo misma corrida 1..12 ventanas (15 s .. 3 min)
    mem = {f"memoria_{l * VENTANA_S}s": float(np.corrcoef(s[:-l], s[l:])[0, 1]) for l in (1, 2, 4, 8, 12)}
    # después de un minuto flojo / bueno: reacción media de los 2 minutos siguientes
    prev = np.array([s[i - 4:i].mean() for i in range(4, len(s) - k)])
    sig = np.array([s[i:i + k].mean() for i in range(4, len(s) - k)])
    flojo, bueno = prev < np.percentile(prev, 20), prev > np.percentile(prev, 80)
    secos = [{"id": id_, "inicio_min": round(a * VENTANA_S / 60, 1), "duracion_s": (b - a) * VENTANA_S,
              "recupera_s": recuperacion_s(s, b)} for a, b in tramos_secos(s)]
    return {"id": id_, **mem,
            "despues_de_minuto_flojo": float(sig[flojo].mean()), "despues_de_minuto_bueno": float(sig[bueno].mean()),
            "tramos_secos_hora": 60 * len(secos) / (len(s) * VENTANA_S / 60),
            "recupera_mediana_s": float(np.nanmedian([x["recupera_s"] if x["recupera_s"] is not None else np.nan
                                                      for x in secos])) if secos else np.nan}, secos


def main():
    cat = pd.read_csv(M.CATALOGO, dtype=str)
    filas, secos = [], []
    for _, r in cat.iterrows():
        p = M.FRAMES / f"{r['id']}.npz"
        if not p.exists():
            continue
        fila, s = analizar(r["id"], dict(np.load(p)))
        filas.append({**fila, "comediante": r["comediante"], "especial": r["especial"], "grupo": r["grupo"]})
        secos += [{**x, "comediante": r["comediante"], "especial": r["especial"], "grupo": r["grupo"]} for x in s]
    d, sd = pd.DataFrame(filas), pd.DataFrame(secos)
    d.to_parquet(M.DF / "df_audio_sala.parquet", index=False)
    sd.to_parquet(M.DF / "df_audio_secos.parquet", index=False)

    pd.set_option("display.width", 220)
    cols = [c for c in d.columns if c.startswith("memoria") or c.startswith("despues") or c in ("tramos_secos_hora", "recupera_mediana_s")]
    print("Mediana por grupo:")
    print(d.groupby("grupo")[cols].median().round(2).T)
    print(f"\nTramos secos: {len(sd)} en total. Segundos hasta volver al 80 %, por duración del tramo seco:")
    sd["dur"] = pd.cut(sd["duracion_s"], [0, 59, 89, 10_000], labels=["45-59 s", "60-89 s", "90 s o más"])
    print(sd.groupby(["grupo", "dur"], observed=True)["recupera_s"].describe()[["count", "25%", "50%", "75%"]].round(0))
    print("\nNunca volvieron (hasta el final del show):", int(sd["recupera_s"].isna().sum()))
    print("\nShows propios:")
    print(d[d["grupo"] == "propio"][["comediante", "especial", "memoria_60s", "despues_de_minuto_flojo",
                                     "despues_de_minuto_bueno", "tramos_secos_hora", "recupera_mediana_s"]].round(2).to_string(index=False))
    print("\nTramos secos de los shows propios:")
    print(sd[sd["grupo"] == "propio"][["especial", "inicio_min", "duracion_s", "recupera_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
