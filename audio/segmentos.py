"""
Reacción del público por COMEDIANTE dentro de un mismo show de Alta Comedia.

Solo en los shows donde se sabe quién tiene el micrófono:
- Hoy Nace un Chiste (17 jul 2026): cuatro sets seguidos. Cortes sacados de la transcripción (el
  presentador anuncia a cada uno) y verificados con fotogramas el 13 nov 2026: Wilmar en el minuto 8,
  Andrés en el 25, Daniela en el 42, Alex en el 55.
- Comedia de Alto Riesgo (15 oct 2026): turnos de cada juego de Analisis_videos/.../juegos.json
  (nombres dados por Andrés, tiempos verificados con fotogramas). Los turnos de dos o tres personas
  quedan como pareja o trío, no se reparten.

Por qué esto sí es comparable y el nivel entre especiales no: todos los comediantes de un mismo show
comparten micrófono, sala y público. Con el umbral FIJO, la diferencia entre ellos no es de grabación.
Lo que sí cambia es el momento: el público de las 9 no es el de las 10. La columna "orden" lo deja a la vista.

Además, los shows completos de una sola persona (Ansioso y el set de La Korte, de Andrés), para ver a
cada integrante en todos sus formatos.

Uso:  python audio/segmentos.py   -> data/data_frame/df_audio_comediantes.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metricas as M  # noqa: E402

VIDEOS = M.RAIZ.parent / "Analisis_videos" / "output"
ALIAS = {"Dani Villa": "Daniela Villa", "Alex Marín": "Alex Marín"}

# (id del catálogo, comediante, inicio s, fin s). Fin del último set: 71,5 min, cuando suben todos a despedir.
HNUC = [("alta-comedia-hoy-nace-un-chiste-2026-07-17", "Wilmar Tovar", 0, 900),
        ("alta-comedia-hoy-nace-un-chiste-2026-07-17", "Andrés Loaiza", 900, 2244),
        ("alta-comedia-hoy-nace-un-chiste-2026-07-17", "Daniela Villa", 2244, 2718),
        ("alta-comedia-hoy-nace-un-chiste-2026-07-17", "Alex Marín", 2718, 4290)]


# Shows completos de una sola persona: todo el show es suyo. El índice del show no aplica (sería 1 por
# definición); sirven para ver a esa persona en un formato largo y su ritmo, no para compararla con compañeros.
SOLOS = [("alta-comedia-ansioso-2025-08-22", "Andrés Loaiza", "Ansioso (ago 2025)"),
         ("andres-loaiza-0731", "Andrés Loaiza", "Set en La Korte (31 jul 2025)")]
NOMBRES = {"alta-comedia-hoy-nace-un-chiste-2026-07-17": "Hoy Nace un Chiste (17 jul 2026)",
           "alta-comedia-comedia-de-alto-riesgo-2026-10-15": "Comedia de Alto Riesgo (15 oct 2026)",
           **{id_: nombre for id_, _, nombre in SOLOS}}
TIPO = {"alta-comedia-hoy-nace-un-chiste-2026-07-17": "set en show de varios",
        "alta-comedia-comedia-de-alto-riesgo-2026-10-15": "turnos de impro"}


def tramos_solos() -> list[tuple]:
    out = []
    for id_, quien, _ in SOLOS:
        c = dict(np.load(M.FRAMES / f"{id_}.npz"))
        out.append((id_, quien, *M.ventana(c["t"], c["habla"])))
    return out


def tramos_alto_riesgo() -> list[tuple]:
    j = json.loads((VIDEOS / "comedia_de_alto_riesgo" / "2026-10-15" / "juegos.json").read_text(encoding="utf-8"))
    id_ = "alta-comedia-comedia-de-alto-riesgo-2026-10-15"
    out = [(id_, "Andrés Loaiza (presentador)", *j["fuera_de_juegos"]["apertura"])]
    for juego in j["juegos"]:
        turnos = juego.get("turnos", [])
        for i, t in enumerate(turnos):
            fin = turnos[i + 1]["inicio"] if i + 1 < len(turnos) else juego["fin"]
            quien = " y ".join(ALIAS.get(x.strip(), x.strip()) for x in t["quien"].replace(",", " y").split(" y ") if x.strip())
            out.append((id_, quien, t["inicio"], fin))
    return out


def medir(c: dict, ini: float, fin: float) -> dict:
    m = (c["t"] >= ini) & (c["t"] < fin)
    t, p, ap = c["t"][m], c["risa"][m] + c["aplauso"][m], c["aplauso"][m]
    paso = float(np.median(np.diff(c["t"])))
    ev = M.eventos(t, p, M.UMBRAL, paso)
    fuertes = [e for e in M.eventos(t, ap, 0.5, paso) if e[1] - e[0] >= 3]
    return {"segundos": float(fin - ini), "reacciones": len(ev), "masa": float(p.sum()),
            "cuadros_reaccion": int((p >= M.UMBRAL).sum()), "cuadros": int(m.sum()),
            "aplausos": len(fuertes), "intervalos": list(np.diff([a for a, _ in ev]))}


def main():
    tramos = HNUC + tramos_alto_riesgo() + tramos_solos()
    frames = {}
    filas = []
    for orden, (id_, quien, ini, fin) in enumerate(tramos):
        c = frames.setdefault(id_, dict(np.load(M.FRAMES / f"{id_}.npz")))
        filas.append({"id": id_, "comediante": quien, "inicio_min": round(ini / 60, 1), **medir(c, ini, fin)})
    d = pd.DataFrame(filas)
    g = d.groupby(["id", "comediante"], sort=False).agg(
        tramos=("segundos", "size"), primer_minuto=("inicio_min", "min"), segundos=("segundos", "sum"),
        reacciones=("reacciones", "sum"), masa=("masa", "sum"), cuadros_reaccion=("cuadros_reaccion", "sum"),
        cuadros=("cuadros", "sum"), aplausos=("aplausos", "sum"),
        intervalos=("intervalos", lambda s: [x for lista in s for x in lista])).reset_index()
    mins = g["segundos"] / 60
    g["minutos"] = mins.round(1)
    g["reacciones_min"] = (g["reacciones"] / mins).round(2)
    g["masa_reaccion_min"] = (g["masa"] / mins).round(2)
    g["pct_tiempo_reaccion"] = (100 * g["cuadros_reaccion"] / g["cuadros"]).round(1)
    g["aplausos_hora"] = (60 * g["aplausos"] / mins).round(1)
    g["segundos_entre_reacciones"] = g["intervalos"].map(lambda x: round(float(np.median(x)), 1) if x else np.nan)
    # índice dentro del show: 1 = el promedio de ese show. Es la comparación justa entre compañeros.
    g["indice_show"] = (g["masa_reaccion_min"] / g.groupby("id")["masa"].transform("sum")
                        * g.groupby("id")["segundos"].transform("sum") / 60).round(2)
    g["show"] = g["id"].map(NOMBRES)
    solo_show = g["id"].isin([x[0] for x in SOLOS])
    g["tipo"] = g["id"].map(TIPO).fillna("show completo, solo")
    g.loc[g["comediante"].str.contains("presentador"), "tipo"] = "presentador"
    g.loc[solo_show, "indice_show"] = np.nan   # todo el show es suyo: no hay con quién compararlo
    g["solo"] = ~g["comediante"].str.contains(" y ")
    cols = ["show", "comediante", "tipo", "solo", "tramos", "primer_minuto", "minutos", "reacciones_min", "masa_reaccion_min",
            "pct_tiempo_reaccion", "segundos_entre_reacciones", "aplausos_hora", "indice_show"]
    g[cols].to_parquet(M.DF / "df_audio_comediantes.parquet", index=False)
    pd.set_option("display.width", 220)
    print(g[cols].to_string(index=False))


if __name__ == "__main__":
    main()
