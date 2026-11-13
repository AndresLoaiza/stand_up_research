"""
De los cuadros de YAMNet (extraer_risas.py) a métricas por especial.

Escribe en data/data_frame/:
  df_audio.parquet        una fila por especial: ritmo de risa, aplausos, silencios, arco...
  df_audio_arco.parquet   risas por minuto en cada décimo del show (para la curva promedio)
  df_audio_sensibilidad.parquet  cuánto cambia el ranking si se mueve el umbral

Corre con el Python normal del proyecto (pandas):  python audio/metricas.py

Decisiones de medición (y por qué):
- Se mide la REACCIÓN del público (probabilidad de risa + aplauso), no solo la risa: en mezclas de TV
  la voz del comediante domina y YAMNet le da a la risa del público probabilidades bajas, o la llama
  "aplauso". El umbral de Analisis_videos/02b (0,25, solo risa) encontró el 4 % de las risas marcadas
  en subtítulos para sordos (validar.py, 6 especiales, 13 nov 2026).
- DOS umbrales, para dos preguntas distintas:
  * Ritmo y forma (cuándo reacciona el público): umbral RELATIVO a cada show, el percentil 70 de su
    propia señal. Detecta el 76 % de las risas marcadas (mediana) contra el 63 % de un umbral fijo,
    porque se adapta a cuánto se oye el público en esa grabación.
  * Nivel (cuánta reacción hay): umbral FIJO de 0,05 (reacciones_min_fijo) y masa sin umbral. Estos SÍ
    dependen de la grabación: la detección fue del 14 % al 87 % según el especial. Diferencias de
    nivel entre dos especiales no se pueden atribuir al comediante sin mirar senal_p90.
- senal_p90: qué tan fuerte llega a sonar la reacción en esa grabación. Bajo = público poco audible.
- Ventana de contenido: se recorta la intro y los créditos (música, aplauso de entrada) usando
  el habla. Si no, un especial con 5 minutos de créditos parece peor.
- "masa de risa" (suma de probabilidad por minuto) no depende de ningún umbral: es la medida
  de control. Si una conclusión solo aparece con el conteo de eventos, es frágil.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
FRAMES = RAIZ / "data" / "audio" / "frames"
CATALOGO = RAIZ / "data" / "audio" / "catalogo.csv"
DF = RAIZ / "data" / "data_frame"

UMBRAL = 0.05          # nivel: umbral fijo de risa + aplauso (ver validar.py)
PERCENTIL = 70         # ritmo y forma: umbral relativo, percentil de la señal del propio show
UMBRAL_PISO = 0.02     # el relativo nunca baja de esto (un show casi sin público no inventa risas)
UMBRAL_HABLA = 0.5
UNIR_S = 1.0           # dos risas separadas por menos de esto son una sola
MIN_RISA_S = 1.0
RISA_GRANDE_S = 4.0    # una risa de 4 s o más: el público se quedó riendo
RACHA_GAP_S = 8.0      # risas a menos de 8 s una de otra forman racha
RACHA_MIN = 3
SECO_S = 60.0          # un minuto entero hablando sin una risa
BORDE_S = 120          # aplausos en los primeros/últimos 2 min son entrada y salida, no "aplauso de chiste"

# Especiales que también están en TextMining (catálogo de audio -> imdbID). Verificados a mano por
# comediante, título y año el 13 nov 2026.
IMDB = {
    "bill-burr-let-it-go": "1717578", "bill-hicks-relentless": "0148668", "bill-hicks-revelations": "0152183",
    "chris-rock-bring-the-pain": "0200529", "chris-rock-bigger-and-blacker": "0206636",
    "chris-rock-never-scared": "0405832", "eddie-izzard-definite-article": "0116066",
    "eddie-izzard-glorious": "0195753", "eddie-izzard-dress-to-kill": "0184424",
    "eddie-murphy-delirious": "0085474", "eddie-murphy-raw": "0092948",
    "george-carlin-carlin-on-campus": "0248828", "george-carlin-what-am-i-doing-in-new-jersey": "0246644",
    "george-carlin-doin-it-again": "0246642", "george-carlin-back-in-town": "0246641",
    "george-carlin-you-are-all-diseased": "0246645", "george-carlin-life-is-worth-losing": "0484855",
    "jim-jefferies-i-swear-to-god": "1439235", "john-mulaney-new-in-town": "2246779",
    "louis-c-k-shameless": "0927619", "louis-c-k-chewed-up": "1190722", "louis-c-k-oh-my-god": "2510998",
    "louis-c-k-comedy-store": "4368814", "louis-c-k-sincerely": "12087624", "louis-c-k-sorry": "16478030",
    "louis-c-k-dolby": "27430909", "mike-birbiglia-my-girlfriends-boyfriend": "2937390",
    "russell-peters-outsourced": "0823187",
}


def eventos(t: np.ndarray, p: np.ndarray, umbral: float, paso: float) -> list[tuple[float, float]]:
    """Tramos (inicio, fin) donde p >= umbral, unidos si están a menos de UNIR_S y de al menos MIN_RISA_S."""
    out, cur = [], None
    for ti, pi in zip(t, p):
        if pi < umbral:
            continue
        if cur and ti - cur[1] <= UNIR_S:
            cur[1] = ti + paso
        else:
            if cur:
                out.append(tuple(cur))
            cur = [ti, ti + paso]
    if cur:
        out.append(tuple(cur))
    return [(a, b) for a, b in out if b - a >= MIN_RISA_S]


def ventana(t: np.ndarray, habla: np.ndarray) -> tuple[float, float]:
    """Del primer al último momento con habla sostenida (media de 30 s > 0,4): fuera intro y créditos."""
    k = 30
    media = np.convolve(habla, np.ones(k) / k, mode="same")
    activos = np.flatnonzero(media > 0.4)
    if len(activos) == 0:
        return float(t[0]), float(t[-1])
    return float(t[activos[0]]), float(t[activos[-1]])


def metricas(c: dict, umbral: float | None = None) -> tuple[dict, list[float]]:
    """`umbral` None = relativo al show (percentil PERCENTIL); un número = fijo (para la prueba de robustez)."""
    t = c["t"]
    paso = float(np.median(np.diff(t))) if len(t) > 1 else 0.975
    t0, t1 = ventana(t, c["habla"])
    m = (t >= t0) & (t <= t1)
    t, aplauso, habla, db = t[m], c["aplauso"][m], c["habla"][m], c["db"][m]
    risa = c["risa"][m] + aplauso   # reacción del público
    mins = (t1 - t0) / 60
    fijo = len(eventos(t, risa, UMBRAL, paso))
    if umbral is None:
        umbral = max(UMBRAL_PISO, float(np.percentile(risa, PERCENTIL)))
    ev = eventos(t, risa, umbral, paso)
    inicios = np.array([a for a, _ in ev])
    dur = np.array([b - a for a, b in ev]) if ev else np.array([0.0])
    huecos = np.diff(np.concatenate([[t0], inicios, [t1]])) if ev else np.array([t1 - t0])

    # rachas: secuencias de RACHA_MIN o más risas seguidas, cada una a menos de RACHA_GAP_S de la anterior
    rachas, seguidas = 0, 1
    for g in np.diff(inicios) if len(inicios) > 1 else []:
        if g <= RACHA_GAP_S:
            seguidas += 1
        else:
            rachas += seguidas >= RACHA_MIN
            seguidas = 1
    rachas += seguidas >= RACHA_MIN and len(inicios) >= RACHA_MIN

    # aplausos a mitad de show (no los de entrada y salida)
    medio = (t >= t0 + BORDE_S) & (t <= t1 - BORDE_S)
    # aplauso fuerte y sostenido: el público aplaude un remate, no solo se ríe
    aplausos = [e for e in eventos(t[medio], aplauso[medio], 0.5, paso) if e[1] - e[0] >= 3]

    # Arco con la MASA de reacción (sin umbral) de cada décimo: la mezcla de sonido mueve el nivel de
    # todo el show, no su forma; dividido por el promedio del propio show, la forma se puede comparar.
    deciles = np.linspace(t0, t1, 11)
    arco = [float(risa[(t >= a) & (t < b)].sum() / ((b - a) / 60)) for a, b in zip(deciles[:-1], deciles[1:])]
    masa_media = float(np.mean(arco)) or np.nan
    por_min = np.histogram(inicios, bins=np.arange(t0, t1 + 60, 60))[0] if ev else np.zeros(1)
    masa_min = np.histogram(t, bins=np.arange(t0, t1 + 60, 60), weights=risa)[0]
    lpm = len(ev) / mins if mins else 0.0
    # Medidas RELATIVAS al propio show (resisten la mezcla de sonido):
    # - concentración: qué parte de toda la reacción cae en el 20 % de minutos más fuertes
    # - posición de los picos: dónde caen los 10 minutos más fuertes (0 = inicio, 1 = final)
    # - tramos secos relativos: silencios de más de 3 veces el intervalo típico de ese mismo show
    top = np.sort(masa_min)[::-1]
    n20 = max(1, int(round(0.2 * len(top))))
    picos = np.argsort(masa_min)[::-1][:10]
    g_med = float(np.median(np.diff(inicios))) if len(inicios) > 1 else np.nan
    secos_rel = int((huecos > 3 * g_med).sum()) if not np.isnan(g_med) else 0
    es_risa = risa >= umbral
    # cuánto suena la risa por encima del habla del mismo show: independiente del volumen de grabación
    db_habla = np.median(db[(habla >= UMBRAL_HABLA) & ~es_risa]) if ((habla >= UMBRAL_HABLA) & ~es_risa).any() else np.nan
    db_risa = np.median(db[es_risa]) if es_risa.any() else np.nan

    return {
        "minutos": round(mins, 1),
        "reacciones_min": round(lpm, 2),
        "reacciones_min_fijo": round(fijo / mins, 2) if mins else 0.0,
        "senal_p90": round(float(np.percentile(risa, 90)), 3),
        "umbral_usado": round(umbral, 3),
        "masa_reaccion_min": round(float(risa.sum() / mins), 2) if mins else 0.0,
        "pct_tiempo_reaccion": round(100 * float(es_risa.mean()), 1),
        "pct_tiempo_habla": round(100 * float((habla >= UMBRAL_HABLA).mean()), 1),
        "reaccion_dur_mediana_s": round(float(np.median(dur)), 1),
        "reacciones_grandes_hora": round(60 * float((dur >= RISA_GRANDE_S).sum()) / mins, 1) if mins else 0.0,
        "aplausos_hora": round(60 * len(aplausos) / mins, 1) if mins else 0.0,
        "segundos_entre_reacciones": round(float(np.median(np.diff(inicios))), 1) if len(inicios) > 1 else np.nan,
        "tramo_seco_max_s": round(float(huecos.max()), 0),
        "tramos_secos_hora": round(60 * float((huecos > SECO_S).sum()) / mins, 1) if mins else 0.0,
        "rachas_hora": round(60 * rachas / mins, 1) if mins else 0.0,
        "primera_reaccion_s": round(float(inicios[0] - t0), 0) if ev else np.nan,
        "regularidad_cv": round(float(por_min.std() / por_min.mean()), 2) if por_min.mean() else np.nan,
        "apertura": round(arco[0] / masa_media, 2) if masa_media else np.nan,   # >1: arranca más fuerte que su promedio
        "cierre": round(arco[-1] / masa_media, 2) if masa_media else np.nan,
        "concentracion_top20": round(float(top[:n20].sum() / top.sum()), 2) if top.sum() else np.nan,
        "posicion_picos": round(float(np.median(picos) / max(1, len(masa_min) - 1)), 2) if len(masa_min) else np.nan,
        "secos_relativos_hora": round(60 * secos_rel / mins, 1) if mins else 0.0,
        "peor_decil": int(np.argmin(arco)) + 1,
        "reaccion_sobre_habla_pct": round(100 * float((habla[es_risa] >= UMBRAL_HABLA).mean()), 1) if es_risa.any() else np.nan,
        "reaccion_db_sobre_habla": round(float(db_risa - db_habla), 1) if not np.isnan(db_risa - db_habla) else np.nan,
    }, arco


def main():
    with CATALOGO.open(encoding="utf-8", newline="") as f:
        cat = {r["id"]: r for r in csv.DictReader(f)}
    filas, arcos, sens = [], [], []
    for p in sorted(FRAMES.glob("*.npz")):
        id_ = p.stem
        if id_ not in cat:
            continue
        c = dict(np.load(p))
        m, arco = metricas(c)
        r = cat[id_]
        filas.append({"id": id_, "comediante": r["comediante"], "especial": r["especial"],
                      "anio": int(r["anio"]) if r["anio"] else None, "grupo": r["grupo"], "formato": r["formato"],
                      "imdbID": IMDB.get(id_), **m})
        arcos += [{"id": id_, "decil": i + 1, "masa_min": v} for i, v in enumerate(arco)]
        for u in (0.03, 0.05, 0.08):
            sens.append({"id": id_, "umbral": u, "reacciones_min": metricas(c, u)[0]["reacciones_min"]})
        for pct in (60, 70, 80):   # robustez del umbral relativo
            u = max(UMBRAL_PISO, float(np.percentile(c["risa"] + c["aplauso"], pct)))
            mm = metricas(c, u)[0]
            sens.append({"id": id_, "umbral": f"p{pct}", "reacciones_min": mm["reacciones_min"],
                         "segundos_entre_reacciones": mm["segundos_entre_reacciones"], "rachas_hora": mm["rachas_hora"]})
    df = pd.DataFrame(filas)
    DF.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DF / "df_audio.parquet", index=False)
    pd.DataFrame(arcos).to_parquet(DF / "df_audio_arco.parquet", index=False)
    pd.DataFrame(sens).astype({"umbral": str}).to_parquet(DF / "df_audio_sensibilidad.parquet", index=False)
    print(f"{len(df)} especiales -> {DF.relative_to(RAIZ)}/df_audio.parquet")


if __name__ == "__main__":
    main()
