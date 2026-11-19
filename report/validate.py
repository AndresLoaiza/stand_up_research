"""
Three checks that do not depend on anyone's judgment. Adds the "validacion" key to report_data.json.

1. Attribution: does style identify the comedian of a special the model never saw?
2. Topics vs Wikipedia: do the curated themes found in the transcripts match the comedian's article?
3. Laughter markers vs YAMNet: do the transcript's [laughter] marks agree with the measured audio?

    python report/validate.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore
from scipy.stats import spearmanr

truststore.inject_into_ssl()  # Avast re-signs HTTPS with a root only in the Windows store

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from analysis import REACTION_RE, attribution_test, build_comedian_corpus, load_unified, marker_positions  # noqa: E402
from analysis.compute_curated_topics import THEMES, score_show, tokenize_for_scoring  # noqa: E402
from analysis.core import _words  # noqa: E402

DF = ROOT / "data" / "data_frame"
CHUNK = 2000
MFW = 150
WIKI_FIVE = ["George Carlin", "Russell Peters", "Jim Gaffigan", "Hannah Gadsby", "Bill Hicks"]
UA = {"User-Agent": "TextMiningComedyReport/1.0 (personal research project; python-requests)"}
STYLE_SECTION = re.compile(r"style|comed|stand-up|standup|material|theme|influence|career|act\b", re.IGNORECASE)
SKIP_SECTION = re.compile(r"personal life|filmography|discography|awards|bibliography|references|external|see also|notes|death|legacy",
                          re.IGNORECASE)


def f(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def attribution(corpus: pd.DataFrame) -> dict:
    toks = {i: _words(t) for i, t in zip(corpus.index, corpus["transcript"])}
    mfw = [w for w, _ in sum((Counter(t) for t in toks.values()), Counter()).most_common(MFW)]
    rows = []
    for i, r in corpus.iterrows():
        t = toks[i]
        for k in range(0, len(t) - CHUNK // 2, CHUNK):
            ch = t[k:k + CHUNK]
            if len(ch) < CHUNK // 2:
                continue
            c = Counter(ch)
            rows.append({"comedian": r["comedian"], "special": f"{r['comedian']}|{r['title']}",
                         **{w: c[w] / len(ch) for w in mfw}})
    res = attribution_test(pd.DataFrame(rows), n_perm=1000)
    ratio = res["acierto_especial"] / res["azar"]
    return {
        "metodo": (f"Cada especial se corta en fragmentos de {CHUNK} palabras y se describe con la frecuencia de las "
                   f"{MFW} palabras más comunes del corpus (estilometría clásica: Delta de Burrows en su forma de coseno). "
                   "Para cada especial, el modelo aprende con todos los demás especiales y adivina de quién es; el especial "
                   "va al comediante que eligen la mayoría de sus fragmentos. Las permutaciones barajan los nombres entre "
                   "especiales para ver cuánto acierta el método por suerte."),
        "comediantes": str(res["comediantes"]), "especiales": str(res["especiales"]), "fragmentos": str(res["fragmentos"]),
        "acierto_especial": f(100 * res["acierto_especial"], 0), "acierto_fragmento": f(100 * res["acierto_fragmento"], 0),
        "azar": f(100 * res["azar"], 1), "mayoritaria": f(100 * res["mayoritaria"], 0),
        "permutacion_p95": f(100 * res["permutacion_p95"], 0), "permutaciones": str(res["permutaciones"]),
        "p": f(res["p_permutacion"], 3), "veces_azar": f(ratio, 1),
        "lectura": (f"Lectura: acertar {f(ratio, 1)} veces más que el azar con palabras tan comunes como «the», «and» o «I» "
                    "dice que cada comediante tiene una firma de estilo estable entre especiales, y que las métricas de "
                    "estilo de las páginas miden algo del comediante y no solo ruido. No dice que esas métricas sean las "
                    "que hacen reír. Parte de la firma puede ser del transcriptor: casi todo viene del mismo sitio, lo que "
                    "reduce ese riesgo pero no lo elimina."),
    }


def wiki_text(title: str) -> str:
    for attempt in range(6):
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30,
                         params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                 "format": "json", "redirects": 1})
        if r.status_code != 429:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def style_sections(text: str) -> str:
    parts = re.split(r"\n(={2,}\s*[^=\n]+?\s*={2,})\n", text)
    keep, intro = [], parts[0]
    for i in range(1, len(parts) - 1, 2):
        head, body = parts[i].strip("= ").strip(), parts[i + 1]
        if STYLE_SECTION.search(head) and not SKIP_SECTION.search(head):
            keep.append(body)
    return " ".join([intro] + keep)


def wiki_check(data: dict) -> dict:
    cache = json.loads((HERE / "wiki_cache.json").read_text(encoding="utf-8"))
    labels = {t["id"]: t["label"] for t in THEMES}
    rows = []
    for n in WIKI_FIVE:
        c = next(x for x in data["comedians"] if x["name"] == n)
        mine = {t["id"]: float(t["valor"].replace(",", ".").replace("−", "-")) for t in c["topics"]}
        title = cache.get(n, {}).get("enwiki_title") or n
        text = style_sections(wiki_text(title))
        cnt = tokenize_for_scoring(text)
        total = sum(cnt.values())
        theirs = {t["id"]: score_show(cnt, t["lexicon"], total) for t in THEMES}
        ids = list(labels)
        rho, p = spearmanr([mine[i] for i in ids], [theirs[i] for i in ids])
        ids7 = [i for i in ids if i != "meta-comedy"]
        rho7 = spearmanr([mine[i] for i in ids7], [theirs[i] for i in ids7])[0]
        top = lambda d: ", ".join(labels[i] for i in sorted(ids, key=lambda i: -d[i])[:2])
        rows.append({"name": n, "top_text": top(mine), "top_wiki": top(theirs), "rho": f(rho, 2), "p": f(p, 3), "rho7": f(rho7, 2), "_rho7": rho7,
                     "_sex": "sex-body" in sorted(ids, key=lambda i: -mine[i])[:2] and "sex-body" not in sorted(ids, key=lambda i: -theirs[i])[:2],
                     "palabras": str(total), "_rho": rho, "_overlap": len(set(sorted(ids, key=lambda i: -mine[i])[:2])
                                                                           & set(sorted(ids, key=lambda i: -theirs[i])[:2]))})
        time.sleep(1)
    pos = sum(r["_rho"] > 0 for r in rows)
    pos7 = sum(r["_rho7"] > 0 for r in rows)
    ov = sum(r["_overlap"] >= 1 for r in rows)
    sex = sum(r["_sex"] for r in rows)
    return {
        "metodo": ("Del artículo en inglés de Wikipedia se toma la introducción y las secciones sobre su comedia, estilo o "
                   "carrera (sin vida personal ni filmografía) y se le aplican los mismos ocho léxicos. ρ es la correlación "
                   "de Spearman entre los ocho temas de sus transcripts y los ocho del artículo; con ocho temas, p chica es difícil."),
        "filas": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "lectura": (f"En {pos} de {len(rows)} la correlación es positiva y en {ov} de {len(rows)} coincide al menos uno de los dos "
                    "temas más fuertes. La prueba no confirma los temas. Dos cosas que muestra: (a) un artículo sobre un "
                    "comediante habla de su oficio, así que meta-comedia gana casi siempre del lado de Wikipedia; sin ese tema "
                    f"(decidido después de ver los datos, columna ρ sin meta) la correlación es positiva en {pos7} de {len(rows)}. "
                    f"(b) En {sex} de {len(rows)}, «sexo, cuerpo y comedia cruda» está entre los dos temas más fuertes del "
                    "transcript pero no del artículo: su léxico incluye groserías (fuck, ass), así que en comediantes muy "
                    "groseros mide lenguaje crudo más que sexo como tema. Leer ese tema junto con la métrica de groserías."),
    }


def markers_check() -> dict:
    df = load_unified()
    aud = pd.read_parquet(DF / "df_audio.parquet")
    arco = pd.read_parquet(DF / "df_audio_arco.parquet")
    aud = aud[(aud["grupo"] == "referente") & aud["imdbID"].notna()]
    rows = []
    for a in aud.itertuples():
        t = df.loc[df["imdbID"] == a.imdbID, "transcript_raw"]
        if t.empty:
            continue
        raw = t.iloc[0]
        n = len(REACTION_RE.findall(raw))
        if n < 10:
            continue
        pos = marker_positions(raw)
        hist = np.histogram(pos, bins=10, range=(0, 1))[0]
        sh = arco[arco["id"] == a.id].sort_values("decil")["masa_min"].to_numpy()
        rho = spearmanr(hist, sh)[0] if len(sh) == 10 and hist.std() > 0 else np.nan
        rows.append({"especial": f"{a.comediante}: {a.especial}", "marcas_min": n / a.minutos, "reac_min": a.reacciones_min,
                     "rho": rho})
    r = pd.DataFrame(rows)
    rho_level, p_level = spearmanr(r["marcas_min"], r["reac_min"]) if len(r) >= 4 else (np.nan, np.nan)
    med = float(np.nanmedian(r["rho"])) if len(r) else np.nan
    return {
        "metodo": ("Especiales con transcript que marca las risas (diez marcas o más) y con video medido. Nivel: marcas por "
                   "minuto contra reacciones por minuto del audio, entre especiales. Forma: cuántas marcas caen en cada décimo "
                   "del texto contra la reacción del audio en cada décimo del show (posición en palabras como aproximación del "
                   "tiempo)."),
        "resumen": (f"{len(r)} especiales tienen las dos fuentes. Nivel: ρ = {f(rho_level, 2)} (p = {f(p_level, 3)}). "
                    f"Forma por décimos: mediana de ρ = {f(med, 2)}; positiva en {int((r['rho'] > 0).sum())} de {len(r)}."),
        "filas": [{"especial": x.especial, "marcas_min": f(x.marcas_min, 2), "reac_min": f(x.reac_min, 2), "rho": f(x.rho, 2)}
                  for x in r.itertuples()],
        "lectura": (f"Con {len(r)} especiales no hay acuerdo: ni el nivel ni la forma de las marcas siguen a la risa medida. Las "
                    "marcas dicen más del transcriptor (cuántas anota y dónde) que del público. Por eso «palabras entre risas» "
                    "aparece en las páginas marcada como no validada y no se usa en «qué robarle»."),
    }


def main():
    path = HERE / "report_data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus, _ = build_comedian_corpus()
    v = {"atrib": attribution(corpus)}
    print("atribución", v["atrib"]["acierto_especial"], v["atrib"]["azar"], v["atrib"]["p"])
    v["marcas"] = markers_check()
    print(v["marcas"]["resumen"])
    v["wiki"] = wiki_check(data)
    print(v["wiki"]["lectura"])
    data["validacion"] = v
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
