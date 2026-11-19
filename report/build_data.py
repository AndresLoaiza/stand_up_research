"""
Every number of the per-comedian report, computed here and written to report/report_data.json.
The template only prints strings from this file; report/check_numbers.py fails if the PDF shows a
number that is not in it.

    python report/build_data.py          (needs report/wiki_cache.json from fetch_wiki.py)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audio"))

from analysis import (  # noqa: E402
    DEFAULT_STOPWORDS, build_comedian_corpus, bootstrap_percentile, callbacks, dry_chains, fightin_words,
    laugh_gaps, load_unified, memory_permutation_p, percentile_of, style_from_blocks, text_blocks, tokenize,
)
from analysis.compute_curated_topics import THEMES, score_show, tokenize_for_scoring  # noqa: E402
from analysis.core import BLOCK, _ES_STOP, _words  # noqa: E402
import metricas as M  # noqa: E402
import sala  # noqa: E402

DF = ROOT / "data" / "data_frame"
VIDEOS = ROOT.parent / "Analisis_videos" / "output"
FOTOS_AC = ROOT.parent / "AltaComediaWeb" / "public" / "imagenes" / "comediantes"
RNG_SEED = 0
N_BOOT = int(__import__("os").environ.get("N_BOOT", 1000))

# ---------------------------------------------------------------- formatting: every printed number is a string
def f(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def pct(x):
    return "—" if x is None or not np.isfinite(x) else str(int(round(x)))


STYLE = [  # key, label, unit, decimals, source of the CI
    ("palabras_min", "Palabras por minuto", "palabras por minuto de duración", 0, "especiales"),
    ("ttr", "Variedad de vocabulario", "palabras distintas por cada 1000 (TTR)", 0, "bloques"),
    ("largo_frase", "Largo de frase", "palabras por frase", 1, "bloques"),
    ("yo_vs_tu", "Yo frente a tú", "% de pronombres en primera persona", 0, "bloques"),
    ("preguntas_1000", "Preguntas", "frases con pregunta por 1000 palabras", 1, "bloques"),
    ("groserias_1000", "Groserías", "por 1000 palabras", 1, "bloques"),
    ("callbacks_10k", "Callbacks", "frases que vuelven 1500+ palabras después, por 10000 palabras", 1, "especiales"),
    ("tramo_risas", "Palabras entre risas", "mediana entre marcas de risa del transcript; no validada contra el audio", 0, "tramos"),
]
SCALE = {"ttr": 1000}  # TTR shown as distinct words per 1000

READ = {  # (metric, high/low) -> reading. Interpretation, not data: the report labels it "lectura".
    ("palabras_min", "alto"): "Ritmo de ametralladora: la energía sale de no soltar al público. Para robarlo: recorta conectores y muletillas de un bit y pruébalo más rápido sin acelerar el remate.",
    ("palabras_min", "bajo"): "Habla despacio y deja trabajar el silencio. Para robarlo: antes del remate más fuerte de tu set, baja la velocidad y sostén la pausa un tiempo más del que te parece cómodo.",
    ("ttr", "alto"): "Vocabulario preciso y variado: nombra las cosas exactas. Para robarlo: reescribe un bit cambiando cada palabra genérica (cosa, vaina, man) por la concreta.",
    ("ttr", "bajo"): "Repite palabras a propósito: lenguaje llano y estructuras que vuelven. Para robarlo: repite la misma estructura de frase tres veces; el público anticipa y el giro de la tercera pega más.",
    ("largo_frase", "alto"): "Frases largas: el chiste vive en la acumulación de detalles dentro de una historia. Para robarlo: deja que una frase crezca con detalles absurdos antes de cerrarla.",
    ("largo_frase", "bajo"): "Frases cortas, golpes secos. Para robarlo: corta tus remates a la mínima cantidad de palabras; la última palabra debe ser la sorpresa.",
    ("yo_vs_tu", "alto"): "Material confesional: casi todo es sobre sí mismo. Para robarlo: el chiste a tu costa baja la defensa del público antes de un tema delicado.",
    ("yo_vs_tu", "bajo"): "Le habla a la sala (tú, ustedes) más que de sí mismo. Para robarlo: convierte una observación propia en «ustedes también hacen esto» y mira si la sala se reconoce.",
    ("preguntas_1000", "alto"): "Pregunta mucho, retórica o directamente a la sala. Para robarlo: abre un bit con una pregunta que el público conteste en su cabeza antes de tu respuesta.",
    ("preguntas_1000", "bajo"): "Casi no pregunta: afirma. Para robarlo: di la premisa como un hecho, sin pedir permiso; la seguridad también es parte del chiste.",
    ("groserias_1000", "alto"): "La grosería es ritmo y acento. Para robarlo: úsala para marcar el golpe, no de relleno entre frases.",
    ("groserias_1000", "bajo"): "Limpio: el chiste no depende de la grosería. Para robarlo: prueba tu bit más grosero sin una sola; si sigue funcionando, el chiste era bueno.",
    ("callbacks_10k", "alto"): "Vuelve a frases de antes del show: el público premia reconocerlas. Para robarlo: anota tus tres imágenes más fuertes del primer tercio y regresa a una en el cierre.",
    ("callbacks_10k", "bajo"): "Bits autónomos que no se citan entre sí. Para robarlo: es el formato para probar material nuevo, porque el orden importa menos.",
    ("tramo_risas", "alto"): "Mucho texto entre risas: historias largas que pagan al final. Para robarlo: un tramo sin risa se tolera si la recompensa es grande, pero no encadenes dos.",
    ("tramo_risas", "bajo"): "Risa cada pocas palabras: densidad de chistes muy alta. Para robarlo: busca un remate o una etiqueta cada dos o tres frases del bit.",
}

ES_THEMES = {  # Spanish seed lexicons for the same 8 themes (translation of THEMES, same stem-prefix scoring)
    "family-domestic": ["esposa", "esposo", "marido", "hijo", "hija", "niño", "niña", "mamá", "mami", "madre", "papá",
                        "papi", "padre", "familia", "casa", "hogar", "bebé", "matrimonio", "casad", "divorci",
                        "colegio", "tarea", "abuel", "herman", "novi"],
    "race-identity": ["negro", "negra", "blanco", "blanca", "indio", "indígena", "racis", "minoría", "cultura",
                      "inmigra", "venezolan", "gringo", "paisa", "costeñ", "rolo", "estereotipo", "barrio"],
    "sex-body": ["sexo", "sexual", "culo", "teta", "pene", "verga", "vagina", "chimba", "desnud", "porno", "condón",
                 "orgasm", "masturb", "polvo", "cuca", "huevas"],
    "politics-america": ["petro", "uribe", "presidente", "gobierno", "política", "polític", "congreso", "votar", "voto",
                         "elecciones", "guerra", "ejército", "colombia", "país", "estados unidos", "gringo"],
    "religion-death": ["dios", "jesús", "cristo", "religión", "iglesia", "cura", "rezar", "orar", "cielo", "infierno",
                       "alma", "pecado", "morir", "muerte", "muert", "matar", "mató", "funeral", "ateo", "creer", "fe"],
    "meta-comedy": ["chiste", "comedia", "comediante", "público", "show", "tarima", "escenario", "remate", "stand",
                    "material", "escribir", "aplauso", "micrófono"],
    "technology-modern": ["celular", "teléfono", "internet", "google", "facebook", "instagram", "twitter", "tiktok",
                          "youtube", "netflix", "app", "computador", "correo", "whatsapp", "redes", "wifi",
                          "inteligencia artificial", "chatgpt"],
    "health-aging": ["médico", "doctor", "hospital", "enferm", "cáncer", "pastilla", "medicina", "droga", "marihuana",
                     "trago", "borrach", "alcohol", "gimnasio", "gordo", "gorda", "peso", "dieta", "viejo", "vieja",
                     "edad", "corazón", "ansiedad", "psicólog", "terapia"],
}
NO_LESSON = {"tramo_risas"}  # transcript laugh marks did not agree with measured audio (validate.py)
TOPIC_UNIT = "palabras del tema por cada 1000"
GREETING = re.compile(r"gracias|thank|buenas noches|good night|aplauso|applause")
ES_FILLERS = {"entonces", "pues", "bueno", "como", "así", "cierto", "porque", "vez", "ahí", "acá", "allá", "aquí",
              "también", "dijo", "digo", "dice", "tiene", "tengo", "hacer", "hace", "voy", "va", "ver", "uno", "una",
              "dos", "tres", "sí", "no", "ya", "ahora", "todo", "toda", "todos", "nada", "algo", "cosa", "cosas", "eh",
              "ah", "oye", "mira", "listo", "bien", "gracias", "gente", "hay", "era", "fue", "ser", "estar", "están",
              "estoy", "está", "estaba", "puede", "quiero", "sé", "creo", "hizo", "dije", "vamos"}

MEMBERS = {  # Alta Comedia: (show, catalog id, transcript folder, start s, end s or None = whole show)
    "Andrés Loaiza": [("Ansioso (ago 2025)", "alta-comedia-ansioso-2025-08-22", "ansioso/2025-08-22", None, None),
                      ("Hoy Nace un Chiste (17 jul 2026)", "alta-comedia-hoy-nace-un-chiste-2026-07-17",
                       "hoy_nace_un_chiste/2026-07-17", 900, 2244)],
    "Wilmar Tovar": [("Hoy Nace un Chiste (17 jul 2026)", "alta-comedia-hoy-nace-un-chiste-2026-07-17",
                      "hoy_nace_un_chiste/2026-07-17", 0, 900)],
    "Daniela Villa": [("Hoy Nace un Chiste (17 jul 2026)", "alta-comedia-hoy-nace-un-chiste-2026-07-17",
                       "hoy_nace_un_chiste/2026-07-17", 2244, 2718)],
    "Alex Marín": [("Hoy Nace un Chiste (17 jul 2026)", "alta-comedia-hoy-nace-un-chiste-2026-07-17",
                    "hoy_nace_un_chiste/2026-07-17", 2718, 4290)],
}
FOTO_MEMBER = {"Andrés Loaiza": "andres-loaiza/retrato-01.jpg", "Wilmar Tovar": "wilmar-tovar/retrato-01.jpg",
               "Daniela Villa": "daniela-villa/retrato-01.jpg", "Alex Marín": "alex-marin/retrato-01.jpg"}


def slug(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fix_mojibake(s: str) -> str:
    """Some parquet strings were written with the Windows code page ('Andr�s')."""
    return s.replace("Andr�s", "Andrés").replace("Mar�n", "Marín") if isinstance(s, str) else s


# ---------------------------------------------------------------- sentences with location, for quotes
def sentences_of(text_raw: str) -> list[tuple[str, float]]:
    from analysis.core import _ANY_STAGE_RE
    s = _ANY_STAGE_RE.sub(" ", (text_raw or "").replace("’", "'").replace("♪", " "))
    sents = [x.strip() for x in re.split(r"(?<=[.!?…])\s+", s) if x.strip()]
    total = sum(len(_words(x)) for x in sents) or 1
    out, acc = [], 0
    for x in sents:
        out.append((x, acc / total))
        acc += len(_words(x))
    return out


def trim(sent: str, key: str | None = None, max_words: int = 28) -> str:
    w = sent.split()
    if len(w) <= max_words:
        return sent
    i = next((k for k, t in enumerate(w) if key and key in t.lower()), 0)
    a = max(0, min(i - max_words // 2, len(w) - max_words))
    return ("… " if a else "") + " ".join(w[a:a + max_words]) + (" …" if a + max_words < len(w) else "")


def where(title: str, frac: float, runtime) -> str:
    if runtime and np.isfinite(runtime):
        return f"{title}, hacia el minuto {int(round(frac * runtime))}"
    return f"{title}, al {int(round(frac * 100))} % del texto"


# ---------------------------------------------------------------- per-special measurements
def measure_special(text_raw: str, text_clean: str, lang: str = "en") -> dict:
    lex = {t["id"]: t["lexicon"] for t in THEMES} if lang == "en" else ES_THEMES
    cnt = tokenize_for_scoring(text_clean) if lang == "en" else Counter(_words(text_clean))
    total = sum(cnt.values())
    return {
        "blocks": text_blocks(text_raw, lang),
        "callbacks": callbacks(text_clean, lang),
        "gaps": laugh_gaps(text_raw) if lang == "en" else None,
        "topics": {k: score_show(cnt, v, total) for k, v in lex.items()},
        "tokens": len(_words(text_clean)),
    }


def boot_specials(vals: list[float], weights: list[float], others: list[float], rng) -> tuple[float, float] | None:
    """Percentile CI by resampling specials (metrics that only exist per special). None with one special."""
    if len(vals) < 2:
        return None
    v, w = np.array(vals, float), np.array(weights, float)
    res = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(v), len(v))
        res.append(percentile_of(np.average(v[i], weights=w[i]), others))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def comedian_metrics(specs: list[dict]) -> dict:
    blocks = pd.concat([s["m"]["blocks"] for s in specs], ignore_index=True)
    st = style_from_blocks(blocks)
    rt = [(s["words"], s["runtime"]) for s in specs if s["runtime"] and np.isfinite(s["runtime"])]
    st["palabras_min"] = sum(w for w, _ in rt) / sum(r for _, r in rt) if rt else np.nan
    tok = sum(s["m"]["tokens"] for s in specs)
    st["callbacks_10k"] = 10000 * sum(s["m"]["callbacks"]["n"] for s in specs) / tok
    gaps = [g for s in specs if s["m"]["gaps"] for g in s["m"]["gaps"]]
    st["tramo_risas"] = float(np.median(gaps)) if gaps else np.nan
    # TTR depends on length: below one full block it is not comparable with the corpus
    st["ttr"] = st["ttr"] * SCALE["ttr"] if blocks["tokens"].sum() >= BLOCK else np.nan
    topics = {k: float(np.average([s["m"]["topics"][k] for s in specs], weights=[s["m"]["tokens"] for s in specs]))
              for k in specs[0]["m"]["topics"]}
    return {"style": st, "topics": topics, "blocks": blocks, "gaps": gaps}


def ci_for(key: str, data: dict, specs: list[dict], others: list[float], rng):
    if key in ("ttr", "largo_frase", "yo_vs_tu", "preguntas_1000", "groserias_1000"):
        o = [x / SCALE.get(key, 1) for x in others]
        return bootstrap_percentile(data["blocks"], key, o, n_boot=N_BOOT, seed=RNG_SEED)
    if key == "palabras_min":
        rt = [s for s in specs if s["runtime"] and np.isfinite(s["runtime"])]
        return boot_specials([s["words"] / s["runtime"] for s in rt], [s["runtime"] for s in rt], others, rng)
    if key == "callbacks_10k":
        return boot_specials([s["m"]["callbacks"]["por_10k"] for s in specs], [s["m"]["tokens"] for s in specs], others, rng)
    if key == "tramo_risas" and len(data["gaps"]) >= 10:
        g = np.array(data["gaps"])
        res = [percentile_of(float(np.median(g[rng.integers(0, len(g), len(g))])), others) for _ in range(N_BOOT)]
        return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))
    return None


# ---------------------------------------------------------------- audio
def load_frames(id_: str) -> dict:
    return dict(np.load(M.FRAMES / f"{id_}.npz"))


def audio_block(ids: list[str], aud: pd.DataFrame, sala_df: pd.DataFrame, secos: pd.DataFrame,
                arco: pd.DataFrame, ref_ids: list[str]) -> dict | None:
    rows = aud[aud["id"].isin(ids)]
    if rows.empty:
        return None
    ref = aud[aud["id"].isin(ref_ids)]
    ref_sala = sala_df[sala_df["id"].isin(ref_ids)]
    out = []
    for r in rows.itertuples():
        s = sala.serie(load_frames(r.id))
        mem, p = memory_permutation_p(s)
        sec = secos[secos["id"] == r.id]
        ch = dry_chains(sec)
        sr = sala_df[sala_df["id"] == r.id].iloc[0]
        a = arco[arco["id"] == r.id].sort_values("decil")["masa_min"].to_numpy()
        shape = (a / a.mean()).tolist() if len(a) and a.mean() else []
        out.append({
            "especial": fix_mojibake(r.especial), "minutos": f(r.minutos, 0),
            "reacciones_min": f(r.reacciones_min, 2), "reacciones_pct": pct(percentile_of(r.reacciones_min, ref["reacciones_min"])),
            "apertura": f(r.apertura, 2), "cierre": f(r.cierre, 2),
            "peor_decil": str(int(r.peor_decil)) if pd.notna(r.peor_decil) else "—",
            "memoria": f(mem, 2), "memoria_p": f(p, 3), "ventanas": str(len(s)),
            "secos_hora": f(sr.tramos_secos_hora, 1), "secos_pct": pct(percentile_of(sr.tramos_secos_hora, ref_sala["tramos_secos_hora"])),
            "secos_n": str(len(sec)), "cadenas_n": str(len(ch)),
            "cadenas": [{"desde": f(c["desde_min"], 1), "hasta": f(c["hasta_min"], 1), "tramos": str(c["tramos"]),
                         "seco_s": str(c["seco_s"])} for c in ch],
            "recupera_s": f(sr.recupera_mediana_s, 0),
            "shape": shape,
        })
    return {"especiales": out}


def ref_shape(arco: pd.DataFrame, ref_ids: list[str]) -> list[list[float]]:
    sub = arco[arco["id"].isin(ref_ids)].copy()
    sub["rel"] = sub["masa_min"] / sub.groupby("id")["masa_min"].transform("mean")
    q = sub.groupby("decil")["rel"].quantile([0.25, 0.5, 0.75]).unstack()
    return [q[0.25].tolist(), q[0.5].tolist(), q[0.75].tolist()]


# ---------------------------------------------------------------- lessons
def lessons(style_rows: list[dict], topic_rows: list[dict], n_specials: int, k: int = 3, skip=frozenset()) -> list[dict]:
    cand = []
    for r in style_rows:
        if r["_pct"] is None or not np.isfinite(r["_pct"]) or r["key"] in NO_LESSON or r["key"] in skip:
            continue
        lo, hi = r["_ci"] if r["_ci"] else (r["_pct"], r["_pct"])
        if lo >= 70 or hi <= 30 or (r["_ci"] is None and (r["_pct"] >= 85 or r["_pct"] <= 15)):
            d = "alto" if r["_pct"] >= 50 else "bajo"
            cand.append((abs(r["_pct"] - 50), {
                "titulo": r["label"],
                "dato": f"{r['label']}: {r['valor']} ({r['unidad']}); percentil {r['pct']} del corpus"
                        + (f", IC 95 % {r['ci_lo']}–{r['ci_hi']}" if r["_ci"] else ", sin intervalo") + ".",
                "lectura": READ[(r["key"], d)]}))
    top = max(topic_rows, key=lambda t: t["_pct"]) if topic_rows else None
    if top and top["_pct"] >= 85:
        cand.append((top["_pct"] - 50 - 0.5, {
            "titulo": f"Tema: {top['label']}",
            "dato": f"{top['label']}: {top['valor']} palabras del tema por 1000; percentil {top['pct']} del corpus.",
            "lectura": f"Su terreno es «{top['label'].lower()}». Para robarlo: mira en la cita cómo entra al tema; el ángulo personal, no el tema, es lo que hace el chiste."}))
    cand.sort(key=lambda x: -x[0])
    out = [c for _, c in cand[:k]]
    if n_specials == 1:
        for o in out:
            o["dato"] += " Rasgo del especial: hay un solo texto."
    return out


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(RNG_SEED)
    wiki = json.loads((HERE / "wiki_cache.json").read_text(encoding="utf-8"))
    df = load_unified()
    corpus, log = build_comedian_corpus(df)
    corpus["title"] = corpus["title"].map(display_title)

    # ---- measure every special
    specs_by = {}
    for r in corpus.itertuples():
        m = measure_special(r.transcript_raw, r.transcript)
        specs_by.setdefault(r.comedian, []).append({
            "title": r.title, "year": r.year, "runtime": r.runtime_min, "source": r.source,
            "words": len(_words(r.transcript)), "raw": r.transcript_raw, "clean": r.transcript, "m": m})
    names = sorted(specs_by)
    data = {n: comedian_metrics(specs_by[n]) for n in names}
    decade = {}
    for n in names:
        ys = [s["year"] for s in specs_by[n] if pd.notna(s["year"])]
        decade[n] = int(np.median(ys) // 10 * 10) if ys else None

    # ---- word counts for Fightin' Words
    counts = {n: Counter(t for s in specs_by[n] for t in tokenize(s["clean"])) for n in names}
    docfreq = {n: Counter(t for s in specs_by[n] for t in set(tokenize(s["clean"]))) for n in names}
    prior = sum(counts.values(), Counter())

    # ---- audio tables
    aud = pd.read_parquet(DF / "df_audio.parquet")
    aud["comediante"] = aud["comediante"].map(fix_mojibake)
    sala_df = pd.read_parquet(DF / "df_audio_sala.parquet")
    secos = pd.read_parquet(DF / "df_audio_secos.parquet")
    arco = pd.read_parquet(DF / "df_audio_arco.parquet")
    ref_ids = aud.loc[aud["grupo"] == "referente", "id"].tolist()
    shape_ref = ref_shape(arco, ref_ids)

    comedians = []
    for n in names:
        specs, d = specs_by[n], data[n]
        others = [x for x in names if x != n]
        peers = [x for x in others if decade[x] is not None and decade[x] == decade[n]]
        style_rows = []
        for key, label, unit, dec, src in STYLE:
            v = d["style"][key]
            oth = [data[x]["style"][key] for x in others]
            p = percentile_of(v, oth)
            ci = ci_for(key, d, specs, oth, rng) if np.isfinite(v) else None
            pdec = percentile_of(v, [data[x]["style"][key] for x in peers]) if len(peers) >= 4 else None
            style_rows.append({"key": key, "label": label, "unidad": unit, "valor": f(v, dec),
                               "pct": pct(p), "ci_lo": pct(ci[0]) if ci else "—", "ci_hi": pct(ci[1]) if ci else "—",
                               "pct_decada": pct(pdec) if pdec is not None else "—",
                               "_pct": p if np.isfinite(v) else None, "_ci": ci,
                               "x": p if np.isfinite(p) else None, "x_lo": ci[0] if ci else None, "x_hi": ci[1] if ci else None})
        topic_rows = []
        for t in THEMES:
            v = d["topics"][t["id"]]
            p = percentile_of(v, [data[x]["topics"][t["id"]] for x in others])
            ci = boot_specials([s["m"]["topics"][t["id"]] for s in specs], [s["m"]["tokens"] for s in specs],
                               [data[x]["topics"][t["id"]] for x in others], rng)
            topic_rows.append({"id": t["id"], "label": t["label"], "unidad": TOPIC_UNIT, "valor": f(v, 1), "pct": pct(p),
                               "ci_lo": pct(ci[0]) if ci else "—", "ci_hi": pct(ci[1]) if ci else "—",
                               "_pct": p, "x": p, "x_lo": ci[0] if ci else None, "x_hi": ci[1] if ci else None})
        # distinctive words: vs everyone else, recurring across specials when there are 2+
        rest = sum((counts[x] for x in others), Counter())
        fw = fightin_words(counts[n], rest, prior)
        if len(specs) >= 2:
            fw = fw[fw["palabra"].map(lambda w: docfreq[n][w] >= 2)]
        fw = fw[fw["z"] > 1.96].head(10)
        words = [{"palabra": r.palabra, "z": f(r.z, 1), "n": str(int(r.n_a))} for r in fw.itertuples()]

        # quotes
        quotes = []
        if words:
            w0 = words[0]["palabra"]
            for s in specs:
                hit = next(((x, fr) for x, fr in sentences_of(s["raw"]) if re.search(rf"\b{re.escape(w0)}\b", x.lower())
                            and 6 <= len(x.split()) <= 60), None)
                if hit:
                    quotes.append({"tipo": f"Palabra distintiva «{w0}»", "texto": trim(hit[0], w0),
                                   "donde": where(s["title"], hit[1], s["runtime"])})
                    break
        best_t = max(topic_rows, key=lambda t: t["_pct"])
        lex = next(t["lexicon"] for t in THEMES if t["id"] == best_t["id"])
        best = None
        for s in specs:
            for x, fr in sentences_of(s["raw"]):
                if not 6 <= len(x.split()) <= 45:
                    continue
                toks = re.findall(r"[a-z\-']+", x.lower())
                h = sum(any(tk.startswith(w) for w in lex) for tk in toks)
                if h >= 2 and (best is None or h > best[0]):
                    best = (h, x, fr, s)
        if best:
            quotes.append({"tipo": f"Tema: {best_t['label']}", "texto": trim(best[1]),
                           "donde": where(best[3]["title"], best[2], best[3]["runtime"])})
        cb = next(((s, e) for s in specs for e in s["m"]["callbacks"]["ejemplos"] if not GREETING.search(e["frase"])), None)
        if cb:
            s, e = cb
            quotes.append({"tipo": "Callback", "texto": f"«{e['frase']}»",
                           "donde": f"{s['title']}: aparece al {int(round(100 * e['pos1']))} % y vuelve al {int(round(100 * e['pos2']))} % del texto"})

        w = wiki.get(n, {})
        ys = sorted({int(s["year"]) for s in specs if pd.notna(s["year"])})
        ids = aud.loc[(aud["comediante"] == n) & (aud["grupo"] == "referente"), "id"].tolist()
        rating = df[df["comedian"] == n]
        comedians.append({
            "name": n, "slug": slug(n), "n": str(len(specs)), "n1": len(specs) == 1,
            "decada": f"{decade[n]}s" if decade[n] else "—", "pares_decada": str(len(peers)),
            "years": f"{ys[0]}–{ys[-1]}" if len(ys) > 1 else (str(ys[0]) if ys else "—"),
            "palabras": str(sum(s["words"] for s in specs)),
            "specials": [{"title": s["title"], "year": str(int(s["year"])) if pd.notna(s["year"]) else "",
                          "fuente": "IMDb + transcript" if s["source"] == "imdb" else "transcript",
                          "rating": next((f(x, 1) for x in rating.loc[rating["title"] == s["title"], "rating"] if pd.notna(x)), "")}
                         for s in sorted(specs, key=lambda s: (s["year"] if pd.notna(s["year"]) else 9999))],
            "bio": {"nacimiento": str(w["birth_year"]) if w.get("birth_year") else "—",
                    "muerte": str(w["death_year"]) if w.get("death_year") else "",
                    "origen": w.get("birthplace") or "—",
                    "activo": f"desde {w['active_from']}" + (f" hasta {w['active_to']}" if w.get("active_to") else "") if w.get("active_from") else "—",
                    "texto": first_sentences((w.get("wikipedia") or {}).get("extract")),
                    "fuente": (w.get("wikipedia") or {}).get("url") or "", "wikidata": w.get("wikidata_url") or "",
                    "lang": (w.get("wikipedia") or {}).get("lang") or ""},
            "photo": w.get("photo"),
            "style": [{k: v for k, v in r.items() if not k.startswith("_")} for r in style_rows],
            "topics": [{k: v for k, v in r.items() if not k.startswith("_")} for r in topic_rows],
            "words": words, "quotes": quotes[:3],
            "lessons": lessons(style_rows, topic_rows, len(specs)),
            "audio": audio_block(ids, aud, sala_df, secos, arco, ref_ids),
        })

    members = build_members(data, names, aud, sala_df, secos, arco, ref_ids, rng)
    annex = build_annex(aud, sala_df, secos, names, ref_ids)
    mapa = build_map(comedians, data, names)

    out = {
        "meta": {
            "fecha": date.today().isoformat(), "comediantes": str(len(names)), "especiales": str(len(corpus)),
            "especiales_imdb": str(int((corpus["source"] == "imdb").sum())),
            "especiales_raw": str(int((corpus["source"] == "raw").sum())),
            "raw_total": str(len(pd.read_parquet(DF / "raw_transcripts.parquet"))),
            "emparejamiento": {k: str(v) for k, v in log["decision"].value_counts().items()},
            "rechazos": [{"comedian": r.comedian, "title": r.title, "decision": r.decision, "reason": r.reason}
                         for r in log[log["decision"] == "rechazado"].itertuples()],
            "n1": str(sum(len(specs_by[n]) == 1 for n in names)),
            "bloque": str(BLOCK), "ic": "95 %", "boot": str(N_BOOT), "callback_gap": "1500", "rating_min": "7,0", "rating_max": "8,7",
            "con_marcas": str(sum(1 for n in names for s in specs_by[n] if s["m"]["gaps"])),
            "referentes_audio": str(len(ref_ids)),
            "metricas": [{"label": l, "unidad": u} for _, l, u, _, _ in STYLE],
            "shape_ref": shape_ref,
        },
        "mapa": mapa, "comedians": comedians, "members": members, "annex": annex,
    }
    (HERE / "report_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.to_csv(HERE / "emparejamiento.csv", index=False, encoding="utf-8-sig")
    print("comediantes", len(comedians), "integrantes", len(members), "anexo", len(annex))


def display_title(t: str) -> str:
    t = re.sub(r"\s+", " ", str(t)).strip()
    if sum(c.isupper() for c in t) > 0.6 * sum(c.isalpha() for c in t):  # ALL CAPS scraped titles
        t = " ".join(w if w.strip(":,") in {"HBO", "USC", "SNL", "TX", "CA", "II", "IV"} else w.capitalize() for w in t.split())
        t = re.sub(r"\b([A-Za-z])\.([a-z])\.", lambda m: m.group(0).upper(), t)  # C.k. -> C.K.
    return t


def first_sentences(text: str | None, k: int = 2, max_chars: int = 330) -> str:
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sents[:k])
    return out if len(out) <= max_chars else out[:max_chars].rsplit(" ", 1)[0] + " …"


# ---------------------------------------------------------------- Alta Comedia
def member_text(folder: str, t0: float | None, t1: float | None) -> list[dict]:
    j = json.loads((VIDEOS / folder / "transcript.json").read_text(encoding="utf-8"))
    words = []
    for seg in j["segments"]:
        dur = seg["end"] - seg["start"]
        if dur > 30 and len(seg["text"].split()) / dur < 0.3:  # Whisper hallucination over silence/music
            continue
        if re.search(r"stand ?-?up comedy en espa|comediante de stand", seg["text"], re.IGNORECASE):
            continue  # Whisper repeating its own prompt
        for w in seg.get("words", []):
            if (t0 is None or w["start"] >= t0) and (t1 is None or w["start"] < t1):
                words.append({"t": w["start"], "w": w["word"]})
    return words


def build_members(data, names, aud, sala_df, secos, arco, ref_ids, rng) -> list[dict]:
    com = pd.read_parquet(DF / "df_audio_comediantes.parquet")
    com["comediante"] = com["comediante"].map(fix_mojibake)
    com["show"] = com["show"].map(fix_mojibake)
    quiet = pd.read_parquet(DF / "df_audio_quietud.parquet")
    texts, meas = {}, {}
    for n, parts in MEMBERS.items():
        texts[n] = []
        for show, id_, folder, t0, t1 in parts:
            wl = member_text(folder, t0, t1)
            raw = " ".join(x["w"].strip() for x in wl)
            mins = (wl[-1]["t"] - wl[0]["t"]) / 60 if wl else np.nan
            texts[n].append({"title": show, "id": id_, "words_list": wl, "raw": raw, "minutes": mins,
                             "t0": t0, "t1": t1, "m": measure_special(raw, raw, "es"),
                             "words": len(_words(raw)), "runtime": mins, "year": np.nan, "source": "propio"})
        meas[n] = comedian_metrics(texts[n])
    cnt = {n: Counter(t for s in texts[n] for t in _words(s["raw"]) if len(t) > 2 and t not in _ES_STOP and t not in ES_FILLERS)
           for n in MEMBERS}
    prior = sum(cnt.values(), Counter())
    out = []
    for n in MEMBERS:
        d, specs = meas[n], texts[n]
        style_rows = []
        for key, label, unit, dec, src in STYLE:
            v = d["style"][key]
            oth = [data[x]["style"][key] for x in names]
            p = percentile_of(v, oth)
            ci = ci_for(key, d, specs, oth, rng) if np.isfinite(v) else None
            rank = sorted([meas[x]["style"][key] for x in MEMBERS], reverse=True).index(v) + 1 if np.isfinite(v) else None
            style_rows.append({"key": key, "label": label, "unidad": unit, "valor": f(v, dec), "pct": pct(p),
                               "ci_lo": pct(ci[0]) if ci else "—", "ci_hi": pct(ci[1]) if ci else "—",
                               "pct_decada": "—", "rank": str(rank) if rank else "—",
                               "_pct": p if np.isfinite(v) else None, "_ci": ci,
                               "x": p if np.isfinite(p) else None, "x_lo": ci[0] if ci else None, "x_hi": ci[1] if ci else None})
        topic_rows = []
        for t in THEMES:
            v = d["topics"][t["id"]]
            p = percentile_of(v, [data[x]["topics"][t["id"]] for x in names])
            topic_rows.append({"id": t["id"], "label": t["label"], "unidad": TOPIC_UNIT, "valor": f(v, 1), "pct": pct(p), "ci_lo": "—",
                               "ci_hi": "—", "_pct": p, "x": p, "x_lo": None, "x_hi": None})
        rest = sum((cnt[x] for x in MEMBERS if x != n), Counter())
        fw = fightin_words(cnt[n], rest, prior, alpha0=500, min_count=3)
        fw = fw[fw["z"] > 1.96].head(10)
        words = [{"palabra": r.palabra, "z": f(r.z, 1), "n": str(int(r.n_a))} for r in fw.itertuples()]
        quotes = []
        for wd in [x["palabra"] for x in words[:3]]:
            q = member_quote(specs, wd)
            if q:
                quotes.append({"tipo": f"Palabra distintiva «{wd}»", **q})
            if len(quotes) == 2:
                break
        cb = next(((s, e) for s in specs for e in s["m"]["callbacks"]["ejemplos"] if not GREETING.search(e["frase"])), None)
        if cb:
            s, e = cb
            quotes.append({"tipo": "Callback", "texto": f"«{e['frase']}»",
                           "donde": f"{s['title']}: aparece al {int(round(100 * e['pos1']))} % y vuelve al {int(round(100 * e['pos2']))} % de su texto"})
        # audio: every format where the member was measured
        fmt_rows = []
        for r in com[com["comediante"] == n].itertuples():
            fmt_rows.append({"show": r.show, "tipo": r.tipo, "minutos": f(r.minutos, 1),
                             "reacciones_min": f(r.reacciones_min, 2),
                             "indice": f(r.indice_show, 2) if pd.notna(r.indice_show) else "—"})
        salas = []
        for s in specs:
            salas.append(member_sala(s, sala_df, secos, n))
        q = quiet[quiet["id"] == "alta-comedia-ansioso-2025-08-22"] if n == "Andrés Loaiza" else quiet.iloc[0:0]
        quiet_rows = []
        if n == "Andrés Loaiza":
            for r in quiet.itertuples():
                quiet_rows.append({"show": "Ansioso" if "ansioso" in r.id else "Set en La Korte",
                                   "n": str(int(r.reacciones_fuertes)), "detectado": f(r.detectado_pct, 1),
                                   "caminar": f(r.caminar_proporcion, 2), "caminar_p": f(r.caminar_p_mas_quieto, 3),
                                   "gesticular": f(r.gesticular_proporcion, 2), "gesticular_p": f(r.gesticular_p_mas_quieto, 3)})
            # La Korte set has audio but no transcript: its room metrics still belong on the page
            lk = sala_df[sala_df["id"] == "andres-loaiza-0731"].iloc[0]
            sec = secos[secos["id"] == "andres-loaiza-0731"]
            s_ = sala.serie(load_frames("andres-loaiza-0731"))
            mem, p = memory_permutation_p(s_)
            salas.append({"show": "Set en La Korte (31 jul 2025)", "memoria": f(mem, 2), "memoria_p": f(p, 3),
                          "ventanas": str(len(s_)), "secos_n": str(len(sec)), "secos_hora": f(lk.tramos_secos_hora, 1),
                          "cadenas": [{"desde": f(c["desde_min"], 1), "hasta": f(c["hasta_min"], 1),
                                       "tramos": str(c["tramos"]), "seco_s": str(c["seco_s"])} for c in dry_chains(sec)],
                          "recupera_s": f(lk.recupera_mediana_s, 0), "nota": "audio, sin transcripción"})
        # Whisper punctuation: sentence length and questions are not comparable with human transcripts
        ls = lessons(style_rows, topic_rows, len(specs), skip={"largo_frase", "preguntas_1000"})
        foto = FOTO_MEMBER.get(n)
        out.append({
            "name": n, "slug": slug(n), "n": str(len(specs)), "n1": len(specs) == 1,
            "textos": [{"title": s["title"], "minutos": f(s["minutes"], 1), "palabras": str(s["words"])} for s in specs],
            "palabras_total": str(sum(s["words"] for s in specs)), "corto": sum(s["words"] for s in specs) < BLOCK,
            "photo": {"file": str((FOTOS_AC / foto).as_posix()), "license": "Alta Comedia", "artist": "", "source": ""} if foto else None,
            "style": [{k: v for k, v in r.items() if not k.startswith("_")} for r in style_rows],
            "topics": [{k: v for k, v in r.items() if not k.startswith("_")} for r in topic_rows],
            "words": words, "quotes": quotes[:3], "lessons": ls,
            "formatos": fmt_rows, "salas": salas, "quietud": quiet_rows,
        })
    return out


def member_quote(specs: list[dict], word: str) -> dict | None:
    for s in specs:
        wl = s["words_list"]
        sent, start = [], None
        for x in wl:
            if not sent:
                start = x["t"]
            sent.append(x["w"].strip())
            if re.search(r"[.!?…]$", x["w"].strip()) or len(sent) >= 40:
                txt = " ".join(sent)
                if re.search(rf"\b{re.escape(word)}\b", txt.lower()) and len(sent) >= 6:
                    base = s["t0"] or 0
                    return {"texto": trim(txt, word), "donde": f"{s['title']}, minuto {int((start - base) // 60)} de su set"
                            if s["t0"] is not None else f"{s['title']}, minuto {int(start // 60)}"}
                sent = []
    return None


def member_sala(s: dict, sala_df, secos, n) -> dict:
    """Room metrics on the member's own window: the 15 s series is relative to the whole night's average,
    so a member's stretch is comparable to their castmates' on the same recording."""
    c = load_frames(s["id"])
    t0, _ = M.ventana(c["t"], c["habla"])
    ser = sala.serie(c)
    if s["t0"] is not None:
        a, b = int(max(s["t0"] - t0, 0) // sala.VENTANA_S), int(max(s["t1"] - t0, 0) // sala.VENTANA_S)
        ser = ser[a:b]
    mem, p = memory_permutation_p(ser)
    dry = sala.tramos_secos(ser)
    sec = pd.DataFrame([{"inicio_min": i * sala.VENTANA_S / 60, "duracion_s": (j - i) * sala.VENTANA_S} for i, j in dry],
                       columns=["inicio_min", "duracion_s"])
    rec = [sala.recuperacion_s(ser, j) for _, j in dry]
    rec = [x for x in rec if x is not None]
    minutes = len(ser) * sala.VENTANA_S / 60
    return {"show": s["title"], "memoria": f(mem, 2), "memoria_p": f(p, 3), "ventanas": str(len(ser)),
            "secos_n": str(len(dry)), "secos_hora": f(60 * len(dry) / minutes if minutes else np.nan, 1),
            "cadenas": [{"desde": f(c_["desde_min"], 1), "hasta": f(c_["hasta_min"], 1), "tramos": str(c_["tramos"]),
                         "seco_s": str(c_["seco_s"])} for c_ in dry_chains(sec)],
            "recupera_s": f(float(np.median(rec)), 0) if rec else "—",
            "nota": "minutos contados desde el inicio de su set" if s["t0"] is not None else ""}


# ---------------------------------------------------------------- annex and map
def build_annex(aud, sala_df, secos, names, ref_ids) -> list[dict]:
    only = aud[(aud["grupo"] == "referente") & (~aud["comediante"].isin(names))]
    ref = aud[aud["id"].isin(ref_ids)]
    out = []
    for r in only.sort_values(["comediante", "anio"]).itertuples():
        s = sala.serie(load_frames(r.id))
        mem, p = memory_permutation_p(s)
        sr = sala_df[sala_df["id"] == r.id].iloc[0]
        out.append({"comediante": r.comediante, "especial": r.especial, "anio": str(int(r.anio)) if pd.notna(r.anio) else "",
                    "minutos": f(r.minutos, 0), "reacciones_min": f(r.reacciones_min, 2),
                    "reacciones_pct": pct(percentile_of(r.reacciones_min, ref["reacciones_min"])),
                    "apertura": f(r.apertura, 2), "cierre": f(r.cierre, 2),
                    "secos_hora": f(sr.tramos_secos_hora, 1), "cadenas": str(len(dry_chains(secos[secos["id"] == r.id]))),
                    "memoria": f(mem, 2), "memoria_p": f(p, 3)})
    return out


def build_map(comedians, data, names) -> dict:
    """2-D map of style: PCA of the comedians' style percentiles (metrics present for everyone)."""
    keys = ["ttr", "largo_frase", "yo_vs_tu", "preguntas_1000", "groserias_1000", "callbacks_10k"]
    X = np.array([[percentile_of(data[n]["style"][k], [data[x]["style"][k] for x in names if x != n]) for k in keys]
                  for n in names])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    var = S ** 2 / (S ** 2).sum()
    load = {k: (Vt[0][i], Vt[1][i]) for i, k in enumerate(keys)}
    lab = {k: l for k, l, *_ in STYLE}

    def axis(j):
        hi = sorted(keys, key=lambda k: -load[k][j])[:2]
        lo = sorted(keys, key=lambda k: load[k][j])[:2]
        return {"mas": " y ".join(lab[k].lower() for k in hi), "menos": " y ".join(lab[k].lower() for k in lo)}

    W, H, pad = 100, 92, 6  # SVG viewBox of the map in the template
    sx = lambda v: pad + (v - P[:, 0].min()) / (np.ptp(P[:, 0]) or 1) * (W - 2 * pad)
    sy = lambda v: H - pad - (v - P[:, 1].min()) / (np.ptp(P[:, 1]) or 1) * (H - 2 * pad)
    pts = []
    for i, n in enumerate(names):
        x, y = float(sx(P[i, 0])), float(sy(P[i, 1]))
        right = x > W - 18
        pts.append({"name": n, "label": n.split(" ")[-1], "slug": comedians[i]["slug"], "sx": x, "sy": y,
                    "lx": x - 1.5 if right else x + 1.5, "ly": y, "anchor": "end" if right else "start"})
    placed = []  # greedy: push a label down until it no longer overlaps one already placed
    for p in sorted(pts, key=lambda p: p["sy"]):
        w = 1.25 * len(p["label"])
        x0 = p["lx"] - (w if p["anchor"] == "end" else 0)
        while any(abs(p["ly"] - q[1]) < 2.6 and x0 < q[0] + q[2] and q[0] < x0 + w for q in placed):
            p["ly"] += 2.6
        placed.append((x0, p["ly"], w))
    return {"puntos": pts, "var1": pct(100 * var[0]), "var2": pct(100 * var[1]), "eje_x": axis(0), "eje_y": axis(1),
            "zx": float(sx(0)), "zy": float(sy(0))}


if __name__ == "__main__":
    main()
