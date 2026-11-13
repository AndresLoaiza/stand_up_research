# CLAUDE.md

Text mining on stand-up comedy transcripts cross-referenced with IMDb
metadata. The headline question the project is built around:
*¿qué del lenguaje, sentimiento y temas correlaciona con el rating IMDb
de un especial?*

Repo: [AndresLoaiza/stand_up_research](https://github.com/AndresLoaiza/stand_up_research).

## Layout

```
analysis/                  reusable text-mining helpers (notebook + dashboard share this)
  core.py                  load_unified, cleaning, NRC, topics, predict_rating, ...
  compute_curated_topics.py  8 curated themes scored by lexicon match
  precompute.py            writes df_enriched.parquet + emotion_words.parquet + topic_scores.parquet
dashboard/app.py           Streamlit app (9 tabs)
notebooks/analysis.ipynb   end-to-end notebook
transcripts/
  transcript.py            original scraper for scrapsfromtheloft.com
  fetch_new_specials.py    one-shot adder with curated metadata
  backfill_missing.py      slug-matches existing IMDb shows against the full index
imdb/extraction_IMDb.py    IMDb metadata scraper (status: HTML changed, partly broken)
data/
  data_frame/              all parquets live here
  list_id/                 manual curation of IMDb IDs to add/delete
```

## Dataflow (read this before touching parquets)

There are three "source of truth" parquets and three derived ones, all in `data/data_frame/`:

| Parquet | Role |
|---|---|
| `raw_transcripts.parquet` | scraped transcripts, may include shows we don't have IMDb info for |
| `df_imdb.parquet` | IMDb metadata, may include shows with no transcript |
| `df_unified.parquet` | merge of the two by title — **the dataset for analysis** |
| `df_enriched.parquet` | df_unified + sentiment + ttr + 8 `nrc_<emotion>` columns (derived) |
| `emotion_words.parquet` | long-form [show_idx, emotion, word, count] for the "top words per emotion" panel (derived) |
| `topic_scores.parquet` | long-form [show_idx, topic_id, score] for 8 curated themes (derived) |
| `curated_topics.json` | static metadata: theme labels, descriptions, seed lexicons |

**Rule**: after editing any source parquet or cleaning/lexicon code, run
`python analysis/precompute.py` to regenerate the derived parquets.
`load_unified()` automatically prefers the enriched parquet when it
exists, which keeps dashboard tabs under 1s instead of 25s.

## Commands

```bash
# Dashboard (the primary UI)
python -m streamlit run dashboard/app.py

# Refresh derived parquets after data or cleaning changes
python analysis/precompute.py

# Notebook
jupyter notebook notebooks/analysis.ipynb

# Add new specials (one-shot, edits the catalog at top of the script)
python transcripts/fetch_new_specials.py

# Backfill transcripts for IMDb-known shows that don't have one yet
python transcripts/backfill_missing.py
```

NLTK on Windows often needs the SSL workaround:

```bash
python -c "import ssl; ssl._create_default_https_context=ssl._create_unverified_context; import nltk; [nltk.download(p) for p in ['stopwords','punkt','punkt_tab','wordnet']]"
```

## Audio: la risa medida en el video _(nov 2026)_

`audio/` mide la reacción del público escuchando los videos de `D:\ANDRES\Stand up` (79 especiales
de referencia, sin duplicados) y los shows propios de Alta Comedia que ya tienen audio en
`Analisis_videos/output/`. Se cruza con el texto por `imdbID` (28 especiales tienen las dos fuentes).

```bash
python audio/catalogo.py                     # decide qué videos entran (duplicados, muestras, compilaciones fuera)
<venv de Analisis_videos> audio/extraer_risas.py   # YAMNet necesita mediapipe; ~1 h para todo
python audio/metricas.py                     # -> df_audio, df_audio_arco, df_audio_sensibilidad
python audio/validar.py                      # sensibilidad contra subtítulos para sordos
python audio/hallazgos.py                    # todos los números del informe
python audio/segmentos.py                    # por comediante dentro de los shows propios con turnos conocidos
python audio/test_metricas.py                # pruebas con datos sintéticos
```

Dashboard: pestaña "🔊 Risa medida (audio)" (`dashboard/audio_tab.py`). Responde al filtro de
comediantes y años de la barra lateral (no al de rating ni votos: casi ningún especial con audio los
tiene). La lista de comediantes incluye a los que solo tienen audio y a los integrantes de Alta
Comedia; si el filtro deja sin transcripciones, el dashboard muestra solo la parte de audio.

**Por comediante** (`audio/segmentos.py` -> `df_audio_comediantes.parquet`): solo en *Hoy Nace un
Chiste* (4 sets, cortes verificados con fotogramas) y *Comedia de Alto Riesgo* (turnos de
`Analisis_videos/.../juegos.json`). Dentro de un show todos comparten grabación, así que el nivel
sí se compara; el índice del show (1 = promedio de esa noche) es la medida justa. NAPE y Fuera de
Charla no tienen turnos: la identificación por voz de `Analisis_videos/23_speakers.py` no funcionó.
También entran los shows completos de una sola persona (Ansioso y el set de La Korte), para ver a cada
integrante en todos sus formatos; ahí el índice no aplica. La sección "La curva de un show" compara
cualquier show (por defecto Ansioso) contra la franja central de los especiales y lista los décimos flojos.

Lo que hay que saber antes de concluir algo:
- **Se mide "reacción" (risa + aplauso), umbral 0,05.** En mezclas de TV la voz del comediante
  domina y YAMNet le da a la risa probabilidades bajas o la llama "aplauso". El umbral de
  `Analisis_videos/02b` (0,25, solo risa) encontró el 5 % de las risas marcadas en subtítulos
  para sordos; este, el 76 % (mediana de los especiales con subtítulos con hora).
- **El nivel (reacciones por minuto) depende de la grabación.** `hallazgos.py` mide qué parte de
  la varianza de cada métrica ocurre dentro de un mismo comediante: si es casi toda, la métrica
  dice más de la mezcla de sonido que del comediante. Preferir las métricas relativas a cada
  show (forma por décimos, concentración, posición de los picos).
- Se guardan los cuadros (`data/audio/frames/*.npz`, ~50 KB cada uno, en el repo) para poder
  recalcular con otro umbral sin volver a escuchar todo.
- Los shows propios son grabaciones de celular o cámara en sala, en español y en formatos
  distintos (panel, juegos, dúo): comparables entre sí, no de igual a igual con un especial de TV.

## Conventions and gotchas

**Transcript cleaning** (in `analysis.core.clean_for_analysis`) is
applied during `load_unified()`. It:
- Normalizes curly apostrophes (`’` `‘` `` ` `` `´`) to straight `'`
  so contraction patterns match.
- Strips `[laughter]`, `[applause]`, `(crowd laughs)`, etc.
- Strips repeated-name chants (`Louis! Louis! Louis!`).
- Strips `ha ha` / `haha`.
- Expands contractions (`don't` → `do not`, `you're` → `you are`)
  so the tokenizer doesn't produce stray `don` / `re` tokens that
  triggered spurious NRC matches.
- Keeps the raw text in `df.transcript_raw` so `laughter_triggers`
  can still see the markers.

**Comedian deduplication** is done in `load_unified()` via three steps:
title-based fallback for blank/junk values, normalize ALLCAPS to
Title Case, a manual `COMEDIAN_OVERRIDES` map for specials whose
title doesn't reveal the comedian (e.g. `THE AGE OF SPIN` → Dave
Chappelle), and finally a slug-collapse pass that merges variants like
`Louis C.K.` and `Louis C.K. 2017`.

**Dataset philosophy**: the dataset only contains shows we *actually
have transcripts for* (currently 82). If a backfill run can't find a
transcript, the row is dropped from `df_unified.parquet` and
`df_imdb.parquet`. This is intentional — partial rows pollute every
analysis.

**Curated topics, not LDA**: an early version had a dynamic LDA tab.
On this corpus (~80 small, profanity-heavy docs) LDA produced messy
topics dominated by curse words and foreign-language bits. Replaced
with 8 hand-curated themes scored by stem-prefix lexical matching
(`analysis/compute_curated_topics.py`). To add or rebalance themes,
edit `THEMES` in that file and re-run `precompute.py`. **Do not
re-introduce dynamic LDA without strong reason.**

**Catchphrases and Clustering UI removed**: the user found them not
useful for their workflow. The underlying functions
(`catchphrases_by_comedian`, `cluster_comedians`) still exist in
`analysis/core.py` and the notebook for any future revival, but
**do not add them back to the dashboard without asking**.

**IMDb scraping is broken**: `imdb.com/search/title/?...` returns
HTTP 202 with an empty body for bot detection, and `cinemagoer`
returns `None` because IMDb changed their HTML. For new shows we
get the imdbID via WebSearch and provide rating/year manually in
`fetch_new_specials.py`. Do not waste time trying to scrape IMDb
search live.

## Code style

- This is a personal research project; prefer terse, readable code
  over heavy abstraction.
- New analysis functions should live in `analysis/core.py` and be
  exposed in `analysis/__init__.py` so both notebook and dashboard
  can import them.
- Streamlit caches: use `@st.cache_data` for dataframe-returning
  helpers and `@st.cache_resource` for static lookup tables.
- Spanish is the project's working language for UI strings,
  commit messages and explanations. Python code (identifiers,
  comments inside functions) is English.

## Things the user has said NOT to do

- Don't re-add the Catchphrases or Clustering tabs to the dashboard
  unless asked.
- Don't use dynamic LDA — the curated-topics approach is the
  agreed solution.
- The `comedian` field from scraping is unreliable (`"Stand-up
  transcripts"` etc.); the fallback parser in `load_unified` is the
  source of truth.
