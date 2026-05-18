# stand_up_research

Pipeline de **text mining sobre stand-up comedy**: cruza transcripciones de
especiales con su metadata y rating de IMDb para responder preguntas como
*¿qué temas y estilos predicen un mejor rating?*, *¿cómo evoluciona el
sentimiento dentro de un show?*, *¿qué tan distinto es el vocabulario entre
comediantes?*

## Datos

| Archivo | Filas | Qué tiene |
|---|---|---|
| `data/data_frame/raw_transcripts.parquet` | 369 | `title`, `comedian`, `transcript` (scraping de [scrapsfromtheloft.com](https://scrapsfromtheloft.com/stand-up-comedy-scripts/)) |
| `data/data_frame/df_imdb.parquet` | 135 | metadata IMDb: año, rating, votos, runtime, director, distribuidor, plot, demographics |
| `data/data_frame/df_unified.parquet` | 138 | merge por título: transcript + metadata IMDb (el dataset principal para análisis) |

Listas de curación manual en `data/list_id/`:
- `imdb_id_to_delete.txt` — IDs que el search devuelve pero no son stand-up.
- `imdb_id_to_add.txt` — stand-ups que faltan en el search.

## Estructura

```
stand_up_research/
├── analysis/                # módulo Python reusable (notebook + dashboard)
│   ├── __init__.py
│   └── core.py
├── dashboard/
│   └── app.py               # dashboard interactivo Streamlit
├── notebooks/
│   └── analysis.ipynb       # análisis end-to-end con filtros
├── imdb/extraction_IMDb.py  # scraping IMDb + cinemagoer
├── transcripts/transcript.py  # scraping de transcripts
├── data/
│   ├── list_id/             # curaduría manual de IDs IMDb
│   └── data_frame/          # parquets (salida de los pipelines)
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

Primera vez (corpus NLTK):

```bash
python -c "import ssl; ssl._create_default_https_context=ssl._create_unverified_context; import nltk; [nltk.download(p) for p in ['stopwords','punkt','punkt_tab']]"
```

> El truco de `ssl._create_unverified_context` es por un problema común
> de certificados en Windows. Si tu Python ya tiene certifi configurado,
> basta con `nltk.download(...)`.

## Uso

### Dashboard interactivo (recomendado)

```bash
streamlit run dashboard/app.py
```

Abre el navegador con un dashboard con sidebar de filtros (comediantes,
año, rating, votos) y 9 pestañas:

- **Overview** — métricas y distribuciones del subset.
- **Vocabulario** — top n-gramas y wordcloud.
- **Sentimiento** — VADER por show, por comediante, vs rating.
- **Emociones (NRC)** — 8 emociones de Plutchik por comediante (heatmap).
- **Tópicos (LDA)** — temas latentes + composición por show.
- **Catchphrases** — n-gramas distintivos por comediante (TF-IDF).
- **Clustering** — comediantes en 2D (UMAP) coloreados por k-means.
- **Predicción rating** — Ridge sobre TF-IDF: qué palabras suben/bajan el rating.
- **Datos** — tabla del subset con descarga CSV.

Cada visualización trae una **descripción auto-generada** que se actualiza
con el filtro (no es texto fijo: lee los resultados y los interpreta).

### Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Mismo análisis end-to-end (35 celdas) parametrizado por filtros editables
en una celda al principio.

### (Opcional) Re-extraer datos

```bash
python imdb/extraction_IMDb.py        # refresca df_imdb.parquet
python transcripts/transcript.py      # refresca raw_transcripts.parquet
```

> Los selectores CSS del scraper son frágiles (clases tipo
> `elementor-element-74af9a5b`). Si la web cambia, hay que actualizar
> `transcripts/transcript.py`. Los parquets versionados sirven como
> snapshot si el scraping falla.

## Catálogo de análisis (en orden de profundidad)

| Análisis | Técnica | Implementación |
|---|---|---|
| EDA | histogramas, ranking | `analysis/core.py::load_unified` |
| Riqueza léxica | TTR (type-token ratio) | inline |
| N-gramas frecuentes | `CountVectorizer` | `top_ngrams` |
| WordCloud | `wordcloud` | inline |
| Sentimiento (polaridad) | VADER | `sentiment_compound` |
| Emociones discretas | NRC lexicon (NRCLex) | `emotion_profile` |
| Tópicos latentes | LDA (sklearn) | `extract_topics` |
| Catchphrases | TF-IDF n-gramas | `catchphrases_by_comedian` |
| Clustering estilístico | k-means + UMAP | `cluster_comedians` |
| ¿Qué predice el rating? | Ridge interpretable | `predict_rating` |

## Limitaciones conocidas

- La columna `comedian` del scraping a veces trae `"Stand-up transcripts"`
  (etiqueta de la web, no el comediante); el módulo `analysis` hace un
  fallback parseando el título.
- Solo shows en inglés (filtro IMDb `languages=en`).
- IMDb scraping puede romperse si IMDb cambia su HTML; preferir
  `cinemagoer` por ID cuando sea posible.
- El modelo predictivo del rating tiene n pequeño (~130) → los coeficientes
  son **sugerentes, no causales**.
