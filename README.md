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
├── imdb/extraction_IMDb.py       # scraping IMDb + cinemagoer
├── transcripts/transcript.py     # scraping de transcripts
├── notebooks/
│   └── analysis.ipynb            # EDA + minería de texto + sentimiento + cruce IMDb
├── data/
│   ├── list_id/                  # curaduría manual de IDs IMDb
│   └── data_frame/               # parquets (salida de los pipelines)
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

Primera vez (descarga corpus NLTK que usa el notebook):

```bash
python -c "import nltk; [nltk.download(p) for p in ['stopwords','punkt','punkt_tab']]"
```

## Uso

### 1. (Opcional) Re-extraer datos

```bash
python imdb/extraction_IMDb.py        # refresca df_imdb.parquet
python transcripts/transcript.py      # refresca raw_transcripts.parquet
```

> Los selectores CSS del scraper son frágiles (clases tipo
> `elementor-element-74af9a5b`). Si la web cambia, hay que actualizar
> `transcripts/transcript.py`. Los parquets versionados sirven como
> snapshot si el scraping falla.

### 2. Analizar

```bash
jupyter notebook notebooks/analysis.ipynb
```

El notebook está parametrizado por filtros (comediante, rango de años,
rating mínimo, top-N por votos) y todas las visualizaciones se recalculan
sobre el subset filtrado.

## Qué se puede analizar

Implementado en `notebooks/analysis.ipynb`:

- **EDA**: distribución de longitudes, riqueza léxica, palabras/minuto.
- **N-gramas**: top unigramas / bigramas / trigramas con stopwords filtradas.
- **WordCloud**: global y por filtro.
- **Sentimiento**: VADER por show + evolución intra-show (segmentado).
- **TF-IDF**: palabras distintivas por comediante.
- **Similaridad estilística**: matriz coseno + heatmap entre comediantes.
- **Cadenas de Markov**: generador de frases al estilo del comediante.
- **Cruce con IMDb**: rating vs sentimiento, vs diversidad léxica, vs longitud.

## Limitaciones conocidas

- La columna `comedian` del scraping a veces trae `"Stand-up transcripts"`
  (etiqueta de la web, no el comediante); el notebook hace un fallback
  parseando el título.
- Solo shows en inglés (filtro IMDb `languages=en`).
- IMDb scraping puede romperse si IMDb cambia su HTML; preferir
  `cinemagoer` por ID cuando sea posible.
