# Improvements — TextMining (Stand-up Comedy)

## Contexto
82 especiales en inglés. Dashboard Streamlit 9 tabs. Ridge TF-IDF para predicción de rating. IMDb scraping roto (HTTP 202). Ruta local: `D:\ANDRES\Claude Projects\TextMining`.

---

## Mejoras por prioridad

### 🔴 Alta prioridad

#### 1. OMDB API para ampliar dataset (desbloquea todo)
**Problema:** el scraper de IMDb está roto. Nuevos shows requieren curación manual de ID + rating.  
**Solución:** OMDB API es gratuita (1000 req/día) y devuelve el mismo dato sin bot detection.

```python
# transcripts/fetch_imdb_omdb.py
import requests, os

OMDB_KEY = os.getenv('OMDB_API_KEY')  # registrar en omdbapi.com gratis

def fetch_imdb_details(imdb_id: str) -> dict:
    r = requests.get('https://www.omdbapi.com/', params={
        'i': imdb_id, 'apikey': OMDB_KEY
    })
    d = r.json()
    return {
        'imdb_id': d['imdbID'],
        'title': d['Title'],
        'year': int(d['Year'][:4]),
        'runtime': int(d.get('Runtime', '0 min').split()[0]),
        'imdb_rating': float(d['imdbRating']) if d['imdbRating'] != 'N/A' else None,
        'imdb_votes': int(d.get('imdbVotes', '0').replace(',', '')),
    }
```

Con esto desbloqueado, ampliar de 82 → 200+ especiales usando `transcripts/fetch_new_specials.py` + la lista en `data/list_id/`.

---

#### 2. Correlación TTR vs rating (pendiente documentado)
**Problema:** la correlación TTR–rating nunca se calculó.

```python
# analysis/ttr_correlation.py
from scipy import stats
import pandas as pd
from analysis.core import load_unified, compute_ttr

df = load_unified()
df['ttr'] = df['transcript_clean'].apply(compute_ttr)
valid = df.dropna(subset=['ttr', 'imdb_rating'])

r, p = stats.pearsonr(valid['ttr'], valid['imdb_rating'])
print(f"TTR vs Rating: r={r:.3f}, p={p:.4f}, n={len(valid)}")

# Scatter plot con plotly: x=TTR, y=rating, hover=comedian+title
```

Agregar resultado a `hallazgos-clave.md` con el número real.

---

#### 3. Análisis de tópicos vs rating (pendiente documentado)
8 temas ya están en `topic_scores.parquet`. Falta calcular correlación de cada tema con el rating.

```python
# analysis/topic_rating.py
topics = pd.read_parquet('data/topic_scores.parquet')
ratings = load_unified()[['show_idx', 'imdb_rating']]
merged = topics.merge(ratings, on='show_idx')

results = []
for topic in merged['topic_id'].unique():
    sub = merged[merged['topic_id'] == topic]
    r, p = stats.pearsonr(sub['score'], sub['imdb_rating'])
    results.append({'topic': topic, 'r': r, 'p': p, 'n': len(sub)})

# Bar chart: temas ordenados por correlación con rating
# Responde: ¿qué temas predicen mejor rating?
```

---

### 🟡 Media prioridad

#### 4. Análisis del propio material (aplicación más valiosa)
**Pipeline:** usar Whisper de `Analisis_videos` para transcribir shows propios → agregar al dataset.

```python
# transcripts/add_own_material.py
# Input: transcript.txt de scripts/01_transcribe.py de Analisis_videos
import pandas as pd

def add_own_show(transcript_path, show_name, comedian='Andrés Loaiza', year=2026):
    df = pd.read_parquet('data/df_unified.parquet')
    transcript = open(transcript_path, encoding='utf-8').read()
    
    new_row = {
        'show_idx': df['show_idx'].max() + 1,
        'comedian': comedian,
        'title': show_name,
        'year': year,
        'imdb_rating': None,  # sin rating, solo análisis descriptivo
        'transcript': transcript,
        'own_material': True,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_parquet('data/df_unified.parquet', index=False)
    print(f"✓ Agregado '{show_name}'. Correr: python analysis/precompute.py")
```

En el dashboard: filtro "Solo mi material" + tab "Mi Perfil" que compara NRC, TTR y tópicos de Andrés vs top 10 comediantes de rating ≥8.

---

#### 5. Fix IMDb con Playwright (alternativa a OMDB)
Para scraperear shows que OMDB no tiene bien catalogados.

```python
# transcripts/fetch_imdb_playwright.py
from playwright.async_api import async_playwright

async def fetch_imdb_details(imdb_id: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_extra_http_headers({'User-Agent': 'Mozilla/5.0 ...'})
        await page.goto(f'https://www.imdb.com/title/{imdb_id}/')
        rating_el = page.locator('[data-testid="hero-rating-bar__aggregate-rating__score"]')
        text = await rating_el.text_content()
        return float(text.split('/')[0])
```

---

#### 6. Análisis temporal (¿cambia la comedia con el tiempo?)
¿Los especiales post-2020 tienen vocabulario/emociones diferentes a los pre-2015?

```python
# En análisis existente, agregar columna:
df['era'] = pd.cut(df['year'],
    bins=[0, 2010, 2015, 2020, 2030],
    labels=['pre-2010', '2010-15', '2015-20', '2020+'])

# Heatmap NRC por era
# TTR promedio por era  
# ¿La comedia se volvió más oscura o más positiva?
```

---

#### 7. Despliegue en Streamlit Cloud
Para compartir hallazgos con el equipo o publicarlos sin necesidad de correr el dashboard localmente.

```bash
# requirements.txt ya está
# Agregar a GitHub (solo parquets sin PII, o datos de muestra)
# streamlit.io → Deploy from GitHub
```

Alternativa: exportar las gráficas clave como HTML con Plotly y hostearlas en GitHub Pages.

---

### 🟢 Baja prioridad

#### 8. Publicación de hallazgos en LinkedIn
Los 5 hallazgos más sorprendentes como post de LinkedIn o artículo de blog:
> "Analicé 82 especiales de stand-up con Python. Esto predice un buen rating."

Formato: hallazgo → gráfica → implicación para escribir material.

#### 9. Sentence embeddings para clustering
Con la GPU disponible (GTX 1660 SUPER), probar `sentence-transformers` en vez de TF-IDF:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # ligero, funciona sin GPU también
embeddings = model.encode(df['transcript'].tolist(), show_progress_bar=True)
```

No re-introducir LDA (descartado — ver `metodologia.md`).

#### 10. Monitor de nuevos especiales
Script que busca en IMDb (via OMDB) especiales del año actual → detecta cuáles no están en el dataset → lista para agregar.

---

## Rutinas remotas (agents agendados)

| Routine | Schedule | Descripción |
|---------|----------|-------------|
| **Monitor nuevos especiales** | Mensual | Busca especiales año actual en OMDB → lista candidatos para agregar al dataset |
| **Análisis material propio** | Trigger manual | Recibe transcript de show de Andrés → corre análisis → reporte "Tu perfil vs tus referentes" |

## Notas técnicas
- Correr `python analysis/precompute.py` después de cualquier cambio en parquets fuente o lexicones
- Función `clean_for_analysis()` preserva `transcript_raw` para `laughter_triggers`
- IMDb scraping roto → usar OMDB API o entrada manual como solución actual
- `python -m streamlit run dashboard/app.py` para el dashboard
