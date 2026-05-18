"""
Stand-up Comedy — Text Mining Dashboard.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from analysis import (
    EMOTIONS,
    apply_filters,
    catchphrases_by_comedian,
    cluster_comedians,
    emotion_profile,
    emotion_summary,
    emotion_top_words,
    extract_topics,
    laughter_triggers,
    load_unified,
    narrate_catchphrases,
    narrate_emotions,
    narrate_filters,
    narrate_predictor,
    narrate_topics,
    predict_rating,
    sentiment_compound,
    tokenize,
    top_ngrams,
    yearly_emotion_trends,
    yearly_sentiment_rating,
    yearly_word_trends,
)

st.set_page_config(page_title="Stand-up Text Mining", layout="wide", page_icon="🎤")
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datos...")
def get_data() -> pd.DataFrame:
    df = load_unified()
    df["sentiment"] = df["transcript"].map(sentiment_compound)
    df["tokens"] = df["transcript"].map(tokenize)
    df["unique_words"] = df["tokens"].map(lambda t: len(set(t)))
    df["ttr"] = df["unique_words"] / df["tokens"].map(len).replace(0, np.nan)
    df["words_per_min"] = df["word_count"] / df["runtime_min"]
    return df


@st.cache_data(show_spinner="Calculando emociones (NRC)...")
def cached_emotions(idx_key: tuple) -> pd.DataFrame:
    sub = df_all.loc[list(idx_key)]
    return emotion_profile(sub)


@st.cache_data(show_spinner="Modelando tópicos (LDA)...")
def cached_topics(idx_key: tuple, n_topics: int) -> dict:
    sub = df_all.loc[list(idx_key)]
    return extract_topics(sub, n_topics=n_topics)


@st.cache_data(show_spinner="Detectando catchphrases...")
def cached_catchphrases(idx_key: tuple, lo: int, hi: int, top_k: int) -> dict:
    sub = df_all.loc[list(idx_key)]
    # Pass the full unfiltered df as background so single-comedian subsets
    # still return distinctive n-grams (vs everyone else in the corpus).
    return catchphrases_by_comedian(
        sub, ngram_range=(lo, hi), top_k=top_k, background=df_all
    )


@st.cache_data(show_spinner="Calculando palabras por emoción...")
def cached_emotion_words(idx_key: tuple, top_k: int) -> pd.DataFrame:
    sub = df_all.loc[list(idx_key)]
    return emotion_top_words(sub, top_k=top_k)


@st.cache_data(show_spinner="Clusterizando comediantes (UMAP)...")
def cached_clusters(idx_key: tuple, k: int) -> pd.DataFrame:
    sub = df_all.loc[list(idx_key)]
    return cluster_comedians(sub, k=k, method="umap")


@st.cache_data(show_spinner="Entrenando modelo de rating...")
def cached_predict(idx_key: tuple, alpha: float) -> dict:
    sub = df_all.loc[list(idx_key)]
    return predict_rating(sub, alpha=alpha)


df_all = get_data()


# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------
st.sidebar.title("🎤 Filtros")

all_comedians = sorted(df_all["comedian"].dropna().unique())
top_by_count = df_all["comedian"].value_counts().head(20).index.tolist()

st.sidebar.markdown("**Comediantes** (vacío = todos)")
selected = st.sidebar.multiselect(
    "Selecciona uno o varios", all_comedians, default=[],
    help="Filtro principal. Si vacío, considera a todos.",
)
if st.sidebar.button("Top 10 más populares"):
    selected = top_by_count[:10]
    st.session_state["_force_selected"] = selected

yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
year_range = st.sidebar.slider("Rango de años", yr_min, yr_max, (yr_min, yr_max))

min_rating = st.sidebar.slider("Rating IMDb mínimo", 0.0, 10.0, 0.0, 0.1)
min_votes = st.sidebar.number_input("Votos mínimos", 0, 100000, 0, 100)

# Show filter — dynamic: only shows that survive the other filters
df_pre = apply_filters(
    df_all,
    comedians=selected or None,
    year_range=year_range,
    min_rating=min_rating,
    min_votes=int(min_votes),
)
available_titles = sorted(df_pre["title"].dropna().unique())
st.sidebar.markdown(f"**Shows** ({len(available_titles)} disponibles)")
selected_titles = st.sidebar.multiselect(
    "Selecciona uno o varios", available_titles, default=[],
    help="Filtra a shows específicos dentro del subset ya filtrado por los criterios de arriba.",
)

st.sidebar.markdown("---")
st.sidebar.caption("Los filtros se aplican a todas las pestañas.")

df = apply_filters(
    df_all,
    comedians=selected or None,
    year_range=year_range,
    min_rating=min_rating,
    min_votes=int(min_votes),
    titles=selected_titles or None,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎤 Stand-up Comedy — Text Mining Dashboard")
st.markdown(narrate_filters(df_all, df))

if len(df) == 0:
    st.warning("Sin shows. Relaja los filtros.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shows", len(df))
c2.metric("Comediantes", df["comedian"].nunique())
c3.metric("Rating medio", f"{df['rating'].mean():.2f}" if df["rating"].notna().any() else "n/d")
c4.metric("Palabras totales", f"{int(df['word_count'].sum()):,}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "☁️ Vocabulario",
    "😊 Sentimiento",
    "🎭 Emociones (NRC)",
    "🧠 Tópicos (LDA)",
    "🗣️ Catchphrases",
    "🔗 Clustering",
    "⭐ Predicción rating",
    "📅 Por año",
    "🎯 Triggers de risa",
    "📋 Datos",
])

# === Overview ===========================================================
with tabs[0]:
    st.subheader("Distribuciones del subset")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="word_count", nbins=30, title="Palabras por show")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(df.dropna(subset=["rating"]), x="rating", nbins=20, title="Rating IMDb")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        yr = df.dropna(subset=["year"]).copy()
        yr["year"] = yr["year"].astype(int)
        fig = px.histogram(yr, x="year", title="Shows por año")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(df.dropna(subset=["runtime_min"]), x="runtime_min", nbins=20, title="Runtime (min)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranking por riqueza léxica (TTR)")
    lex = df.groupby("comedian").agg(
        shows=("title", "count"),
        avg_words=("word_count", "mean"),
        avg_ttr=("ttr", "mean"),
        avg_wpm=("words_per_min", "mean"),
        avg_rating=("rating", "mean"),
    ).sort_values("avg_ttr", ascending=False).head(20).round(2)
    st.dataframe(lex, use_container_width=True)
    st.caption("TTR = type-token ratio: proporción de palabras únicas. Más alto = vocabulario más variado.")

# === Vocabulario ========================================================
with tabs[1]:
    st.subheader("N-gramas más frecuentes")
    col1, col2, col3 = st.columns(3)
    corpus = df["transcript"].tolist()
    for col, n, title in [(col1, 1, "Unigramas"), (col2, 2, "Bigramas"), (col3, 3, "Trigramas")]:
        with col:
            try:
                s = top_ngrams(corpus, n=n, top_k=15).sort_values()
                fig = px.bar(s, orientation="h", title=title)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError:
                st.info(f"Sin suficientes datos para {title.lower()}.")

    st.subheader("WordCloud")
    text_blob = " ".join(" ".join(t) for t in df["tokens"])
    if text_blob.strip():
        wc = WordCloud(width=1200, height=500, background_color="white",
                       max_words=200, collocations=False).generate(text_blob)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        st.pyplot(fig)

# === Sentimiento ========================================================
with tabs[2]:
    st.subheader("VADER — sentimiento por show")
    avg = df["sentiment"].mean()
    st.metric("Sentimiento promedio (compound)", f"{avg:+.3f}",
              help="Rango [-1, 1]. >0 positivo, <0 negativo.")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="sentiment", nbins=30, title="Distribución por show")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_com = df.groupby("comedian")["sentiment"].mean().sort_values()
        if len(by_com) > 1:
            fig = px.bar(by_com.tail(20), orientation="h",
                         title="Sentimiento promedio por comediante (top 20)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentimiento vs rating")
    sub = df.dropna(subset=["rating", "sentiment"])
    if len(sub) >= 3:
        fig = px.scatter(sub, x="sentiment", y="rating", color="comedian",
                         hover_data=["title", "year"], trendline="ols",
                         title=f"r = {sub[['sentiment','rating']].corr().iloc[0,1]:.2f}")
        st.plotly_chart(fig, use_container_width=True)

# === Emociones NRC ======================================================
with tabs[3]:
    st.subheader("Perfil emocional — NRC Word-Emotion Association Lexicon")
    with st.expander("📖 ¿Qué es esto y cómo se calcula?"):
        st.markdown("""
**NRC Lexicon** es un diccionario que asocia ~14.000 palabras del inglés con
8 emociones básicas (modelo de Plutchik: *anger, anticipation, disgust, fear,
joy, sadness, surprise, trust*) más polaridad positiva/negativa. Cada palabra
puede pertenecer a varias emociones (ej. *love* → joy + positive + trust).

**Cómo se calcula el score**: tokeniza el transcript, cuenta cuántas
palabras coinciden con cada emoción, y divide entre el total de palabras
emocionales del show. El resultado es la **proporción** del léxico
emocional dedicada a cada emoción.

**Limitaciones**: no entiende contexto ni sarcasmo, las palabras pesan igual
sin importar la intensidad, y solo cuenta palabras del diccionario (slang
y profanity quedan fuera). Sirve para **comparar perfiles relativos** entre
comediantes, no como medida absoluta de "cuánto enojo siente".
""")

    idx_key = tuple(df.index.tolist())
    try:
        emo = cached_emotions(idx_key)
        st.markdown(narrate_emotions(emo))

        # --- Sección 1: resumen agregado del subset ---
        st.markdown("#### 1. Promedio del subset")
        avg = emo.groupby("emotion")["score"].mean().sort_values(ascending=False)
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(avg, title="Proporción promedio por emoción",
                         labels={"value": "Proporción", "index": "Emoción"},
                         color=avg.values, color_continuous_scale="RdYlBu_r")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            summary = emotion_summary(emo).round(4)
            st.dataframe(summary, use_container_width=True)
            st.caption("`argmax_show` = el show con mayor score en esa emoción.")

        # --- Sección 2: heatmap por comediante ---
        st.markdown("#### 2. Perfil por comediante")
        top_coms = df["comedian"].value_counts().head(15).index
        heat = emo[emo["comedian"].isin(top_coms)].groupby(
            ["comedian", "emotion"])["score"].mean().unstack()
        if not heat.empty:
            fig = px.imshow(
                heat, aspect="auto", color_continuous_scale="RdYlBu_r",
                title="Heatmap — emoción promedio por comediante (top 15 por # shows)",
                labels=dict(color="Score"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Filas más rojas en una columna = ese comediante usa más palabras "
                "de esa emoción que el resto. Útil para detectar especialistas "
                "(p.ej. comedia de miedo, de tristeza, de optimismo)."
            )

        # --- Sección 3: top palabras por emoción ---
        st.markdown("#### 3. Palabras que más contribuyen a cada emoción")
        st.caption("Tokenizamos el subset entero y contamos cuántas veces "
                   "aparece cada palabra que el NRC asocia a cada emoción.")
        top_k_words = st.slider("Top K palabras por emoción", 5, 30, 12, key="emo_topk")
        words_df = cached_emotion_words(idx_key, top_k_words)

        cols = st.columns(4)
        for i, emo_name in enumerate(EMOTIONS):
            with cols[i % 4]:
                sub = words_df[words_df["emotion"] == emo_name].head(top_k_words)
                if len(sub):
                    fig = px.bar(
                        sub.sort_values("count"), x="count", y="word",
                        orientation="h", title=emo_name.upper(),
                        labels={"count": "Apariciones", "word": ""},
                    )
                    fig.update_layout(showlegend=False, height=350,
                                      margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

        # --- Sección 4: tabla completa exportable ---
        st.markdown("#### 4. Tabla de palabras-emoción (descargable)")
        st.dataframe(words_df.round(4), use_container_width=True, height=300)
        st.download_button(
            "⬇️ Descargar CSV palabras x emoción",
            words_df.to_csv(index=False).encode("utf-8"),
            "nrc_words_by_emotion.csv",
            "text/csv",
        )

        # --- Sección 5: emoción vs rating ---
        st.markdown("#### 5. ¿Correlaciona alguna emoción con el rating IMDb?")
        sub_emo = emo.merge(
            df[["title", "rating"]].drop_duplicates("title"),
            on="title", how="left", suffixes=("", "_r"),
        )
        sub_emo = sub_emo.dropna(subset=["rating"])
        if len(sub_emo) >= 4:
            corrs = (
                sub_emo.groupby("emotion")
                .apply(lambda g: g["score"].corr(g["rating"]))
                .sort_values()
            )
            fig = px.bar(corrs, title="Correlación de Pearson: score emoción vs rating",
                         labels={"value": "r", "index": "Emoción"},
                         color=corrs.values, color_continuous_scale="RdBu", range_color=[-0.5, 0.5])
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "r positivo = más palabras de esa emoción tienden a coincidir con "
                "mayor rating IMDb. r cercano a 0 = no hay relación. Con n pequeño "
                "los valores son sugerentes, no significativos."
            )
        else:
            st.info("Necesitas más shows con rating para el cruce.")
    except ImportError:
        st.error("Falta el paquete `nrclex`. Instala: `pip install nrclex`")

# === Tópicos LDA ========================================================
with tabs[4]:
    with st.expander("📖 ¿Qué es Topic Modeling (LDA)?"):
        st.markdown("""
**LDA (Latent Dirichlet Allocation)** descubre **tópicos latentes** en un
conjunto de documentos sin saber de antemano de qué hablan. Cada tópico es
una **distribución de probabilidad sobre palabras** (las que tienden a
aparecer juntas), y cada show es una **mezcla** de esos tópicos en
distintas proporciones.

**Cómo leer los resultados**:
- La tabla muestra las 10 palabras más representativas de cada tópico.
  Un tópico es "claro" cuando esas palabras forman un tema coherente
  (p.ej. *wife, kids, family, marriage* = familia).
- La barra muestra cuántos shows tienen ese tópico como dominante.
- La tabla final ("Composición") muestra qué proporción de cada tópico
  hay en cada show.

**Limitaciones del stand-up**: las transcripciones tienen muchas
palabrotas comunes y filler ("fuck", "shit", "man") que dominan el
vocabulario y pueden producir tópicos poco interpretables. Aumenté la
lista de stopwords pero algunos tópicos aún saldrán difusos —
**necesita probar con distintos N (3–15) y filtros** para encontrar
agrupaciones claras.
""")
    n_topics = st.slider("Número de tópicos", 3, 15, 8)
    idx_key = tuple(df.index.tolist())
    tr = cached_topics(idx_key, n_topics)
    st.markdown(narrate_topics(tr, df))

    st.subheader("Palabras por tópico")
    topic_df = pd.DataFrame({f"T{i} ({tr['labels'][i]})": words for i, words in tr["topics"]})
    st.dataframe(topic_df, use_container_width=True)

    st.subheader("Shows por tópico dominante")
    dominant = tr["doc_topic"].argmax(axis=1)
    counts = pd.Series(dominant).value_counts().sort_index()
    fig = px.bar(x=[tr["labels"][i] for i in counts.index], y=counts.values,
                 labels={"x": "Tópico (etiqueta)", "y": "Shows"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Composición de tópicos por show")
    doc_df = pd.DataFrame(tr["doc_topic"], columns=[f"T{i}" for i in range(n_topics)])
    doc_df.insert(0, "title", df["title"].values)
    doc_df.insert(1, "comedian", df["comedian"].values)
    st.dataframe(doc_df.round(2), use_container_width=True)

# === Catchphrases =======================================================
with tabs[5]:
    with st.expander("📖 ¿Qué son las catchphrases?"):
        st.markdown("""
Calculamos **TF-IDF sobre n-gramas** (frases de 2, 3 o 4 palabras
consecutivas). Por cada comediante:

- **TF** (Term Frequency) = qué tanto usa esa frase en sus shows.
- **IDF** (Inverse Document Frequency) = qué tan rara es esa frase en
  el resto del corpus.

Lo que sale arriba: frases que **este comediante usa mucho y los demás
casi nada**. Filtra el dashboard a un solo comediante (p.ej. Louis C.K.)
y aún funciona: se compara contra el corpus completo de fondo, no solo
contra el subset visible.
""")
    col1, col2, col3 = st.columns(3)
    lo = col1.slider("n-grama min", 2, 4, 2)
    hi = col2.slider("n-grama max", lo, 5, max(lo, 4))
    top_k = col3.slider("Top por comediante", 5, 30, 10)

    idx_key = tuple(df.index.tolist())
    cp = cached_catchphrases(idx_key, lo, hi, top_k)
    st.markdown(narrate_catchphrases(cp, top_n=5))

    candidates = sorted([c for c, s in cp.items() if len(s)])
    if candidates:
        com_choice = st.selectbox("Ver detalle de un comediante", candidates)
        s = cp[com_choice]
        fig = px.bar(s.sort_values(), orientation="h",
                     title=f"Catchphrases de {com_choice}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        if len(cp) == 1:
            st.caption(
                "Solo hay un comediante en el subset. Las frases mostradas "
                "son distintivas de él **comparadas con el corpus completo** "
                "(no solo el subset visible)."
            )
    else:
        st.info("Sin catchphrases detectables en este subset.")

# === Clustering =========================================================
with tabs[6]:
    with st.expander("📖 ¿Qué significa el clustering?"):
        st.markdown("""
Cada **comediante** se representa como un vector con la frecuencia
relativa (TF-IDF) de cada palabra que usa en todos sus shows
combinados. Dos pasos:

1. **K-means** agrupa esos vectores en *k* clusters: comediantes en
   el mismo cluster tienen vocabularios parecidos.
2. **UMAP** reduce esos vectores de ~10.000 dimensiones a **2
   coordenadas (x, y)** preservando la cercanía relativa. Comediantes
   cerca en el gráfico = vocabularios similares.

**Cómo leerlo**:
- Distancia entre puntos = qué tan distinto es el lenguaje que usan.
- Color = cluster asignado por k-means.
- Tamaño = número de shows del comediante en el subset (más shows =
  estimación más confiable).

**Qué NO significa**:
- No mide "qué tan parecidos son sus chistes" — solo el léxico.
- Dos comediantes en clusters distintos pueden hablar del mismo tema
  con palabras distintas; dos comediantes en el mismo cluster pueden
  tratar temas opuestos si usan vocabulario parecido.

**Casos de uso**:
- Encontrar el comediante con estilo más parecido al tuyo favorito.
- Detectar "escuelas" estilísticas dentro del subset filtrado.
""")
    k = st.slider("Número de clusters (k-means)", 2, 10, 5)

    # If the filtered subset has <3 comedians, fall back to the full corpus
    # but highlight the comedians the user selected.
    if df["comedian"].nunique() < 3:
        idx_key = tuple(df_all.index.tolist())
        highlighted = set(df["comedian"].unique())
        st.info(
            f"Subset con solo {df['comedian'].nunique()} comediante(s). "
            f"Mostrando el mapa estilístico **del corpus completo** con "
            f"{', '.join(highlighted)} resaltado(s)."
        )
    else:
        idx_key = tuple(df.index.tolist())
        highlighted = set()

    clus = cached_clusters(idx_key, k)
    if len(clus) == 0:
        st.info("Subset muy pequeño para clustering.")
    else:
        if highlighted:
            clus = clus.copy()
            clus["highlight"] = clus["comedian"].isin(highlighted)
            fig = px.scatter(
                clus, x="x", y="y", color=clus["cluster"].astype(str),
                size="n_shows", hover_name="comedian",
                symbol="highlight", symbol_map={True: "star", False: "circle"},
                labels={"color": "Cluster", "symbol": "Resaltado"},
                title="Mapa estilístico — corpus completo (★ = resaltado)",
                height=600,
            )
        else:
            st.markdown(
                f"**{len(clus)} comediantes** proyectados a 2D con UMAP. "
                f"Tamaño del punto ∝ número de shows. Colores = clusters de k-means."
            )
            fig = px.scatter(
                clus, x="x", y="y", color=clus["cluster"].astype(str),
                size="n_shows", hover_name="comedian",
                labels={"color": "Cluster"},
                title="Mapa estilístico de comediantes",
                height=600,
            )
        fig.update_traces(textposition="top center", marker=dict(opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

        # If a comedian was highlighted, also show their nearest neighbors
        if highlighted:
            st.subheader("Comediantes más cercanos a los resaltados")
            from scipy.spatial.distance import cdist
            import numpy as np
            coords = clus[["x", "y"]].values
            for com in highlighted:
                if com not in clus["comedian"].values:
                    continue
                i = clus.index[clus["comedian"] == com][0]
                d = cdist([coords[i]], coords).ravel()
                order = np.argsort(d)[1:8]  # closest 7
                neighbors = clus.iloc[order]["comedian"].tolist()
                st.markdown(f"**{com}**: {' · '.join(neighbors)}")

        st.subheader("Comediantes por cluster")
        for c in sorted(clus["cluster"].unique()):
            members = clus[clus["cluster"] == c]["comedian"].tolist()
            with st.expander(f"Cluster {c} — {len(members)} comediantes"):
                st.write(", ".join(members))

# === Predicción rating ==================================================
with tabs[7]:
    with st.expander("📖 ¿Cómo funciona este modelo predictivo?"):
        st.markdown("""
**Pregunta**: ¿hay palabras/features del texto que **correlacionen** con
recibir mejor o peor rating en IMDb?

**Modelo**: una **regresión Ridge** sobre dos tipos de features:
1. **TF-IDF de cada palabra y bigrama** del transcript (~2.000 features).
2. **Features numéricas**: nº palabras totales, runtime en minutos
   (normalizadas).

**Qué reporta**:
- **R² train** = qué tan bien se ajusta a los datos vistos. Alto y CV
  bajo = el modelo memoriza.
- **R² CV-5** = capacidad de generalizar a shows que no vio en
  entrenamiento. **Este es el número honesto**. Con n~80 y ~2.000
  features esperá un R² CV bajo (incluso negativo) — eso es normal y
  refleja que **predecir rating con texto es muy difícil**.
- **Top features positivas/negativas**: las palabras/features cuyo
  coeficiente Ridge es más alto (+) o más bajo (−). Lectura: si una
  palabra está arriba en "suben el rating", los shows que la usan
  **tienden a estar mejor calificados** — eso NO significa que añadir
  la palabra subirá el rating. Es **correlación, no causalidad**.

**Por qué Ridge y no Random Forest**:
- Ridge tiene **coeficientes interpretables** (positivo = sube,
  negativo = baja).
- Con n pequeño y muchos features, Ridge generaliza mejor que árboles
  porque regulariza fuertemente.
- El α controla cuánto: más alto = más regularización = coeficientes
  más cercanos a 0 pero más estables.

**Cuándo confiar**:
- Si R² CV > 0.2 con coeficientes consistentes al cambiar α, hay señal.
- Si R² CV oscila mucho al cambiar α o el subset, los coeficientes
  son ruido.
""")
    alpha = st.slider("Regularización Ridge (alpha)", 0.1, 10.0, 1.0, 0.1)
    idx_key = tuple(df.index.tolist())
    pred = cached_predict(idx_key, alpha)
    st.markdown(narrate_predictor(pred))

    if "error" not in pred:
        col1, col2 = st.columns(2)
        with col1:
            s = pred["top_positive"].sort_values()
            fig = px.bar(s, orientation="h", title="Features que SUBEN el rating",
                         color_discrete_sequence=["#2ecc71"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            s = pred["top_negative"].sort_values()
            fig = px.bar(s, orientation="h", title="Features que BAJAN el rating",
                         color_discrete_sequence=["#e74c3c"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⚠️ **Interpretación con cuidado.** Con pocos datos y muchos features de "
            "texto, los coeficientes son **sugerentes**, no causales. Un R² CV bajo "
            "significa que el modelo no generaliza — úsalo como exploratorio."
        )

# === Por año ============================================================
with tabs[8]:
    st.subheader("Evolución temporal")
    with st.expander("📖 ¿Qué muestra esta pestaña?"):
        st.markdown("""
Cómo cambia el subset filtrado a través del tiempo: qué palabras se usan
más en cada época, cómo evolucionan las emociones y el sentimiento, y
cómo se compara con el rating IMDb. Los años se agrupan en **buckets**
(por defecto de 5 años) para suavizar el ruido con pocas observaciones
por año.
""")
    bucket = st.slider("Tamaño del bucket (años)", 1, 10, 5)

    yr_sent = yearly_sentiment_rating(df, bucket=bucket)
    if len(yr_sent) == 0:
        st.info("No hay años suficientes en el subset.")
    else:
        # Sentimiento + rating + N shows
        st.markdown("#### Sentimiento, rating y volumen por bucket")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(yr_sent.reset_index(), x="bucket", y="sentiment",
                          markers=True, title="Sentimiento promedio (VADER) por bucket")
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(yr_sent.reset_index().dropna(subset=["rating"]),
                          x="bucket", y="rating", markers=True,
                          title="Rating IMDb promedio por bucket")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(yr_sent.round(3), use_container_width=True)

        # Palabras top por época
        st.markdown("#### Evolución de palabras")
        st.caption("Selecciona palabras a seguir, o deja vacío para usar las top del subset completo.")
        custom_words_input = st.text_input(
            "Palabras (separadas por coma)", value="",
            placeholder="ej: trump, marriage, internet, twitter",
        )
        custom_words = [w.strip().lower() for w in custom_words_input.split(",") if w.strip()] or None
        top_n_auto = st.slider("Si no hay lista, top N palabras del subset", 5, 20, 8,
                               key="ywt_topn")
        ywt = yearly_word_trends(df, words=custom_words, bucket=bucket, top_k=top_n_auto)
        if len(ywt):
            ywt_plot = ywt.drop(columns=["_total_tokens", "_shows"], errors="ignore")
            long = ywt_plot.reset_index().melt(id_vars="bucket", var_name="word", value_name="per_1000")
            fig = px.line(long, x="bucket", y="per_1000", color="word", markers=True,
                          title="Frecuencia relativa por bucket (por 1000 tokens)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ywt.round(2), use_container_width=True)

        # Emociones por época
        st.markdown("#### Evolución de las emociones (NRC)")
        try:
            emo_y = cached_emotions(tuple(df.index.tolist()))
            yet = yearly_emotion_trends(emo_y, df, bucket=bucket)
            if len(yet):
                long = yet.reset_index().melt(id_vars="bucket", var_name="emotion", value_name="score")
                fig = px.area(long, x="bucket", y="score", color="emotion",
                              title="Mix emocional promedio por bucket (apilado)")
                st.plotly_chart(fig, use_container_width=True)
                # Heatmap alternativo
                fig = px.imshow(yet.T, aspect="auto", color_continuous_scale="RdYlBu_r",
                                title="Heatmap emoción × bucket",
                                labels=dict(x="Bucket de año", y="Emoción", color="Score"))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No pude calcular emociones por año: {e}")


# === Triggers de risa ===================================================
with tabs[9]:
    st.subheader("¿Qué hace reír al público?")
    with st.expander("📖 ¿Cómo se calculan los triggers?"):
        st.markdown("""
Los transcripts incluyen marcadores como `[laughter]`, `[applause]`,
`[crowd laughs]`. Esta pestaña extrae las **N palabras justo antes**
de cada marcador y busca patrones:

- **Top n-gramas trigger**: frases (2–4 palabras) que más
  frecuentemente preceden a una reacción del público.
- **Última palabra antes de la reacción**: la palabra individual que
  más veces "remata" un chiste o cierra el set-up.

**Limitaciones**:
- No todos los transcripts marcan reacciones (depende del transcriptor
  de scrapsfromtheloft). Sin marcadores no hay triggers.
- Captura **correlación, no causalidad**: una palabra frecuente antes
  de risas puede ser solo común en el lenguaje del comediante.
- Las reacciones a veces se anotan al final del párrafo, así que
  podemos perder el set-up exacto.
""")
    col1, col2, col3 = st.columns(3)
    kind = col1.radio("Tipo de reacción", ["laughter", "applause", "both"],
                       index=0, horizontal=True)
    n_words = col2.slider("Palabras antes del marcador", 3, 12, 6)
    top_k_tr = col3.slider("Top K resultados", 10, 50, 25, key="tr_topk")

    @st.cache_data(show_spinner="Calculando triggers...")
    def cached_triggers(idx_key, kind, n_words, top_k):
        sub = df_all.loc[list(idx_key)]
        return laughter_triggers(sub, kind=kind, n_words=n_words, top_k=top_k)

    tr = cached_triggers(tuple(df.index.tolist()), kind, n_words, top_k_tr)

    st.metric("Marcadores encontrados", tr["total_markers"])
    if tr["total_markers"] == 0:
        st.warning("Este subset no tiene marcadores de reacción del público.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top n-gramas que disparan la reacción")
            if len(tr["top_ngrams"]):
                fig = px.bar(tr["top_ngrams"].sort_values(), orientation="h",
                             labels={"value": "Apariciones", "index": ""})
                fig.update_layout(height=500, showlegend=False,
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sin n-gramas frecuentes detectados (requiere min_df=2).")
        with col2:
            st.markdown("#### Última palabra antes de la reacción")
            if len(tr["top_last_words"]):
                fig = px.bar(tr["top_last_words"].sort_values(), orientation="h",
                             labels={"value": "Apariciones", "index": ""})
                fig.update_layout(height=500, showlegend=False,
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        # Sample windows
        st.markdown("#### Ejemplos de líneas que disparan la reacción")
        st.caption(f"Muestra de hasta 15 fragmentos (de {tr['total_markers']} totales)")
        for w in tr["windows"][:15]:
            st.markdown(f"- *…{w}* → [reacción]")


# === Datos ==============================================================
with tabs[10]:
    st.subheader("Subset filtrado")
    show_cols = ["title", "comedian", "year", "rating", "votes", "runtime_min",
                 "word_count", "ttr", "sentiment"]
    st.dataframe(df[show_cols].round(3), use_container_width=True, height=500)
    csv = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, "stand_up_subset.csv", "text/csv")
