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
    apply_filters,
    catchphrases_by_comedian,
    cluster_comedians,
    emotion_profile,
    extract_topics,
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
    return catchphrases_by_comedian(sub, ngram_range=(lo, hi), top_k=top_k)


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

st.sidebar.markdown("---")
st.sidebar.caption("Los filtros se aplican a todas las pestañas.")

df = apply_filters(
    df_all,
    comedians=selected or None,
    year_range=year_range,
    min_rating=min_rating,
    min_votes=int(min_votes),
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
    st.subheader("Perfil emocional (NRC Word-Emotion Association Lexicon)")
    idx_key = tuple(df.index.tolist())
    try:
        emo = cached_emotions(idx_key)
        st.markdown(narrate_emotions(emo))

        avg = emo.groupby("emotion")["score"].mean().sort_values(ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(avg, title="Promedio del subset", labels={"value": "Proporción", "index": "Emoción"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_coms = df["comedian"].value_counts().head(10).index
            heat = emo[emo["comedian"].isin(top_coms)].groupby(
                ["comedian", "emotion"])["score"].mean().unstack()
            if not heat.empty:
                fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlBu_r",
                                title="Heatmap por comediante (top 10)")
                st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.error("Falta el paquete `nrclex`. Instala: `pip install nrclex`")

# === Tópicos LDA ========================================================
with tabs[4]:
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
    col1, col2, col3 = st.columns(3)
    lo = col1.slider("n-grama min", 2, 4, 2)
    hi = col2.slider("n-grama max", lo, 5, max(lo, 4))
    top_k = col3.slider("Top por comediante", 5, 30, 10)

    idx_key = tuple(df.index.tolist())
    cp = cached_catchphrases(idx_key, lo, hi, top_k)
    st.markdown(narrate_catchphrases(cp, top_n=5))

    if len(cp) >= 2:
        com_choice = st.selectbox(
            "Ver detalle de un comediante",
            sorted([c for c, s in cp.items() if len(s)]),
        )
        s = cp[com_choice]
        fig = px.bar(s.sort_values(), orientation="h",
                     title=f"Catchphrases de {com_choice}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Necesitas al menos 2 comediantes en el subset.")

# === Clustering =========================================================
with tabs[6]:
    k = st.slider("Número de clusters (k-means)", 2, 10, 5)
    idx_key = tuple(df.index.tolist())
    clus = cached_clusters(idx_key, k)
    if len(clus) == 0:
        st.info("Subset muy pequeño para clustering.")
    else:
        st.markdown(
            f"**{len(clus)} comediantes** proyectados a 2D con UMAP. "
            f"Tamaño del punto ∝ número de shows. Colores = clusters de k-means sobre TF-IDF."
        )
        fig = px.scatter(
            clus, x="x", y="y", color=clus["cluster"].astype(str),
            size="n_shows", hover_name="comedian",
            labels={"color": "Cluster"}, title="Mapa estilístico de comediantes",
            height=600,
        )
        fig.update_traces(textposition="top center", marker=dict(opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Comediantes por cluster")
        for c in sorted(clus["cluster"].unique()):
            members = clus[clus["cluster"] == c]["comedian"].tolist()
            with st.expander(f"Cluster {c} — {len(members)} comediantes"):
                st.write(", ".join(members))

# === Predicción rating ==================================================
with tabs[7]:
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

# === Datos ==============================================================
with tabs[8]:
    st.subheader("Subset filtrado")
    show_cols = ["title", "comedian", "year", "rating", "votes", "runtime_min",
                 "word_count", "ttr", "sentiment"]
    st.dataframe(df[show_cols].round(3), use_container_width=True, height=500)
    csv = df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, "stand_up_subset.csv", "text/csv")
