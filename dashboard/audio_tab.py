"""Pestaña "Risa medida (audio)": la reacción del público escuchada en el video, no leída en el texto."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import AUDIO_METRICS, audio_arc_shape, audio_correlations, load_audio, load_audio_arco
from analysis.core import PROJECT_ROOT

VALIDACION = PROJECT_ROOT / "data" / "data_frame" / "df_audio_validacion.parquet"
CLAVE = ["reacciones_min", "segundos_entre_reacciones", "reaccion_dur_mediana_s", "reacciones_grandes_hora",
         "aplausos_hora", "tramos_secos_hora", "rachas_hora", "apertura", "cierre", "concentracion_top20",
         "posicion_picos"]


@st.cache_data(show_spinner="Cargando métricas de audio...")
def _datos():
    return load_audio(), load_audio_arco()


def _nombre(a: pd.DataFrame) -> pd.Series:
    anio = a["anio"].map(lambda x: f" ({int(x)})" if pd.notna(x) else "")
    return a["comediante"] + " — " + a["especial"] + anio


def render_audio_tab():
    a, arco = _datos()
    st.subheader("La risa medida en el audio de cada especial")
    if a.empty:
        st.info("Aún no hay métricas de audio. Corre `audio/extraer_risas.py` y `audio/metricas.py`.")
        return
    a = a.assign(show=_nombre(a))
    ref = a[a["grupo"] == "referente"]
    propios = a[a["grupo"] == "propio"]

    with st.expander("📖 Cómo se midió y cuánto creerle"):
        st.markdown(f"""
Se escuchó el audio completo de **{len(ref)} especiales** de la carpeta `Stand up` (sin duplicados, muestras,
compilaciones ni documentales) y de **{len(propios)} shows propios**. Un modelo de sonido (YAMNet) dice, cada
segundo, cuánto suena a risa, aplauso y habla. Se cuenta una **reacción** cuando risa + aplauso pasan de 0,05.

**Por qué "reacción" y no "risa"**: en la mezcla de TV la voz del comediante tapa al público y el modelo llama
"aplauso" a muchas carcajadas. El umbral se eligió comparando contra subtítulos para sordos que marcan las
risas con hora exacta (tabla de abajo).

**Lo que esto NO permite**: comparar shows cuya diferencia sea menor a un tercio (la mezcla de sonido mueve
la cifra más que eso), ni comparar de igual a igual un especial de TV con una grabación de celular en un bar.
Lo más confiable es la **forma** de cada show (dónde reacciona el público), no su cifra absoluta.
""")
        if VALIDACION.exists():
            v = pd.read_parquet(VALIDACION)
            v = v[(v["senal"] == "risa+aplauso") & (v["umbral"] == "p70")]
            st.dataframe(v.drop(columns=["senal", "umbral"]).rename(columns={
                "marcas_sdh_min": "risas marcadas en subtítulos / min", "detectadas_min": "reacciones detectadas / min",
                "sensibilidad": "% de risas marcadas que se detectan"}), use_container_width=True, hide_index=True)

    # --- Ranking -----------------------------------------------------------
    st.markdown("#### Los especiales, de más a menos reacción por minuto")
    tabla = ref.sort_values("reacciones_min", ascending=False)[["show", "minutos", *CLAVE, "rating"]]
    st.dataframe(tabla.rename(columns={**AUDIO_METRICS, "minutos": "Minutos", "rating": "IMDb"}).round(2),
                 use_container_width=True, hide_index=True, height=420)

    # --- Audio contra rating ----------------------------------------------
    st.markdown("#### ¿Más risa medida = mejor rating en IMDb?")
    con = ref.dropna(subset=["rating"])
    st.caption(f"Solo {len(con)} especiales tienen audio y rating a la vez. Spearman; p < 0,05 sería indicio, "
               "no prueba: con tantas métricas probadas a la vez, alguna sale 'significativa' por azar.")
    col1, col2 = st.columns([3, 2])
    with col1:
        metrica = st.selectbox("Métrica", list(AUDIO_METRICS), format_func=AUDIO_METRICS.get, key="audio_m")
        fig = px.scatter(con, x=metrica, y="rating", color="comediante", hover_name="show",
                         labels={metrica: AUDIO_METRICS[metrica], "rating": "Rating IMDb"})
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(audio_correlations(con), use_container_width=True, hide_index=True)

    # --- Forma del show ------------------------------------------------------
    st.markdown("#### La forma de un especial: dónde reacciona el público")
    st.caption("Cuánta reacción hay en cada décimo del show, dividido por el promedio del propio show "
               "(1 = su ritmo normal). Al ser relativo a cada show, la mezcla de sonido casi no lo mueve. "
               "Línea = mediana; banda = la mitad central de los especiales.")
    fig = go.Figure()
    for nombre, ids, color in (("Referentes", ref["id"], "#2E6F95"), ("Alta Comedia", propios["id"], "#D1495B")):
        f = audio_arc_shape(arco, ids)
        if f.empty:
            continue
        fig.add_trace(go.Scatter(x=list(f["decil"]) + list(f["decil"])[::-1], y=list(f["p75"]) + list(f["p25"])[::-1],
                                 fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f["decil"], y=f["mediana"], name=f"{nombre} (n={int(f['n'].max())})",
                                 line=dict(color=color, width=3)))
    fig.add_hline(y=1, line_dash="dot", line_color="gray")
    fig.update_layout(height=380, xaxis_title="Décimo del show", yaxis_title="Ritmo relativo",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- Perfil por comediante --------------------------------------------
    st.markdown("#### Estilo de cada comediante")
    perfil = ref.groupby("comediante").agg(
        especiales=("id", "count"), reacciones_min=("reacciones_min", "median"),
        segundos_entre=("segundos_entre_reacciones", "median"), dur_reaccion=("reaccion_dur_mediana_s", "median"),
        grandes_hora=("reacciones_grandes_hora", "median"), aplausos_hora=("aplausos_hora", "median"),
        palabras_por_reaccion=("palabras_por_reaccion", "median"),
    ).sort_values("reacciones_min", ascending=False).round(1)
    st.dataframe(perfil, use_container_width=True)
    fig = px.scatter(ref, x="segundos_entre_reacciones", y="reaccion_dur_mediana_s", color="comediante",
                     hover_name="show", size="minutos",
                     labels={"segundos_entre_reacciones": "Segundos entre reacciones (más = construye más)",
                             "reaccion_dur_mediana_s": "Duración de cada reacción (más = risas más grandes)"})
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- Evolución -----------------------------------------------------------
    varios = ref.groupby("comediante")["id"].count()
    varios = varios[varios >= 4].index
    if len(varios):
        st.markdown("#### Cómo cambia un comediante con los años")
        m2 = st.selectbox("Métrica", CLAVE, format_func=AUDIO_METRICS.get, key="audio_evo")
        evo = ref[ref["comediante"].isin(varios)].sort_values("anio")
        fig = px.line(evo, x="anio", y=m2, color="comediante", markers=True, hover_name="show",
                      labels={"anio": "Año", m2: AUDIO_METRICS[m2]})
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- Alta Comedia ----------------------------------------------------------
    if len(propios):
        st.markdown("#### Alta Comedia frente a los referentes")
        st.caption("Percentil = qué parte de los especiales de referencia queda por debajo. Ojo: grabación de "
                   "celular o cámara en sala contra mezcla de TV, en español, y formatos distintos (panel, juegos, "
                   "dúo). Sirve para comparar sus shows entre sí y la forma de cada uno, no para decir quién es más gracioso.")
        filas = []
        for _, r in propios.iterrows():
            fila = {"show": r["show"], "formato": r["formato"]}
            for c in ("reacciones_min", "segundos_entre_reacciones", "tramos_secos_hora", "apertura", "cierre"):
                fila[AUDIO_METRICS[c]] = r[c]
                fila[f"percentil {c}"] = round(100 * float((ref[c] < r[c]).mean())) if pd.notna(r[c]) else None
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
