# MEJORAS — TextMining (stand_up_research)

> 84 especiales con transcript + IMDb, dashboard Streamlit de 9 pestañas.
> ⚠️ Hay ~2 semanas de trabajo SIN COMMITEAR (Chappelle ×2 + fix circularidad precompute.py).
> Prompts listos para pegar en Claude Code (Fable 5) desde esta carpeta.

---

## 0. ~~Rescatar el trabajo sin commitear~~ ✅ YA RESUELTO

Verificado 7 jul 2026: commit `8ac9de7` (Chappelle ×2 + fix circular en precompute)
existe y está pusheado. Working tree limpio.

## 1. Completar los hallazgos vacíos [PLAN]

```
hallazgos-clave del vault tiene 4 secciones marcadas "pendiente extraer patrones":
sentimiento VADER, emociones NRC, correlación TTR↔rating, y temas↔rating.
Ciérralas con análisis real sobre df_enriched.parquet y topic_scores.parquet:

1. Correlación de cada feature (compound VADER, 8 emociones NRC, TTR, 8 temas
   curados) con rating IMDb: Spearman + IC bootstrap (n≈79, reportar incertidumbre
   siempre — regla del proyecto: tendencias sugerentes, no causales).
2. Los 5 hallazgos más sólidos escritos en una frase cada uno CON números
   (ej: "los especiales en el cuartil alto de X promedian rating Y vs Z").
3. Notebook notebooks/hallazgos_2026.ipynb reproducible + actualizar la pestaña
   del dashboard correspondiente.
4. Actualizar obsidian_vaults/negocio_comedia/Comedia/hallazgos-clave.md con los
   resultados verificados (regla anti-fabricación: solo números que salgan del
   análisis).
Entra en plan mode primero.
```

## 2. Ampliar dataset hacia 200 especiales

```
Meta declarada: +200 especiales. El scraping de IMDb está roto y la columna
comedian es poco confiable. Plan de ampliación:
1. Diagnostica el scraper de IMDb actual y arréglalo o reemplázalo (cinemagoer/
   API no oficial — evalúa opciones y elige la más estable).
2. De raw_transcripts.parquet (400 transcripciones) hay ~316 sin match IMDb:
   construye el matcher título+comediante+año con fuzzy matching y cola de
   revisión manual en CSV para ambiguos.
3. Pipeline incremental: agregar un especial nuevo = 1 comando que re-deriva
   enriched/topics/emotions solo para las filas nuevas.
Objetivo de la primera pasada: llegar a 150 unificados. Documenta el corpus
resultante (n exacto por fuente) en README.
```

## 3. Análisis de estructura y timing (lo que Ridge no ve)

```
Limitación conocida: Ridge sobre TF-IDF no captura estructura narrativa ni timing.
Agrega features estructurales por especial:
- Densidad de risas por minuto si el transcript trae marcadores [laughter]/(laughs)
  (verificar cobertura real en el corpus primero — si <50% lo tienen, reportar y parar).
- Arco de sentimiento: VADER por décimo del show → shape (arranque vs cierre),
  y volatilidad emocional.
- Longitud media de "bloque" entre risas como proxy de ritmo setup-punchline.
Correlacionar con rating, sumar al modelo (Ridge con features estructurales vs
solo TF-IDF, comparar R² con CV) y nueva pestaña "Estructura" en el dashboard.
```

## 4. Comparador "mi material vs referentes"

```
Meta personal: aplicar hallazgos al material propio. Construye el comparador:
1. Ingesta de un transcript propio en español (transcripciones ya salen del
   pipeline de Analisis_videos): normalizar y computar TTR, VADER-es (o léxico
   español equivalente — evaluar y documentar la limitación cross-idioma),
   NRC español (el léxico NRC tiene traducciones oficiales), y temas curados
   traducidos.
2. Vista en el dashboard: radar de mi show vs los top-10 por rating del corpus,
   con disclaimer de idioma.
3. Reporte por show propio: en qué me parezco a los de rating alto y en qué no.
Empezar con 1 transcript real de NAPE del proyecto Analisis_videos.
```

## 5. Publicar los hallazgos

```
Meta declarada: publicar (blog/LinkedIn/artículo). Con los hallazgos del prompt #1
cerrados, escribe el artículo en español: "Analicé N especiales de stand-up con
text mining: esto correlaciona con un buen rating". Estructura: pregunta → método
en lenguaje llano → 5 hallazgos con sus gráficas (exportar del dashboard en
paleta limpia) → limitaciones honestas (inglés, n pequeño, correlación≠causa) →
qué sigue. Dos versiones: post LinkedIn (~800 palabras, hook fuerte) y artículo
largo. Firma: Andrés Loaiza, ingeniero de datos y comediante de Alta Comedia.
```
