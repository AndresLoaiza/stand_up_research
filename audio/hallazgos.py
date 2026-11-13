"""
Imprime los números del informe de audio. Nada aquí es nuevo cálculo: lee los parquets de
metricas.py y validar.py y los resume, para que cada cifra del informe se pueda volver a sacar.

Uso:  python audio/hallazgos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from analysis import audio_arc_shape, audio_correlations, load_audio, load_audio_arco  # noqa: E402

DF = RAIZ / "data" / "data_frame"
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 46)


def titulo(t):
    print(f"\n{'=' * 8} {t}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, arco = load_audio(), load_audio_arco()
    a["show"] = a["comediante"] + " — " + a["especial"] + a["anio"].map(lambda x: f" ({int(x)})" if pd.notna(x) else "")
    ref, propios = a[a["grupo"] == "referente"], a[a["grupo"] == "propio"]

    titulo("Validación contra subtítulos (reacción = risa + aplauso; 0.05 fijo vs p70 relativo)")
    v = pd.read_parquet(DF / "df_audio_validacion.parquet")
    print(v[(v["senal"] == "risa+aplauso") & (v["umbral"].isin(["0.05", "p70"]))].to_string(index=False))
    print(v.groupby(["senal", "umbral"])["sensibilidad"].median().unstack(0))

    titulo("Robustez: ¿el ranking cambia si se mueve el umbral?")
    sens = pd.read_parquet(DF / "df_audio_sensibilidad.parquet")
    sens = sens[sens["id"].isin(ref["id"])]
    for col, pares in (("reacciones_min", (("0.03", "0.05"), ("0.05", "0.08"), ("p60", "p70"), ("p70", "p80"))),
                       ("segundos_entre_reacciones", (("p60", "p70"), ("p70", "p80"))),
                       ("rachas_hora", (("p60", "p70"), ("p70", "p80")))):
        s = sens.pivot(index="id", columns="umbral", values=col)
        for u1, u2 in pares:
            print(f"  {col:<26} ranking con {u1} vs {u2}: Spearman {spearmanr(s[u1], s[u2], nan_policy='omit')[0]:.2f}")

    titulo("¿Mide la comedia o la grabación? Varianza dentro y entre comediantes (3+ especiales)")
    var = ref.groupby("comediante").filter(lambda x: len(x) >= 3)
    for c in ("reacciones_min_fijo", "senal_p90", "reacciones_min", "masa_reaccion_min", "tramos_secos_hora", "apertura", "cierre", "concentracion_top20", "posicion_picos", "secos_relativos_hora", "segundos_entre_reacciones"):
        dentro = var.groupby("comediante")[c].var().mean()
        total = var[c].var()
        print(f"  {c:<22} varianza dentro de cada comediante = {100 * dentro / total:.0f}% de la total "
              f"(si se acerca a 100%, la métrica dice más de la grabación que del comediante)")
    print("  Volumen de la reacción frente al habla del mismo show vs reacciones/min: "
          f"rho={spearmanr(ref['reaccion_db_sobre_habla'], ref['reacciones_min'], nan_policy='omit')[0]:.2f}")

    titulo(f"Distribución en {len(ref)} especiales de referencia")
    cols = ["reacciones_min", "segundos_entre_reacciones", "reaccion_dur_mediana_s", "reacciones_grandes_hora",
            "aplausos_hora", "tramos_secos_hora", "rachas_hora", "apertura", "cierre", "palabras_por_reaccion",
            "palabras_min_habladas", "pct_tiempo_habla"]
    print(ref[cols].describe(percentiles=[0.25, 0.5, 0.75]).T[["count", "25%", "50%", "75%", "min", "max"]].round(2))

    titulo("Más y menos reacción por minuto")
    o = ref.sort_values("reacciones_min", ascending=False)[["show", "reacciones_min", "segundos_entre_reacciones",
                                                            "reaccion_dur_mediana_s", "aplausos_hora", "rating"]]
    print(o.head(8).to_string(index=False))
    print("...")
    print(o.tail(8).to_string(index=False))

    titulo("Por comediante (mediana, solo con 2+ especiales)")
    g = ref.groupby("comediante").agg(n=("id", "count"), reac=("reacciones_min", "median"),
                                      entre=("segundos_entre_reacciones", "median"),
                                      dur=("reaccion_dur_mediana_s", "median"), grandes=("reacciones_grandes_hora", "median"),
                                      aplausos=("aplausos_hora", "median"), cv_reac=("reacciones_min", lambda x: x.std() / x.mean()))
    print(g[g["n"] >= 2].sort_values("reac", ascending=False).round(2))

    titulo("Audio contra rating IMDb")
    con = ref.dropna(subset=["rating"])
    print(f"n = {len(con)}")
    print(audio_correlations(con).to_string(index=False))
    print("\nTexto contra rating en los mismos especiales (para comparar fuerzas):")
    for c in ("words_per_min", "ttr", "sentiment"):
        sub = con[[c, "rating"]].dropna()
        r, p = spearmanr(sub[c], sub["rating"])
        print(f"  {c:<16} n={len(sub)}  rho={r:.2f}  p={p:.3f}")

    titulo("Forma del show (referentes): ritmo relativo por décimo")
    print(audio_arc_shape(arco, ref["id"]).round(2).to_string(index=False))
    if len(propios):
        print("\nAlta Comedia:")
        print(audio_arc_shape(arco, propios["id"]).round(2).to_string(index=False))

    titulo("Evolución (comediantes con 4+ especiales)")
    for c, sub in ref.groupby("comediante"):
        if len(sub) < 4:
            continue
        sub = sub.sort_values("anio")
        r, p = spearmanr(sub["anio"], sub["reacciones_min"])
        print(f"  {c:<16} {len(sub)} especiales {int(sub['anio'].min())}-{int(sub['anio'].max())}: "
              f"reacciones/min vs año rho={r:.2f} p={p:.2f} | "
              + ", ".join(f"{int(y)}:{v:.1f}" for y, v in zip(sub["anio"], sub["reacciones_min"])))

    titulo("Alta Comedia (con percentil frente a los referentes)")
    for _, r in propios.iterrows():
        pc = {c: int(round(100 * (ref[c] < r[c]).mean())) for c in ("reacciones_min", "segundos_entre_reacciones",
                                                                      "tramos_secos_hora", "aplausos_hora")}
        print(f"  {r['show']:<52} {r['formato']:<26} reac/min {r['reacciones_min']:.2f} (p{pc['reacciones_min']}) | "
              f"entre {r['segundos_entre_reacciones']} s (p{pc['segundos_entre_reacciones']}) | secos/h {r['tramos_secos_hora']} "
              f"(p{pc['tramos_secos_hora']}) | aplausos/h {r['aplausos_hora']} | apertura {r['apertura']} cierre {r['cierre']} "
              f"| peor décimo {r['peor_decil']} | habla {r['pct_tiempo_habla']}%")


if __name__ == "__main__":
    main()
