"""Pruebas de audio/metricas.py con cuadros sintéticos. Correr: python audio/test_metricas.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metricas as M  # noqa: E402

FALLAS = []


def check(nombre, cond, detalle=""):
    print(f"  {'ok' if cond else 'FALLA'}  {nombre}  {'' if cond else detalle}")
    if not cond:
        FALLAS.append(nombre)


# 10 minutos, un cuadro por segundo. Intro de 60 s con música y sin habla; luego habla continua.
t = np.arange(0, 600, 1.0, dtype=np.float32)
habla = np.where(t < 60, 0.0, 0.9).astype(np.float32)
risa = np.zeros_like(t)
aplauso = np.zeros_like(t)
for s in range(100, 580, 20):          # una reacción de 2 s cada 20 s desde el segundo 100
    risa[(t >= s) & (t < s + 2)] = 0.08
aplauso[(t >= 300) & (t < 305)] = 0.6  # un aplauso fuerte de 5 s a mitad de show
c = {"t": t, "risa": risa, "aplauso": aplauso, "habla": habla, "musica": np.zeros_like(t),
     "db": np.full_like(t, -20.0)}

m, arco = M.metricas(c)
check("recorta la intro sin habla", 40 <= 600 - 60 * m["minutos"] <= 80, m["minutos"])
check("cuenta las 24 reacciones", abs(m["reacciones_min"] * m["minutos"] - 24) <= 1, m["reacciones_min"] * m["minutos"])
check("mide el intervalo entre reacciones", m["segundos_entre_reacciones"] == 20.0, m["segundos_entre_reacciones"])
check("detecta el aplauso de mitad de show", m["aplausos_hora"] > 0, m["aplausos_hora"])
check("el tramo seco más largo es el arranque antes de la primera risa",
      m["tramo_seco_max_s"] >= 40 and m["primera_reaccion_s"] >= 40, (m["tramo_seco_max_s"], m["primera_reaccion_s"]))
check("hay rachas: reacciones a 20 s no cuentan (tope 8 s)", m["rachas_hora"] == 0, m["rachas_hora"])
check("diez décimos en el arco", len(arco) == 10)

# Con umbral más alto que la señal, no hay reacciones: el umbral sí manda.
m2, _ = M.metricas(c, umbral=0.5)
check("umbral por encima de la señal: solo el aplauso", m2["reacciones_min"] < 0.5, m2["reacciones_min"])

# Racha: cuatro reacciones a 5 s.
risa2 = np.zeros_like(t)
for s in (200, 205, 210, 215):
    risa2[(t >= s) & (t < s + 2)] = 0.1
m3, _ = M.metricas({**c, "risa": risa2, "aplauso": np.zeros_like(t)})
check("cuatro reacciones seguidas a 5 s forman una racha", m3["rachas_hora"] > 0, m3["rachas_hora"])

sys.exit(1 if FALLAS else 0)
