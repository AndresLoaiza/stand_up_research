"""
Escucha un especial y guarda, segundo a segundo, cuánto suena a risa, aplauso y habla.

Corre con el venv de Analisis_videos (tiene mediapipe; este repo no):
  D:\\ANDRES\\Claude_Projects\\Analisis_videos\\.venv\\Scripts\\python.exe audio\\extraer_risas.py --todos

Por qué guardar los cuadros y no solo los eventos: el umbral para decidir "esto es
risa" cambia los conteos. Con los cuadros guardados, las métricas se recalculan con
cualquier umbral sin volver a escuchar 100 horas de audio.

Modelo: YAMNet (AudioSet, 521 clases), el mismo que usa Analisis_videos/02b. Un
cuadro dura 0,975 s. Se guardan las sumas de probabilidad de cada grupo de clases
y el volumen (dB) de cada cuadro.

Salida: data/audio/frames/<id>.npz con t, risa, aplauso, habla, musica, db.
Lee data/audio/catalogo.csv (id, ruta, incluir). Se salta lo que ya existe.
"""
import csv
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CATALOGO = RAIZ / "data" / "audio" / "catalogo.csv"
FRAMES = RAIZ / "data" / "audio" / "frames"
VIDEOS = RAIZ.parent / "Analisis_videos"
MODELO = VIDEOS / "models" / "mediapipe" / "yamnet.tflite"
FFMPEG = (r"C:\Users\Home\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe")
SR = 16000
TROZO_S = 30

GRUPOS = {
    "risa": {"Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"},
    "aplauso": {"Applause", "Clapping", "Cheering"},
    "habla": {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue",
              "Conversation"},
    "musica": {"Music"},
}


def audio_de(ruta: Path, destino: Path) -> np.ndarray:
    subprocess.run([FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(ruta), "-vn", "-ac", "1", "-ar", str(SR), str(destino)],
                   check=True)
    with wave.open(str(destino), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def cuadros(audio: np.ndarray) -> dict:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.audio import AudioClassifier, AudioClassifierOptions, RunningMode
    from mediapipe.tasks.python.components.containers import AudioData

    opciones = AudioClassifierOptions(base_options=BaseOptions(model_asset_path=str(MODELO)),
                                      running_mode=RunningMode.AUDIO_CLIPS, max_results=-1)
    filas = {k: [] for k in ("t", *GRUPOS, "db")}
    trozo = TROZO_S * SR
    with AudioClassifier.create_from_options(opciones) as clf:
        for i in range(0, len(audio), trozo):
            seg = audio[i:i + trozo]
            if len(seg) < SR:
                break
            for r in clf.classify(AudioData.create_from_array(seg, SR)):
                t = i / SR + r.timestamp_ms / 1000.0
                por_clase = {c.category_name: c.score for c in r.classifications[0].categories}
                filas["t"].append(t)
                for g, clases in GRUPOS.items():
                    filas[g].append(sum(por_clase.get(c, 0.0) for c in clases))
                j0 = int((t - i / SR) * SR)
                x = seg[j0:j0 + int(0.975 * SR)]
                filas["db"].append(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)) if len(x) else -90.0)
    return {k: np.asarray(v, dtype=np.float32) for k, v in filas.items()}


def procesar(id_: str, ruta: Path) -> str:
    salida = FRAMES / f"{id_}.npz"
    if salida.exists():
        return "ya estaba"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = audio_de(ruta, Path(tmp) / "a.wav")
    c = cuadros(audio)
    np.savez_compressed(salida, **c)
    return f"{len(audio) / SR / 60:.0f} min de audio en {time.time() - t0:.0f} s"


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    with CATALOGO.open(encoding="utf-8", newline="") as f:
        filas = [r for r in csv.DictReader(f) if r["incluir"] == "si"]
    if "--id" in args:
        filas = [r for r in filas if r["id"] == args[args.index("--id") + 1]]
    for r in filas:
        try:
            print(f"{r['id']}: {procesar(r['id'], Path(r['ruta']))}", flush=True)
        except Exception as e:   # un archivo dañado no frena los demás
            print(f"{r['id']}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
