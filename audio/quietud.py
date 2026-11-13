"""
¿Se queda quieto el comediante antes del remate? La otra idea de TIC-TALK, medida en los shows de
Andrés donde está solo en tarima (Ansioso y el set de La Korte).

Paso 1 (venv de Analisis_videos, necesita mediapipe y cv2):
  <venv>\\Scripts\\python.exe audio/quietud.py --pose
Lee el video a 4 cuadros por segundo, detecta la postura de quien está en tarima y guarda cuánto se
mueve entre un cuadro y el siguiente (hombros, codos, muñecas, caderas), dividido por el largo del torso
para que no dependa de qué tan lejos está la cámara. -> data/audio/pose/<id>.npz

Paso 2 (cualquier Python con numpy):
  python audio/quietud.py
Compara el movimiento en los 3 s antes de cada reacción fuerte contra momentos al azar del mismo
show en que no hay reacción cerca. Si antes de las risas se mueve menos que al azar, hay quietud
antes del remate.

Límites: es correlación. Y "antes de la reacción" no es exactamente "antes del remate": el público
reacciona medio segundo o un segundo después de la última palabra.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
POSE = RAIZ / "data" / "audio" / "pose"
VIDEOS = RAIZ.parent / "Analisis_videos"
FFMPEG = (r"C:\Users\Home\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe")
FPS = 4
SHOWS = {  # id del catálogo -> (video, alto al que se escala, tarima: la nariz por encima de esta fracción del alto)
    "alta-comedia-ansioso-2025-08-22": (VIDEOS / "Videos_raw" / "IMG_3229.MOV", 1280, 0.62),
    "andres-loaiza-0731": (Path(r"D:\ANDRES\Stand up\Andres Loaiza\0731.mp4"), 720, 0.75),
}
PUNTOS = [11, 12, 13, 14, 15, 16, 23, 24]   # hombros, codos, muñecas, caderas
ANTES = (-3.0, -0.5)
LEJOS_S = 6.0         # un momento "al azar" no puede tener una reacción a menos de esto


def tamano(video: Path, alto: int) -> tuple[int, int]:
    """Ancho y alto ya rotados (los MOV del celular vienen girados 90°)."""
    ffprobe = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "default=nw=1", str(video)],
                       capture_output=True, text=True).stdout
    vals = dict(l.split("=") for l in r.split() if "=" in l)
    w, h = int(vals["width"]), int(vals["height"])
    if abs(int(float(vals.get("rotation", 0)))) == 90:
        w, h = h, w
    ancho = int(round(w * alto / h / 2) * 2)
    return ancho, alto


def extraer_pose(id_: str):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    video, alto, tarima = SHOWS[id_]
    ancho, alto = tamano(video, alto)
    modelo = VIDEOS / "models" / "mediapipe" / "pose_landmarker_full.task"
    opciones = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(modelo)),
                                     running_mode=RunningMode.VIDEO, num_poses=4,
                                     min_pose_detection_confidence=0.4, min_tracking_confidence=0.4)
    proc = subprocess.Popen([FFMPEG, "-nostdin", "-v", "error", "-i", str(video), "-vf",
                             f"fps={FPS},scale={ancho}:{alto}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    n_bytes = ancho * alto * 3
    ts, puntos, torso = [], [], []
    i = 0
    with PoseLandmarker.create_from_options(opciones) as lm:
        while True:
            buf = proc.stdout.read(n_bytes)
            if len(buf) < n_bytes:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(buf, np.uint8).reshape(alto, ancho, 3))
            res = lm.detect_for_video(img, int(i * 1000 / FPS))
            # quien está en tarima: la nariz más alta por encima del público sentado
            cand = [p for p in res.pose_landmarks if p[0].y < tarima]
            if cand:
                p = min(cand, key=lambda q: q[0].y)
                xy = np.array([[p[k].x * ancho, p[k].y * alto] for k in PUNTOS])
                vis = np.array([p[k].visibility or 0 for k in PUNTOS])
                xy[vis < 0.3] = np.nan
                largo = np.nanmean([np.hypot(*(xy[0] - xy[6])), np.hypot(*(xy[1] - xy[7]))])
            else:
                xy, largo = np.full((len(PUNTOS), 2), np.nan), np.nan
            ts.append(i / FPS)
            puntos.append(xy)
            torso.append(largo)
            i += 1
            if i % (FPS * 600) == 0:
                print(f"  {id_}: {i / FPS / 60:.0f} min", flush=True)
    proc.wait()
    POSE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POSE / f"{id_}.npz", t=np.array(ts), puntos=np.array(puntos), torso=np.array(torso))
    det = np.isfinite(np.array(torso)).mean()
    print(f"{id_}: {i} cuadros, comediante detectado en el {100 * det:.0f} %", flush=True)


def movimiento(d: dict) -> tuple[np.ndarray, dict]:
    """(t, {"caminar": ..., "gesticular": ...}) en largos de torso por segundo, suavizado con mediana de 1 s.
    Caminar = centro de caderas; gesticular = muñecas. Mismo criterio que Analisis_videos/scripts/tictalk.py."""
    p, esc = d["puntos"], np.nanmedian(d["torso"])
    suave = lambda v: np.array([np.nanmedian(v[max(0, j - 2):j + 2]) if np.isfinite(v[max(0, j - 2):j + 2]).any()
                                else np.nan for j in range(len(v))])
    cadera = np.nanmean(p[:, 6:8, :], axis=1)
    return d["t"][1:], {
        "caminar": suave(np.linalg.norm(np.diff(cadera, axis=0), axis=1) / esc * FPS),
        "gesticular": suave(np.nanmean(np.linalg.norm(np.diff(p[:, 4:6, :], axis=0), axis=2), axis=1) / esc * FPS)}


def analizar(id_: str, semilla: int = 7):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import metricas as M
    c = dict(np.load(M.FRAMES / f"{id_}.npz"))
    t0, t1 = M.ventana(c["t"], c["habla"])
    m = (c["t"] >= t0) & (c["t"] <= t1)
    p = c["risa"][m] + c["aplauso"][m]
    ev = M.eventos(c["t"][m], p, max(M.UMBRAL_PISO, float(np.percentile(p, M.PERCENTIL))), float(np.median(np.diff(c["t"]))))
    ini = np.array([a for a, _ in ev])
    masas = np.array([p[(c["t"][m] >= a) & (c["t"][m] < b)].sum() for a, b in ev])
    fuertes = ini[masas >= np.percentile(masas, 67)]
    libres = np.array([x for x in np.arange(t0 + 5, t1 - 5, 0.5) if np.min(np.abs(ini - x)) > LEJOS_S])
    t, series = movimiento(dict(np.load(POSE / f"{id_}.npz")))
    rng = np.random.default_rng(semilla)
    fila = {"id": id_, "reacciones_fuertes": int(len(fuertes)),
            "detectado_pct": round(100 * float(np.isfinite(dict(np.load(POSE / f"{id_}.npz"))["torso"]).mean()), 1)}
    for nombre, v in series.items():
        media = lambda xs: np.array([np.nanmean(v[(t >= x + ANTES[0]) & (t < x + ANTES[1])])
                                     if np.isfinite(v[(t >= x + ANTES[0]) & (t < x + ANTES[1])]).sum() >= 3 else np.nan
                                     for x in xs])
        antes, azar = media(fuertes), media(libres)
        antes, azar = antes[np.isfinite(antes)], azar[np.isfinite(azar)]
        sorteo = np.array([rng.choice(azar, len(antes)).mean() for _ in range(4000)])
        fila[f"{nombre}_proporcion"] = round(float(antes.mean() / azar.mean()), 2)
        fila[f"{nombre}_p_mas_quieto"] = round(float((sorteo <= antes.mean()).mean()), 3)
    return fila


if __name__ == "__main__":
    if "--pose" in sys.argv:
        for id_ in SHOWS:
            if not (POSE / f"{id_}.npz").exists():
                extraer_pose(id_)
    else:
        import pandas as pd
        sys.stdout.reconfigure(encoding="utf-8")
        filas = [analizar(id_) for id_ in SHOWS if (POSE / f"{id_}.npz").exists()]
        d = pd.DataFrame(filas)
        d.to_parquet(RAIZ / "data" / "data_frame" / "df_audio_quietud.parquet", index=False)
        pd.set_option("display.width", 220)
        print(d.to_string(index=False))
