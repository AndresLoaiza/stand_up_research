"""
Arma data/audio/catalogo.csv: un renglón por especial, sin duplicados ni muestras.

La carpeta D:\\ANDRES\\Stand up tiene 124 videos, pero muchos son el mismo especial
en dos calidades, muestras de 1 minuto, extras, compilaciones o documentales.
Medir un especial dos veces le da doble peso a ese comediante; medir una
compilación mezcla años y públicos. Aquí se decide qué entra y por qué.

También agrega los shows de Alta Comedia que ya tienen audio extraído en
Analisis_videos (output/<show>/<fecha>/audio_mono16k.wav).

Uso:  python audio/catalogo.py      (imprime lo que decidió y escribe el CSV)
"""
import csv
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FUENTE = Path(r"D:\ANDRES\Stand up")
VIDEOS = RAIZ.parent / "Analisis_videos" / "output"
SALIDA = RAIZ / "data" / "audio" / "catalogo.csv"
FFPROBE = (r"C:\Users\Home\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
           r"\ffmpeg-8.1.1-full_build\bin\ffprobe.exe")
EXT = {".avi", ".mkv", ".mp4", ".mov"}
MIN_MIN = 15

# No son un especial de stand-up grabado en vivo de principio a fin.
FUERA = {
    "Inside the Actors Studio": "entrevista",
    "George's Best Stuff": "compilación de varios años",
    "Personal Favorites": "compilación de varios años",
    "40 Years of Comedy": "compilación de varios años",
    "Omit the Logic": "documental",
    "Just A Ride": "documental",
    "Carlin at Carnegie": "incompleto (le faltan 11 minutos)",
    "Sexie CD": "partido en dos; se usa la versión completa",
    "Deleted.Scenes": "extra",
    "The.News.From.Yesterday": "extra",
    "Ricky Gervais Live - Fame.avi": "audio AC3 dañado (se usa la otra copia, con MP3)",
}

# Título y año de cada especial, buscado en la ruta. El orden importa: la primera coincidencia gana.
ESPECIALES = [
    ("Bill Bailey", "Part Troll", 2004), ("Bill Burr", "Why Do I Do This", 2008), ("Bill Burr", "Let It Go", 2010),
    ("Bill Burr", "One Night Stand", 2005), ("Bill Cosby", "Himself", 1983),
    ("Bill Hicks", "One Night Stand", 1991), ("Bill Hicks", "Relentless", 1992), ("Bill Hicks", "Revelations", 1993),
    ("Chris Rock", "Big Ass Jokes", 1994), ("Chris Rock", "Bring the Pain", 1996),
    ("Chris Rock", "Bigger and Blacker", 1999), ("Chris Rock", "Never Scared", 2004),
    ("Chris Rock", "Kill the Messenger", 2008), ("Chris Rock", "Tamborine", 2018),
    ("Demetri Martin", "Person", 2007),
    ("Eddie Izzard", "Unrepeatable", 1994), ("Eddie Izzard", "Definite Article", 1996),
    ("Eddie Izzard", "Glorious", 1997), ("Eddie Izzard", "Dress To Kill", 1999), ("Eddie Izzard", "Sexie", 2003),
    ("Eddie Izzard", "Stripped", 2009), ("Eddie Izzard", "Live From Wembley", 2009),
    ("Eddie Izzard", "Force Majeure", 2014), ("Eddie Izzard", "Wunderbar", 2022),
    ("Eddie Murphy", "Delirious", 1983), ("Eddie Murphy", "Raw", 1987),
    ("Frankie Boyle", "Live", 2008),
    ("George Carlin", "at USC", 1977), ("George Carlin", "at Phoenix", 1978), ("George Carlin", "Carlin on Campus", 1984),
    ("George Carlin", "Playin' with Your Head", 1986), ("George Carlin", "What Am I Doing in New Jersey", 1988),
    ("George Carlin", "Doin' It Again", 1990), ("George Carlin", "Jammin' in New York", 1992),
    ("George Carlin", "Back in Town", 1996), ("George Carlin", "You Are All Diseased", 1999),
    ("George Carlin", "Complaints and Grievances", 2001), ("George Carlin", "Life Is Worth Losing", 2005),
    ("George Carlin", "It's Bad for Ya", 2008),
    ("Jeff Dunham", "Arguing with Myself", 2006), ("Jeff Dunham", "Spark Of Insanity", 2007),
    ("Jim Jefferies", "I Swear To God", 2009), ("Jim Jefferies", "Contraband", 2008),
    ("Jim Jefferies", "Fully Functional", 2012),
    ("Jimmy Carr", "2004 Jimmy Carr Live", 2004), ("Jimmy Carr", "Stand Up", 2005), ("Jimmy Carr", "Comedian", 2007),
    ("Jimmy Carr", "In Concert", 2008), ("Jimmy Carr", "Telling Jokes", 2009),
    ("Jimmy Carr", "Making People Laugh", 2010), ("Jimmy Carr", "Been Funny", 2011),
    ("Jimmy Carr", "Laughing and Joking", 2013),
    ("John Mulaney", "New In Town", 2012), ("John Waters", "This Filthy World", 2006),
    ("Louis C.K.", "Shameless", 2007), ("Louis C.K.", "Chewed Up", 2008), ("Louis C.K.", "Beacon", 2011),
    ("Louis C.K.", "Oh My God", 2013), ("Louis C.K.", "Comedy Store", 2015), ("Louis C.K.", "Sincerely", 2020),
    ("Louis C.K.", "Sorry", 2021), ("Louis C.K.", "Back.to.the.Garden", 2023), ("Louis C.K.", "Dolby", 2023),
    ("Mike Birbiglia", "My.Girlfriends.Boyfriend", 2013), ("Mitch Hedberg", "comedy.central.presents", 1999),
    ("Patrice O'Neal", "Killing.Is.Easy", 2021), ("Peter Kay", "Top Of The Tower", 2011),
    ("Peter Kay", "Tour That Didn't Tour", 2011),
    ("Richard Pryor", "Sunset Strip", 1982),
    ("Ricky Gervais", "Animals", 2003), ("Ricky Gervais", "politics", 2004), ("Ricky Gervais", "Fame", 2007),
    ("Ricky Gervais", "Science", 2010), ("Ricky Gervais", "Out.of.England.2", 2010),
    ("Robin Williams", "Live at the Met", 1986), ("Ron White", "Tater Salad", 2004),
    ("Rory Scovel", "Religion.Sex", 2024), ("Russell Peters", "Outsourced", 2006),
    ("Russell Peters", "Red.White.And.Brown", 2008), ("Sebastian Maniscalco", "Embarrassed", 2014),
]
# Entre dos copias del mismo especial: la de mejor fuente.
CALIDAD = ["1080p", "BluRay", "WEB", "720p", "HDTV", ".mkv", ".mp4"]

# Videos sueltos de la carpeta "Andres Loaiza", identificados mirando cuadros del video.
PROPIOS_VIDEO = {"0731": ("Set en La Korte, 31 jul 2025", "stand-up solo", 2025)}

# Shows propios ya procesados en Analisis_videos. "formato" importa: un panel o un show de
# juegos no se compara de igual a igual con un especial de una persona.
PROPIOS = {
    "ansioso/2025-08-22": ("Ansioso", "stand-up solo"),
    "nape/2026-05-02": ("No Apto para Ex", "crowdwork e impro a dos"),
    "nape/2026-07-04": ("No Apto para Ex", "crowdwork e impro a dos"),
    "fuera_de_charla/2026-05-14": ("Fuera de Charla", "panel"),
    "fuera_de_charla/2026-07-30": ("Fuera de Charla", "panel"),
    "hoy_nace_un_chiste/2026-07-17": ("Hoy Nace un Chiste", "varios comediantes"),
    "comedia_de_alto_riesgo/2026-10-15": ("Comedia de Alto Riesgo", "juegos de impro"),
}


def minutos(ruta: Path) -> float:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                        str(ruta)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) / 60
    except ValueError:
        return 0.0


def slug(*partes) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(str(p) for p in partes).lower()).strip("-")


def letras(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def puntaje(ruta: Path) -> int:
    return sum(len(CALIDAD) - i for i, q in enumerate(CALIDAD) if q.lower() in str(ruta).lower())


def decidir(ruta: Path, mins: float):
    """(comediante, especial, año, motivo_de_exclusión o '')."""
    # Ruta RELATIVA: la raíz se llama "Stand up" y casaba con el especial "Stand Up" de Jimmy Carr.
    texto = str(ruta.relative_to(FUENTE)).replace("_", " ")
    for clave, motivo in FUERA.items():
        if clave.lower() in texto.lower():
            return None, None, None, motivo
    if mins < MIN_MIN:
        return None, None, None, f"dura {mins:.0f} min (muestra, extra o archivo dañado)"
    carpeta = letras(ruta.relative_to(FUENTE).parts[0])
    for comediante, clave, anio in ESPECIALES:
        # 6 letras: separa "Bill Hicks" de "Bill Burr" y tolera "Peter Key" por "Peter Kay"
        if clave.lower().replace(".", " ") in texto.lower().replace(".", " ") and letras(comediante)[:6] == carpeta[:6]:
            return comediante, clave.replace(".", " ").strip(), anio, ""
    return None, None, None, "no identificado"


def main():
    filas, elegidos = [], {}
    for ruta in sorted(p for p in FUENTE.rglob("*") if p.is_file() and p.suffix.lower() in EXT):
        mins = minutos(ruta)
        if ruta.parts[len(FUENTE.parts)] == "Andres Loaiza":
            if mins >= MIN_MIN:
                especial, formato, anio = PROPIOS_VIDEO.get(ruta.stem, (ruta.stem, "por identificar", ""))
                filas.append({"id": slug("andres-loaiza", ruta.stem), "comediante": "Andrés Loaiza",
                              "especial": especial, "anio": anio, "grupo": "propio", "formato": formato,
                              "minutos": round(mins), "ruta": str(ruta), "incluir": "si", "nota": ""})
            continue
        comediante, especial, anio, motivo = decidir(ruta, mins)
        if motivo:
            print(f"  fuera  {motivo:<45} {ruta.relative_to(FUENTE)}")
            continue
        clave = slug(comediante, especial)
        previo = elegidos.get(clave)
        if previo and puntaje(Path(previo["ruta"])) >= puntaje(ruta):
            print(f"  fuera  {'copia de ' + clave:<45} {ruta.relative_to(FUENTE)}")
            continue
        if previo:
            print(f"  fuera  {'copia de ' + clave:<45} {Path(previo['ruta']).relative_to(FUENTE)}")
        elegidos[clave] = {"id": clave, "comediante": comediante, "especial": especial, "anio": anio,
                           "grupo": "referente", "formato": "especial" if mins >= 45 else "media hora",
                           "minutos": round(mins), "ruta": str(ruta), "incluir": "si", "nota": ""}
    filas += sorted(elegidos.values(), key=lambda r: (r["comediante"], r["anio"]))
    for clave, (show, formato) in PROPIOS.items():
        wav = VIDEOS / clave / "audio_mono16k.wav"
        if wav.exists():
            filas.append({"id": slug("alta-comedia", clave), "comediante": "Alta Comedia", "especial": show,
                          "anio": clave.split("/")[1][:4], "grupo": "propio", "formato": formato,
                          "minutos": round(minutos(wav)), "ruta": str(wav), "incluir": "si", "nota": clave})
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{len(filas)} especiales -> {SALIDA.relative_to(RAIZ)}")
    for r in filas:
        print(f"  {r['minutos']:>4} min  {r['grupo']:<9} {r['comediante']:<20} {r['anio']:<5} {r['especial']}")


if __name__ == "__main__":
    main()
