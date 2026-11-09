from pathlib import Path

import jiwer
import librosa
from whisper.normalizers.basic import BasicTextNormalizer

RAIZ = Path(__file__).resolve().parent.parent / "Europarl-ST"

RUTAS = {
    "Español": "es/en/test",
    "Inglés": "en/es/test",
    "Francés": "fr/en/test",
    "Alemán": "de/en/test",
}

EJEMPLOS = {
    "Español": ["en.20100210.29.3-529", "en.20081020.15.1-139"],
    "Inglés": ["en.20090914.23.1-076", "en.20090112.13.1-045"],
    "Francés": ["en.20091214.14.1-057", "en.20120611.19.1-081-000"],
    "Alemán": ["en.20090323.13.1-055", "en.20120214.21.2-456-000"],
}

normalizer = BasicTextNormalizer()


def opciones():
    return [f"{idioma} · {audio}" for idioma, audios in EJEMPLOS.items() for audio in audios]


def idioma(opcion):
    return opcion.split(" · ")[0]


def ruta(opcion):
    idioma, audio = opcion.split(" · ")
    return RAIZ / RUTAS[idioma] / audio


def cargar(opcion, sr):
    carpeta = ruta(opcion)
    y, _ = librosa.load(carpeta / "audio_clip_diarization.m4a", sr=sr)
    referencia = (carpeta / "transcription.tok").read_text(encoding="utf-8").strip()
    return y, referencia


def wer(referencia, hipotesis):
    return 100 * jiwer.wer(normalizer(referencia), normalizer(hipotesis))
