import gc
import queue
import threading
import time

import numpy as np
import soxr
import torch
from transformers import TextIteratorStreamer, VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

REPO_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

torch.set_num_threads(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = VoxtralRealtimeProcessor.from_pretrained(REPO_ID)
model = None

sr = processor.feature_extractor.sampling_rate
hop = processor.feature_extractor.hop_length
win = processor.feature_extractor.win_length


class Sesion:
    def __init__(self, retardo_ms=480):
        processor.mistral_common_audio_config.transcription_delay_ms = retardo_ms
        self.retardo_ms = retardo_ms
        self.entrada = queue.Queue()
        self.streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
        self.remuestreo = None
        self.audio = np.zeros(0, dtype=np.float32)
        self.mel = processor.num_mel_frames_first_audio_chunk
        self.fin = False
        self.texto = ""
        self.bloques = 0
        self.computo = 0.0
        self.error = None
        self.terminada = False
        self.cierre = threading.Lock()
        self.ultima_actividad = time.time()
        self.hilo = threading.Thread(target=self._trabajar, daemon=True)
        self.hilo.start()

    def _recibir(self):
        muestras = self.entrada.get()
        if muestras is None:
            npad = processor.num_right_pad_tokens
            npad = npad() if callable(npad) else npad
            muestras = np.zeros(npad * processor.raw_audio_length_per_tok, np.float32)
            self.fin = True
        self.audio = np.concatenate([self.audio, muestras])

    def _trozos(self):
        while True:
            start = self.mel * hop - win // 2
            end = start + processor.num_samples_per_audio_chunk
            while end >= len(self.audio):
                if self.fin:
                    return
                self._recibir()
            t = processor(self.audio[start:end], is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
            self.mel += processor.audio_length_per_tok
            t0 = time.perf_counter()
            yield t.to(model.device, dtype=model.dtype).input_features
            self.computo += time.perf_counter() - t0
            self.bloques += 1

    def _trabajar(self):
        try:
            while len(self.audio) < processor.num_samples_first_audio_chunk:
                if self.fin:
                    return
                self._recibir()
            primero = processor(
                self.audio[:processor.num_samples_first_audio_chunk],
                is_streaming=True,
                is_first_audio_chunk=True,
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            trozos = self._trozos()

            def todos():
                yield primero.input_features
                yield from trozos

            with torch.no_grad():
                model.generate(
                    input_ids=primero.input_ids,
                    input_features=todos(),
                    num_delay_tokens=primero.num_delay_tokens,
                    streamer=self.streamer,
                )
        except Exception as e:
            self.error = e
        finally:
            self.streamer.end()

    def alimentar(self, muestras, sr_entrada=sr):
        if self.terminada:
            return
        muestras = np.asarray(muestras, dtype=np.float32)
        if sr_entrada != sr:
            if self.remuestreo is None:
                self.remuestreo = soxr.ResampleStream(sr_entrada, sr, 1, dtype="float32")
            muestras = self.remuestreo.resample_chunk(muestras)
        self.entrada.put(muestras)
        self.ultima_actividad = time.time()

    def leer(self):
        while True:
            try:
                trozo = self.streamer.text_queue.get_nowait()
            except queue.Empty:
                break
            if trozo is not None:
                self.texto += trozo
        return self.texto

    def terminar(self):
        with self.cierre:
            if not self.terminada:
                self.terminada = True
                if self.remuestreo is not None:
                    self.entrada.put(self.remuestreo.resample_chunk(np.zeros(0, np.float32), last=True))
                self.entrada.put(None)
                self.hilo.join()
        return self.leer()

    def metricas(self):
        segundos = self.bloques * processor.raw_audio_length_per_tok / sr
        return dict(
            retardo_ms=self.retardo_ms,
            computo_ms=1000 * self.computo / self.bloques if self.bloques else 0.0,
            rtf=self.computo / segundos if segundos else 0.0,
            segundos=segundos,
        )


def cargar():
    global model
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(REPO_ID).to(device)
    model.eval()
    s = Sesion()
    s.alimentar(np.zeros(10 * sr, np.float32))
    s.terminar()


def descargar():
    global model
    model = None
    gc.collect()
    torch.cuda.empty_cache()
