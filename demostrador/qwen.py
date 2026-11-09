import os
import secrets
import subprocess
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import soxr

PYTHON = Path.home() / "venvs" / "qwen" / "bin" / "python"
SR = 16000
LIMITE_S = 90
IDIOMAS = {"Automático": None, "Español": "Spanish", "Inglés": "English", "Francés": "French", "Alemán": "German"}


class Motor:
    def __init__(self):
        clave = secrets.token_hex(16)
        with Listener(("127.0.0.1", 0), authkey=clave.encode()) as oyente:
            self.proceso = subprocess.Popen(
                [str(PYTHON), "-W", "ignore", str(Path(__file__).with_name("qwen_proceso.py")),
                 str(oyente.address[1]), clave],
                env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"},
            )
            self.conexion = oyente.accept()
        self.envio = threading.Lock()
        try:
            if self.conexion.recv() != "listo":
                raise RuntimeError("Qwen no ha arrancado")
        except Exception:
            self.cerrar()
            raise

    def enviar(self, mensaje):
        with self.envio:
            self.conexion.send(mensaje)

    def cerrar(self):
        try:
            self.enviar(None)
            self.proceso.wait(timeout=30)
        except Exception:
            self.proceso.kill()
            self.proceso.wait()
        self.conexion.close()


class SesionQwen:
    def __init__(self, motor, bloque_s=1.0, idioma="Automático"):
        self.motor = motor
        self.bloque_s = bloque_s
        self.retardo_ms = int(bloque_s * 1000)
        self.remuestreo = None
        self.recibido = 0
        self.completa = False
        self.texto = ""
        self.computo = 0.0
        self.bloques = 0
        self.error = None
        self.terminada = False
        self.cierre = threading.Lock()
        self.ultima_actividad = time.time()
        motor.enviar({"idioma": IDIOMAS[idioma], "bloque": bloque_s})
        self.hilo = threading.Thread(target=self._escuchar, daemon=True)
        self.hilo.start()

    def _escuchar(self):
        try:
            while True:
                tipo, self.texto, self.computo, self.bloques = self.motor.conexion.recv()
                if tipo == "fin":
                    return
        except Exception as e:
            self.error = e

    def alimentar(self, muestras, sr_entrada=SR):
        if self.terminada or self.completa:
            return
        muestras = np.asarray(muestras, dtype=np.float32)
        if sr_entrada != SR:
            if self.remuestreo is None:
                self.remuestreo = soxr.ResampleStream(sr_entrada, SR, 1, dtype="float32")
            muestras = self.remuestreo.resample_chunk(muestras)
        muestras = muestras[:LIMITE_S * SR - self.recibido]
        self.recibido += len(muestras)
        self.completa = self.recibido >= LIMITE_S * SR
        self.motor.enviar(muestras.tobytes())
        self.ultima_actividad = time.time()

    def leer(self):
        return self.texto

    def terminar(self):
        with self.cierre:
            if not self.terminada:
                self.terminada = True
                if self.remuestreo is not None and not self.completa:
                    self.motor.enviar(self.remuestreo.resample_chunk(np.zeros(0, np.float32), last=True).tobytes())
                self.motor.enviar("fin")
                self.hilo.join()
        return self.texto

    def metricas(self):
        segundos = self.bloques * self.bloque_s
        return dict(
            retardo_ms=self.retardo_ms,
            computo_ms=1000 * self.computo / self.bloques if self.bloques else 0.0,
            rtf=self.computo / segundos if segundos else 0.0,
            segundos=segundos,
        )
