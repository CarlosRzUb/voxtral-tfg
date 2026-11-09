import os
import sys
import threading
import time

import gradio as gr
import numpy as np

import ejemplos
import qwen
import streaming
from streaming import Sesion, sr

VOXTRAL = "Voxtral Mini 4B Realtime"
QWEN = "Qwen3-ASR 1.7B"
RETARDOS = [160, 320, 480, 640, 800, 960, 1200]
BLOQUES = [1, 2, 4]
BLOQUE_S = 0.2
INACTIVIDAD_S = 15

cerrojo = threading.Lock()
activa = {"sesion": None, "modelo": VOXTRAL, "motor": None, "cargando": False}
OCUPADO = "Ahora mismo hay otra persona usando el demostrador. Prueba en unos segundos."


def nueva_sesion(retardo, bloque, idioma):
    if activa["modelo"] == QWEN:
        return qwen.SesionQwen(activa["motor"], bloque, idioma)
    return Sesion(retardo)


def ocupar(retardo, bloque, idioma):
    with cerrojo:
        if activa["cargando"]:
            return None
        actual = activa["sesion"]
        if actual is not None and time.time() - actual.ultima_actividad < INACTIVIDAD_S:
            return None
        if actual is not None:
            actual.terminar()
        sesion = nueva_sesion(retardo, bloque, idioma)
        activa["sesion"] = sesion
        return sesion


def liberar(sesion):
    with cerrojo:
        if activa["sesion"] is sesion:
            activa["sesion"] = None


def controles(modelo):
    return (
        gr.update(value=modelo, interactive=True),
        gr.update(visible=modelo == VOXTRAL),
        gr.update(visible=modelo == QWEN),
        gr.update(visible=modelo == QWEN),
    )


def cambiar_modelo(modelo):
    with cerrojo:
        libre = modelo != activa["modelo"] and not activa["cargando"] and activa["sesion"] is None
        if libre:
            activa["cargando"] = True
    if not libre:
        if modelo != activa["modelo"]:
            gr.Warning("No se puede cambiar de modelo mientras se está transcribiendo.")
        yield *controles(activa["modelo"]), ""
        return
    yield gr.update(interactive=False), gr.skip(), gr.skip(), gr.skip(), f"Cargando {modelo}… puede tardar unos minutos."
    try:
        if modelo == QWEN:
            streaming.descargar()
            activa["motor"] = qwen.Motor()
        else:
            activa["motor"].cerrar()
            activa["motor"] = None
            streaming.cargar()
        activa["modelo"] = modelo
    except Exception:
        gr.Warning(f"No se ha podido cargar {modelo}.")
        if streaming.model is None:
            streaming.cargar()
            activa["modelo"] = VOXTRAL
    finally:
        activa["cargando"] = False
    yield *controles(activa["modelo"]), ""


def tarjetas(sesion=None, wer=None):
    m = sesion.metricas() if sesion else dict(retardo_ms=None, computo_ms=0, rtf=0, segundos=0)
    datos = [
        ("Retardo", f"{m['retardo_ms']} ms" if m["retardo_ms"] else "–"),
        ("Cómputo por bloque", f"{m['computo_ms']:.0f} ms" if m["computo_ms"] else "–"),
        ("RTF", f"{m['rtf']:.2f}".replace(".", ",") if m["rtf"] else "–"),
        ("Audio procesado", f"{m['segundos']:.1f} s".replace(".", ",") if m["segundos"] else "–"),
    ]
    if wer is not None:
        datos.append(("WER", f"{wer:.2f} %".replace(".", ",")))
    celdas = "".join(f"<div class='tarjeta'><span>{k}</span><b>{v}</b></div>" for k, v in datos)
    return f"<div class='tarjetas'>{celdas}</div>"


def a_float(muestras):
    if np.issubdtype(muestras.dtype, np.integer):
        muestras = muestras.astype(np.float32) / np.iinfo(muestras.dtype).max
    muestras = muestras.astype(np.float32)
    if muestras.ndim > 1:
        muestras = muestras.mean(axis=1)
    return muestras


def empezar(retardo, bloque, idioma):
    sesion = ocupar(retardo, bloque, idioma)
    if sesion is None:
        gr.Warning(OCUPADO)
        return None, "", tarjetas()
    return sesion, "", tarjetas(sesion)


def escuchar(fragmento, sesion):
    if fragmento is None or sesion is None or sesion.terminada:
        return gr.skip(), gr.skip()
    sr_entrada, muestras = fragmento
    sesion.alimentar(a_float(muestras), sr_entrada)
    if isinstance(sesion, qwen.SesionQwen) and sesion.completa:
        gr.Warning(f"Qwen admite como máximo {qwen.LIMITE_S} s seguidos. Para seguir, vuelve a grabar.")
        texto = sesion.terminar()
        liberar(sesion)
        return texto, tarjetas(sesion)
    return sesion.leer(), tarjetas(sesion)


def parar(sesion):
    if sesion is None:
        return None, gr.skip(), gr.skip()
    texto = sesion.terminar()
    liberar(sesion)
    return None, texto, tarjetas(sesion)


def reiniciar(sesion):
    if sesion is not None:
        sesion.terminar()
        liberar(sesion)
    return None, "", tarjetas()


def transcribir_ejemplo(opcion, retardo, bloque):
    sesion = ocupar(retardo, bloque, ejemplos.idioma(opcion))
    if sesion is None:
        raise gr.Error(OCUPADO)
    try:
        y, referencia = ejemplos.cargar(opcion, sr)
        audio = str(ejemplos.ruta(opcion) / "audio_clip_diarization.m4a")
        yield gr.Audio(value=audio, autoplay=True), "", "", tarjetas(sesion)
        n = int(BLOQUE_S * sr)
        t0 = time.perf_counter()
        for i in range(0, len(y), n):
            sesion.alimentar(y[i:i + n])
            espera = t0 + (i + n) / sr - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
            yield gr.skip(), sesion.leer(), gr.skip(), tarjetas(sesion)
        texto = sesion.terminar()
        yield gr.skip(), texto, referencia, tarjetas(sesion, ejemplos.wer(referencia, texto))
    finally:
        sesion.terminar()
        liberar(sesion)


tema = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
)

css = """
.gradio-container {max-width: 1100px !important; margin: 0 auto;}
#cabecera h1 {font-size: 1.9rem; margin-bottom: 0.2rem;}
.tarjetas {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;}
.tarjeta {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 14px; background: var(--block-background-fill);}
.tarjeta span {display: block; font-size: 0.8rem; color: var(--body-text-color-subdued);}
.tarjeta b {font-size: 1.35rem; font-variant-numeric: tabular-nums;}
#transcripcion textarea {font-size: 1.05rem; line-height: 1.55;}
"""

with gr.Blocks(title="Transcripción en tiempo real") as demo:
    gr.Markdown("# Transcripción automática del habla en tiempo real", elem_id="cabecera")
    with gr.Row():
        modelo = gr.Dropdown([VOXTRAL, QWEN], value=VOXTRAL, label="Modelo")
        retardo = gr.Dropdown(RETARDOS, value=480, label="Retardo de transcripción (ms)")
        bloque = gr.Dropdown(BLOQUES, value=1, label="Tamaño de bloque (s)", visible=False)
    aviso = gr.Markdown()

    with gr.Tabs():
        with gr.Tab("Micrófono"):
            estado = gr.State(None)
            with gr.Row():
                with gr.Column(scale=2):
                    idioma = gr.Dropdown(list(qwen.IDIOMAS), value="Español", label="Idioma", visible=False)
                    micro = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Micrófono")
                    boton_reiniciar = gr.Button("Reiniciar", variant="secondary")
                with gr.Column(scale=3):
                    texto = gr.Textbox(label="Transcripción", lines=12, max_lines=12, autoscroll=True,
                                       buttons=["copy"], elem_id="transcripcion")
            metricas = gr.HTML(tarjetas())

        with gr.Tab("Audios de Europarl-ST"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    opcion = gr.Dropdown(ejemplos.opciones(), value=ejemplos.opciones()[0], label="Intervención")
                    with gr.Row():
                        boton_transcribir = gr.Button("Transcribir", variant="primary")
                        boton_parar = gr.Button("Parar", variant="stop")
                    reproductor = gr.Audio(label="Audio original", interactive=False)
                with gr.Column(scale=3):
                    texto_ejemplo = gr.Textbox(label="Transcripción", lines=8, max_lines=8, autoscroll=True,
                                               buttons=["copy"])
                    referencia = gr.Textbox(label="Referencia de Europarl-ST", lines=5, max_lines=5)
            metricas_ejemplo = gr.HTML(tarjetas())

    modelo.input(cambiar_modelo, [modelo], [modelo, retardo, bloque, idioma, aviso])
    demo.load(lambda: controles(activa["modelo"]), None, [modelo, retardo, bloque, idioma])

    micro.start_recording(empezar, [retardo, bloque, idioma], [estado, texto, metricas])
    micro.stream(escuchar, [micro, estado], [texto, metricas], stream_every=BLOQUE_S, time_limit=None)
    micro.stop_recording(parar, [estado], [estado, texto, metricas])
    boton_reiniciar.click(reiniciar, [estado], [estado, texto, metricas])

    evento = boton_transcribir.click(
        transcribir_ejemplo, [opcion, retardo, bloque], [reproductor, texto_ejemplo, referencia, metricas_ejemplo]
    )
    boton_parar.click(None, cancels=[evento])

if __name__ == "__main__":
    usuario, clave = os.environ.get("DEMO_USUARIO"), os.environ.get("DEMO_CLAVE")
    compartir = "--share" in sys.argv
    if compartir and not (usuario and clave):
        sys.exit("Para compartir el enlace hay que definir DEMO_USUARIO y DEMO_CLAVE.")
    streaming.cargar()
    demo.queue().launch(
        theme=tema,
        css=css,
        share=compartir,
        auth=(usuario, clave) if compartir else None,
        allowed_paths=[str(ejemplos.RAIZ)],
    )
