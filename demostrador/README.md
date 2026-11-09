# Demostrador

Transcripción en tiempo real con Voxtral Mini 4B Realtime sobre una RTX 3060, con interfaz en Gradio.

Se ejecuta en WSL2 (Ubuntu 24.04). En Windows el modelo no llega a tiempo real: tarda unos 87 ms por cada bloque de 80 ms de audio, frente a unos 65 ms en WSL.

## Arrancar

```bash
cd /mnt/d/carru/voxtral-tfg/demostrador
~/venvs/voxtral/bin/python app.py
```

Se abre en http://localhost:7860.

## Compartir

```bash
DEMO_USUARIO=usuario DEMO_CLAVE=contraseña ~/venvs/voxtral/bin/python app.py --share
```

Genera un enlace `https://….gradio.live` protegido con usuario y contraseña. El enlace:

- solo funciona mientras el ordenador esté encendido y la app arrancada;
- caduca a la semana;
- es HTTPS, así que el navegador permite usar el micrófono.

Solo admite una sesión a la vez.

## Ficheros

- `streaming.py`: transcripción en streaming, con el mismo corte de bloques que `notebooks_latencia/voxtral_idiomas.ipynb`.
- `qwen.py` y `qwen_proceso.py`: Qwen3-ASR en streaming. Corre en un proceso aparte con `~/venvs/qwen`, porque necesita otras versiones de torch y transformers.
- `ejemplos.py`: intervenciones de Europarl-ST y cálculo del WER.
- `app.py`: interfaz.

## Modelos

Los dos modelos no caben a la vez en los 12 GB de la GPU, así que al cambiar de modelo se descarga uno y se carga el otro. Con los modelos en un disco mecánico el cambio tarda varios minutos, así que conviene dejar elegido el modelo antes de enseñar la app. Qwen admite como máximo 90 s seguidos con el micrófono.
