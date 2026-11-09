import sys
import time
from multiprocessing.connection import Client

REPO_ID = "Qwen/Qwen3-ASR-1.7B"


def main():
    conexion = Client(("127.0.0.1", int(sys.argv[1])), authkey=sys.argv[2].encode())

    import numpy as np
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.LLM(
        model=REPO_ID,
        gpu_memory_utilization=0.88,
        max_model_len=24576,
        max_new_tokens=1024,
    )
    for bloque in [1.0, 2.0, 4.0]:
        estado = model.init_streaming_state(language="Spanish", chunk_size_sec=bloque)
        model.streaming_transcribe(np.zeros(int(3 * bloque * 16000), np.float32), estado)
        model.finish_streaming_transcribe(estado)
    conexion.send("listo")

    while True:
        orden = conexion.recv()
        if orden is None:
            break
        if isinstance(orden, dict):
            estado = model.init_streaming_state(language=orden["idioma"], chunk_size_sec=orden["bloque"])
            computo, bloques = 0.0, 0
        elif orden == "fin":
            model.finish_streaming_transcribe(estado)
            conexion.send(("fin", estado.text, computo, bloques))
        else:
            antes = estado.chunk_id
            t0 = time.perf_counter()
            model.streaming_transcribe(np.frombuffer(orden, np.float32), estado)
            if estado.chunk_id > antes:
                computo += time.perf_counter() - t0
                bloques += estado.chunk_id - antes
                conexion.send(("texto", estado.text, computo, bloques))


if __name__ == "__main__":
    main()
