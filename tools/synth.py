"""Genera las voces de «Mar muerto» con Chatterbox (código abierto, MIT) afinado para es-ES, en CPU."""
import os, sys, json, random, time
import numpy as np
import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'space'))
from chatterbox.src.chatterbox.tts import ChatterboxTTS  # noqa: E402

take = int(sys.argv[1])
torch.set_num_threads(os.cpu_count() or 4)
t0 = time.time()
model = ChatterboxTTS.from_pretrained("cpu")
print(f"modelo cargado en {time.time() - t0:.0f}s, sr={model.sr}", flush=True)
lines = json.load(open(os.path.join(HERE, 'lines.json'), encoding='utf-8'))
os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
for L in lines:
    seed = 1000 * take + L['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ref = os.path.join(ROOT, 'ref', 'es_m1.flac' if L['who'] == 'idris' else 'es_es_f1.wav')
    t1 = time.time()
    with torch.inference_mode():
        wav = model.generate(L['text'], audio_prompt_path=ref, exaggeration=L['ex'], cfg_weight=L['cfg'],
                             temperature=L.get('temp', 0.8), language_id='es')
    a = wav.squeeze(0).cpu().numpy().astype(np.float32)
    sf.write(os.path.join(ROOT, 'out', f"{L['id']}_t{take}.wav"), a, model.sr)
    print(f"{L['id']} toma {take}: {len(a) / model.sr:.2f}s de audio en {time.time() - t1:.0f}s", flush=True)
print(f"TOTAL {time.time() - t0:.0f}s", flush=True)
