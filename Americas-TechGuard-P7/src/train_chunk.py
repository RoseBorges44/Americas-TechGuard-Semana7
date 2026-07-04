"""
train_chunk.py — Treino incremental em blocos (resiliente a timeout).
Uso: python3 train_chunk.py N_MORE_EPOCHS
Carrega checkpoint se existir, treina mais N épocas, salva. Datasets são
deterministas (mesma seed), garantindo continuidade.
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import URNN
from data import make_blumenau_terrain, make_dataset
from train import train_swp

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
CKPT = os.path.join(OUT, "ckpt.pt")
H = W = 40; T = 48; W_window = 6; K = 16
N_TRAIN, N_TEST = 12, 6

def get_data():
    dem, imp, dr = make_blumenau_terrain(H, W)
    Xtr, Dtr, Mtr, Itr = make_dataset(N_TRAIN, dem, imp, dr, T=T, W_window=W_window, seed=1)
    return dem, imp, dr, (Xtr, Dtr, Mtr)

def main():
    n_more = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    torch.manual_seed(0); np.random.seed(0)
    dem, imp, dr, train_ds = get_data()
    model = URNN(in_ch=train_ds[0].shape[2], base=16)
    history, done = [], 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        model.load_state_dict(ck["state"]); history = ck["history"]; done = ck["epochs"]
        print(f"[ckpt] retomando de {done} epocas (loss {history[-1]:.4f})")
    else:
        print("[ckpt] iniciando do zero")
    t0 = time.time()
    # lr decai conforme progresso global rumo a ~40 epocas
    h = train_swp(model, train_ds, "cpu", K=K, epochs=n_more, lr=2.0e-3, beta=5.0, log_every=2)
    history += h; done += n_more
    torch.save({"state": model.state_dict(), "history": history, "epochs": done}, CKPT)
    print(f"[ckpt] salvo: {done} epocas totais | +{n_more} em {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
