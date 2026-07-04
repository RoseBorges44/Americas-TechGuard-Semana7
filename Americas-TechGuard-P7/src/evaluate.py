"""evaluate.py — Carrega checkpoint, avalia no teste, salva métricas e arrays p/ figuras."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import URNN
from data import make_blumenau_terrain, make_dataset
from train import rollout, compute_metrics, DEPTH_SCALE

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
CKPT = os.path.join(OUT, "ckpt.pt")
H = W = 40; T = 48; W_window = 6; K = 16
N_TRAIN, N_TEST = 12, 6

def main():
    torch.manual_seed(0); np.random.seed(0)
    dem, imp, dr = make_blumenau_terrain(H, W)
    Xte, Dte, Mte, Ite = make_dataset(N_TEST, dem, imp, dr, T=T, W_window=W_window, seed=999)
    ck = torch.load(CKPT)
    model = URNN(in_ch=Xte.shape[2], base=16); model.load_state_dict(ck["state"])
    n_params = sum(p.numel() for p in model.parameters())

    mets, infer = [], []
    preds, probs = [], []
    for i in range(N_TEST):
        t0 = time.time(); pred, prob = rollout(model, Xte[i], "cpu"); infer.append(time.time()-t0)
        mets.append(compute_metrics(pred, Dte[i], prob)); preds.append(pred); probs.append(prob)
    agg = {k: float(np.nanmean([m[k] for m in mets])) for k in mets[0]}
    std = {k: float(np.nanstd([m[k] for m in mets])) for k in mets[0]}
    print(f"epocas={ck['epochs']}  MAE={agg['MAE']:.4f}+-{std['MAE']:.4f}  "
          f"RMSE={agg['RMSE']:.4f}+-{std['RMSE']:.4f}  CSI={agg['CSI']:.3f}+-{std['CSI']:.3f}  "
          f"PR2={agg['PR2']:.3f}+-{std['PR2']:.3f}  infer={np.mean(infer)*1000:.0f}ms")
    for i, m in enumerate(mets):
        print(f"   evento {i}: CSI={m['CSI']:.3f} PR2={m['PR2']:.3f} MAE={m['MAE']:.4f}")

    # Salva tudo para figuras
    ev_idx = [0, 2]
    saves = {}
    for k in ev_idx:
        saves[f"pred_{k}"] = preds[k]; saves[f"gt_{k}"] = Dte[k]
        saves[f"prob_{k}"] = probs[k]; saves[f"inten_{k}"] = Ite[k]
    np.savez_compressed(os.path.join(OUT, "arrays.npz"),
                        dem=dem, imp=imp, dr=dr, history=np.array(ck["history"]), **saves)
    torch.save(model.state_dict(), os.path.join(OUT, "urnn_blumenau.pt"))
    results = {"n_params": n_params, "K": K, "epochs": ck["epochs"], "T": T, "grid": [H, W],
               "n_train": N_TRAIN, "n_test": N_TEST, "infer_time_ms": float(np.mean(infer)*1000),
               "depth_scale": DEPTH_SCALE, "metrics_mean": agg, "metrics_std": std,
               "per_event": mets}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
