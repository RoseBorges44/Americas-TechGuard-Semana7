"""
run_experiment.py — Experimento completo (treino SWP + avaliação + persistência)
Salva: pesos do modelo, histórico de perda, métricas no teste, e arrays para figuras.
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import URNN
from data import make_blumenau_terrain, make_dataset
from train import train_swp, rollout, compute_metrics, DEPTH_SCALE

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    torch.manual_seed(0); np.random.seed(0)
    device = "cpu"
    H = W = 40
    T = 48
    W_window = 6
    K = 16
    EPOCHS = 34
    N_TRAIN, N_TEST = 12, 6

    print("[1] Gerando terreno de Blumenau e datasets...")
    dem, imp, dr = make_blumenau_terrain(H, W)
    Xtr, Dtr, Mtr, Itr = make_dataset(N_TRAIN, dem, imp, dr, T=T, W_window=W_window, seed=1)
    Xte, Dte, Mte, Ite = make_dataset(N_TEST, dem, imp, dr, T=T, W_window=W_window, seed=999)
    print(f"    treino: {Xtr.shape} | teste: {Xte.shape}")

    print("[2] Treinando U-RNN reduzido com paradigma SWP...")
    model = URNN(in_ch=Xtr.shape[2], base=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    parametros: {n_params:,} | janela SWP K={K} | epocas={EPOCHS}")
    t0 = time.time()
    history = train_swp(model, (Xtr, Dtr, Mtr), device, K=K, epochs=EPOCHS, lr=2.5e-3, beta=5.0, log_every=4)
    train_time = time.time() - t0
    print(f"    tempo de treino: {train_time:.1f}s")

    print("[3] Avaliando no conjunto de teste...")
    all_metrics = []
    infer_times = []
    for i in range(N_TEST):
        t0 = time.time()
        pred = rollout(model, Xte[i], device)
        infer_times.append(time.time() - t0)
        m = compute_metrics(pred, Dte[i])
        all_metrics.append(m)
    agg = {k: float(np.nanmean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    agg_std = {k: float(np.nanstd([m[k] for m in all_metrics])) for k in all_metrics[0]}
    mean_infer = float(np.mean(infer_times))
    print(f"    metricas (media+-dp): "
          f"MAE={agg['MAE']:.4f}+-{agg_std['MAE']:.4f}  "
          f"RMSE={agg['RMSE']:.4f}+-{agg_std['RMSE']:.4f}  "
          f"CSI={agg['CSI']:.3f}+-{agg_std['CSI']:.3f}  "
          f"PR2={agg['PR2']:.3f}+-{agg_std['PR2']:.3f}")
    print(f"    tempo de inferencia/evento: {mean_infer*1000:.0f} ms (6h de evento)")

    print("[4] Salvando artefatos...")
    torch.save(model.state_dict(), os.path.join(OUT, "urnn_blumenau.pt"))
    # Predições de 2 eventos para figuras
    ev_idx = [0, 2]
    saves = {}
    for k in ev_idx:
        saves[f"pred_{k}"] = rollout(model, Xte[k], device)
        saves[f"gt_{k}"] = Dte[k]
        saves[f"inten_{k}"] = Ite[k]
    np.savez_compressed(os.path.join(OUT, "arrays.npz"),
                        dem=dem, imp=imp, dr=dr,
                        history=np.array(history),
                        **saves)
    results = {
        "n_params": n_params, "K": K, "epochs": EPOCHS, "T": T, "grid": [H, W],
        "n_train": N_TRAIN, "n_test": N_TEST, "train_time_s": train_time,
        "infer_time_ms": mean_infer * 1000, "depth_scale": DEPTH_SCALE,
        "metrics_mean": agg, "metrics_std": agg_std,
        "per_event": all_metrics,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("    OK -> outputs/urnn_blumenau.pt, arrays.npz, results.json")
    print("[DONE]")

if __name__ == "__main__":
    main()
