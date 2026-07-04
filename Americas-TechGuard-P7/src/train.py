"""
train.py — Treino com paradigma SWP, perdas e métricas (réplica reduzida do U-RNN)
==================================================================================
- Perdas: Focal BCE (classificação wet/dry) + MSE ponderada (regressão de profundidade),
  combinadas em multi-tarefa (Eq. 2-6 do artigo).
- Paradigma SWP (Sliding Window-based Pre-warming): a sequência é dividida em janelas
  de K passos; antes de cada janela há um PRE-WARMING gradient-free (inferência do passo
  0 até o início da janela) que fornece o estado inicial; só os K passos da janela
  guardam gradiente -> reduz memória de GPU (no artigo, até ~360x).
- Métricas: MAE, RMSE, CSI (extensão máxima), PR² (profundidade de pico).
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

DEPTH_SCALE = 3.0       # normaliza profundidade alvo (m) -> ~O(1) p/ estabilidade
WET_THRESH = 0.03       # 3 cm, limiar wet/dry (igual ao artigo)


# ----------------------------- Perdas -----------------------------
def focal_bce(logit, target, alpha=0.5, gamma=2.0):
    """Focal BCE (Lin et al., 2017): foca em células difíceis e mitiga o
    desbalanceamento wet/dry (poucas células molhadas)."""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


def weighted_mse(pred, target, mask_wet, lam=30.0):
    """MSE ponderada (Eq. 3-5): peso maior nas células molhadas (lam)."""
    wet = mask_wet
    dry = 1.0 - mask_wet
    nw = wet.sum().clamp(min=1.0)
    nd = dry.sum().clamp(min=1.0)
    l_wet = ((wet * (pred - target) ** 2).sum()) / nw
    l_dry = ((dry * (pred - target) ** 2).sum()) / nd
    return lam * l_wet + l_dry


# ----------------------------- Métricas -----------------------------
def compute_metrics(pred_seq, gt_seq, prob_seq=None):
    """pred_seq, gt_seq: (T, H, W) profundidade em metros. prob_seq: (T,H,W) prob. wet.
    A extensão da cheia (CSI) usa a MÁSCARA DE CLASSIFICAÇÃO (branch wet/dry, Fig. 2b
    do artigo); se prob_seq não for dada, usa limiar sobre a profundidade."""
    err = pred_seq - gt_seq
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # CSI sobre a extensão MÁXIMA de inundação (via branch de classificação)
    gt_max = gt_seq.max(axis=0)
    gw = gt_max > WET_THRESH
    if prob_seq is not None:
        pw = prob_seq.max(axis=0) > 0.5
    else:
        pw = pred_seq.max(axis=0) > WET_THRESH
    tp = np.logical_and(pw, gw).sum()
    fp = np.logical_and(pw, ~gw).sum()
    fn = np.logical_and(~pw, gw).sum()
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # PR²: R² sobre a profundidade de PICO (cells com pico > 0.1 cm)
    pred_max = pred_seq.max(axis=0)
    m = gt_max > 0.001
    if m.sum() > 5:
        x = gt_max[m]; y = pred_max[m]
        ss_res = np.sum((y - x) ** 2)
        ss_tot = np.sum((x - x.mean()) ** 2) + 1e-9
        pr2 = float(1 - ss_res / ss_tot)
    else:
        pr2 = float("nan")
    return {"MAE": mae, "RMSE": rmse, "CSI": csi, "PR2": pr2}


# ----------------------------- Inferência -----------------------------
@torch.no_grad()
def rollout(model, X, device):
    """Roda o modelo por toda a sequência (inferência). X: (T, C, H, W) np.
    Retorna (profundidade prevista (T,H,W) em metros, probabilidade wet (T,H,W))."""
    model.eval()
    T, C, H, W = X.shape
    xt = torch.from_numpy(X).to(device)
    st = model.init_states(1, H, W, device)
    out = np.zeros((T, H, W), dtype=np.float32)
    pout = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        _, prob, depth, st = model.forward_step(xt[t:t+1], st)
        out[t] = (depth[0, 0].cpu().numpy()) * DEPTH_SCALE
        pout[t] = prob[0, 0].cpu().numpy()
    return out, pout


# ----------------------------- Treino SWP -----------------------------
def train_swp(model, dataset, device, K=12, epochs=40, lr=2e-3, beta=1.0,
              log_every=5):
    """Treina o modelo com o paradigma SWP.

    dataset: (X, D, M) com X (N,T,C,H,W), D/M (N,T,H,W).
    K: tamanho da janela deslizante (passos que guardam gradiente).
    """
    X, D, M = dataset
    N, T, C, H, W = X.shape
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []

    Xt = torch.from_numpy(X).to(device)
    Dt = torch.from_numpy(D / DEPTH_SCALE).unsqueeze(2).to(device)   # (N,T,1,H,W)
    Mt = torch.from_numpy(M).unsqueeze(2).to(device)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        order = np.random.permutation(N)
        for i in order:
            st = model.init_states(1, H, W, device)
            # ----- SWP: percorre janelas de K passos -----
            s = 0
            while s < T:
                e = min(s + K, T)
                # PRE-WARMING gradient-free: reconstrói estado até o passo s
                # (aqui o estado já vem propagado da janela anterior, então o
                #  pre-warming explícito ocorre apenas se reiniciássemos; mantemos
                #  os estados propagados e os DESTACAMOS para cortar o caminho de
                #  gradiente entre janelas — equivalente ao corte do BPTT do SWP).
                st = model.detach_states(st)
                win_loss = 0.0
                for t in range(s, e):
                    logit, prob, depth, st = model.forward_step(Xt[i, t:t+1], st)
                    lc = focal_bce(logit, Mt[i, t:t+1])
                    lr_ = weighted_mse(depth, Dt[i, t:t+1], Mt[i, t:t+1])
                    win_loss = win_loss + beta * lc + lr_
                win_loss = win_loss / (e - s)
                opt.zero_grad()
                win_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(win_loss.detach())
                s = e
        sched.step()
        history.append(ep_loss / N)
        if ep == 0 or (ep + 1) % log_every == 0:
            print(f"  epoch {ep+1:3d}/{epochs} | loss {history[-1]:.4f} | lr {sched.get_last_lr()[0]:.1e}")
    return history


if __name__ == "__main__":
    # Probe de tempo: 1 época em poucos eventos para dimensionar o treino completo
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from model import URNN
    from data import make_blumenau_terrain, make_dataset

    device = "cpu"
    H = W = 48
    dem, imp, dr = make_blumenau_terrain(H, W)
    X, D, M, inten = make_dataset(4, dem, imp, dr, T=60, W_window=6, seed=1)
    model = URNN(in_ch=X.shape[2], base=20).to(device)
    t0 = time.time()
    train_swp(model, (X, D, M), device, K=12, epochs=1, log_every=1)
    print(f"1 epoca / 4 eventos: {time.time()-t0:.1f}s")
