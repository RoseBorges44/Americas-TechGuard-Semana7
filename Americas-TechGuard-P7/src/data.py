"""
data.py — Gerador de dados sintéticos de inundação urbana (contexto Blumenau/SC)
=================================================================================
O artigo U-RNN usa o modelo hidrodinâmico MIKE+ como "ground truth" para treinar
a rede. Como não temos licença do MIKE+ nem o dataset de 2 m de Shenzhen (volumoso
e em GPU), construímos um SURROGATE: um modelo de inundação por AUTÔMATO CELULAR
fisicamente plausível (chuva -> escoamento superficial -> roteamento por gravidade
-> drenagem), que gera sequências espaço-temporais de profundidade. A U-RNN aprende
a EMULAR esse modelo numérico ~100x mais rápido — exatamente o paradigma do artigo.

Contexto: terreno sintético inspirado no Vale do Itajaí (Blumenau é uma cidade em
vale fluvial, historicamente sujeita a enchentes do rio Itajaí-Açu). Conecta com a
Semana 5 (NDVI Blumenau) e Semana 6 (HAND Blumenau).

LIMITAÇÃO ASSUMIDA: os dados NÃO são observações reais nem saída de um modelo
hidrodinâmico calibrado. São sintéticos e servem para demonstrar o PIPELINE e a
ARQUITETURA, não para inferência operacional sobre Blumenau.
"""

import numpy as np


def make_blumenau_terrain(H=48, W=48, seed=7):
    """Cria DEM sintético tipo vale fluvial + mapas de impermeabilização e bueiros.

    Retorna (dem, impervious, drain) normalizados, todos em [0, 1] aprox.
    - dem: vale com canal fluvial sinuoso (baixo) cercado por encostas (alto)
    - impervious: maior no núcleo urbano (centro/canal), menor nas encostas
    - drain: pontos de captação de drenagem (bueiros) na malha urbana
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")

    # Canal fluvial sinuoso (rio Itajaí-Açu estilizado), atravessando o domínio
    river_x = 0.5 + 0.18 * np.sin(2.2 * np.pi * yy)
    dist_river = np.abs(xx - river_x)
    valley = 1.0 - np.exp(-(dist_river ** 2) / (2 * 0.10 ** 2))  # baixo no rio, sobe nas margens

    # Encostas/morros ao redor (Blumenau é cercada por morros)
    hills = 0.35 * (np.sin(3 * np.pi * xx) ** 2) * (yy ** 0.5)
    noise = 0.04 * rng.standard_normal((H, W))
    noise = _smooth(noise, 2)

    dem = 0.85 * valley + hills + noise
    dem = (dem - dem.min()) / (dem.max() - dem.min())  # min-max -> [0,1]

    # Impermeabilização: alta no fundo do vale (cidade), baixa nas encostas
    impervious = np.clip(1.0 - valley * 1.1, 0, 1)
    impervious = _smooth(impervious, 1)
    impervious = (impervious - impervious.min()) / (impervious.max() - impervious.min() + 1e-9)

    # Bueiros: pontos esparsos na área urbana (baixa cota + alta impermeabilização)
    drain = np.zeros((H, W))
    urban = (dem < 0.4) & (impervious > 0.5)
    idx = np.argwhere(urban)
    if len(idx) > 0:
        pick = rng.choice(len(idx), size=min(18, len(idx)), replace=False)
        for p in idx[pick]:
            drain[p[0], p[1]] = 1.0
    drain = _smooth(drain, 1)
    drain = drain / (drain.max() + 1e-9)

    return dem.astype(np.float32), impervious.astype(np.float32), drain.astype(np.float32)


def _smooth(a, k):
    """Suavização box simples (sem dependências extras)."""
    if k <= 0:
        return a
    out = a.copy()
    for _ in range(k):
        out = (out
               + np.roll(out, 1, 0) + np.roll(out, -1, 0)
               + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0
    return out


def chicago_hyetograph(T, peak_frac, total_mm, dt_min=2.0, seed=None):
    """Hietograma sintético (perfil tipo Chicago) -> intensidade (mm/h) por passo.

    peak_frac: razão tempo-ao-pico (0.1..0.9), como no artigo (time-to-peak ratio).
    total_mm: lâmina total do evento (mm).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    tp = max(1.0, peak_frac * T)
    # Curva assimétrica: sobe rápido até o pico e desce mais lento (ou vice-versa)
    inten = np.where(
        t <= tp,
        np.exp(-((t - tp) ** 2) / (2 * (0.25 * tp) ** 2)),
        np.exp(-((t - tp) ** 2) / (2 * (0.45 * (T - tp) + 1) ** 2)),
    )
    inten += 0.05 * rng.random(T)  # flutuação (chuva observada flutua, como diz o artigo)
    inten = np.clip(inten, 0, None)
    # Escala para a lâmina total desejada (mm) dado dt
    depth_per_step = inten / inten.sum() * total_mm        # mm por passo
    intensity_mmh = depth_per_step / (dt_min / 60.0)       # mm/h
    return intensity_mmh.astype(np.float32)


def simulate_flood(dem, impervious, drain, intensity_mmh, dt_min=2.0,
                   diff=0.20, drain_rate=0.12, runoff_gain=0.8,
                   relief_m=15.0):
    """Autômato celular de inundação (surrogate do MIKE+).

    Para cada passo de tempo:
      1) Chuva efetiva = intensidade * coef. de escoamento (cresce com impermeabilização)
      2) Roteamento por gravidade: fluxo para vizinhos com menor cota da lâmina d'água
      3) Drenagem: bueiros + canal fluvial (vazão do rio) + contorno aberto nas bordas
    Retorna sequência de profundidade h[t] (T, H, W) em metros sintéticos.

    O DEM (normalizado em [0,1]) é escalado para `relief_m` metros para que o
    desnível do vale seja compatível com as lâminas d'água, fazendo a água
    CONCENTRAR nas cotas baixas (inundação localizada, como no artigo).
    """
    H, W = dem.shape
    dem_m = dem * relief_m                                   # relevo em metros
    T = len(intensity_mmh)
    h = np.zeros((H, W), dtype=np.float32)
    seq = np.zeros((T, H, W), dtype=np.float32)
    runoff_coef = 0.25 + runoff_gain * impervious
    dt_h = dt_min / 60.0

    # Canal fluvial = cotas mais baixas (escoa água para fora, como um rio)
    channel = (dem < np.quantile(dem, 0.12)).astype(np.float32)
    # Máscara de borda (contorno aberto)
    edge = np.zeros((H, W), dtype=np.float32)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 1.0

    for t in range(T):
        # 1) Entrada de chuva
        rain_m = (intensity_mmh[t] * dt_h) * 1e-3 * runoff_coef * 1.0
        h = h + rain_m

        # 2) Roteamento por gravidade
        wsurf = dem_m + h
        for ax, shift in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            grad = wsurf - np.roll(wsurf, shift, axis=ax)
            flux = np.clip(grad, 0, None) * diff
            flux = np.minimum(flux, h * 0.25)                # estabilidade
            h = h - flux + np.roll(flux, shift, axis=ax)
        h = np.clip(h, 0, None)

        # 3) Drenagem: bueiros + canal (vazão do rio) + contorno aberto
        h = h - drain * drain_rate * h
        h = h - channel * 0.35 * h                           # rio escoa para fora
        h = h - edge * 0.40 * h                              # água sai do domínio
        h = np.clip(h, 0, None)

        seq[t] = h
    return seq


def build_event(dem, impervious, drain, T, W_window, peak_frac, total_mm,
                dt_min=2.0, seed=None, wet_thresh=0.03):
    """Monta UM evento: entradas (T, C, H, W) + alvos (profundidade e máscara wet/dry).

    Canais de entrada (C = W_window + 1 + 3), todos min-max normalizados:
      - W_window mapas: janela deslizante de intensidade de chuva (mm/h) broadcast
      - 1 mapa: chuva acumulada (capta o "efeito integral" citado no artigo)
      - 3 mapas estáticos: DEM, impermeabilização, bueiros
    """
    H, Wd = dem.shape
    intensity = chicago_hyetograph(T, peak_frac, total_mm, dt_min, seed)
    depth = simulate_flood(dem, impervious, drain, intensity, dt_min)

    cum = np.cumsum(intensity)                      # chuva acumulada (sequência)
    inten_n = intensity / (intensity.max() + 1e-9)
    cum_n = cum / (cum.max() + 1e-9)

    C = W_window + 1 + 3
    X = np.zeros((T, C, H, Wd), dtype=np.float32)
    for t in range(T):
        ch = 0
        # janela de intensidade [t-W+1 .. t]
        for w in range(W_window):
            tt = t - (W_window - 1) + w
            val = inten_n[tt] if tt >= 0 else 0.0
            X[t, ch] = val
            ch += 1
        X[t, ch] = cum_n[t]; ch += 1
        X[t, ch] = dem; ch += 1
        X[t, ch] = impervious; ch += 1
        X[t, ch] = drain; ch += 1

    # Normaliza profundidade alvo para estabilidade de treino (escala por evento-base global)
    depth_target = depth.astype(np.float32)
    mask = (depth_target > wet_thresh).astype(np.float32)
    return X, depth_target, mask, intensity, depth


def make_dataset(n_events, dem, impervious, drain, T=60, W_window=6, dt_min=2.0,
                 seed=0, wet_thresh=0.03):
    """Gera n_events com chuvas variadas (pico e lâmina aleatórios)."""
    rng = np.random.default_rng(seed)
    Xs, Ds, Ms, intens = [], [], [], []
    for i in range(n_events):
        pf = rng.uniform(0.2, 0.7)
        tm = rng.uniform(60, 180)        # lâmina total (mm) — faixa "evento severo"
        X, D, M, inten, _ = build_event(dem, impervious, drain, T, W_window,
                                        pf, tm, dt_min, seed=1000 + i, wet_thresh=wet_thresh)
        Xs.append(X); Ds.append(D); Ms.append(M); intens.append(inten)
    return (np.stack(Xs), np.stack(Ds), np.stack(Ms), np.stack(intens))


if __name__ == "__main__":
    dem, imp, dr = make_blumenau_terrain(48, 48)
    X, D, M, inten, depth = build_event(dem, imp, dr, T=60, W_window=6,
                                        peak_frac=0.4, total_mm=140, seed=42)
    print("DEM range:", round(float(dem.min()), 3), round(float(dem.max()), 3))
    print("Entrada X:", X.shape, "| Profundidade:", D.shape, "| Mascara:", M.shape)
    print("Prof. max (m sintetico):", round(float(D.max()), 3),
          "| Fracao molhada no pico:", round(float(M[D.sum((1,2)).argmax()].mean()), 3))
