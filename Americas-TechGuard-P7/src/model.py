"""
model.py — Reimplementação didática e REDUZIDA do U-RNN
=========================================================
Baseado em: Cao et al. (2025), "U-RNN high-resolution spatiotemporal
nowcasting of urban flooding", Journal of Hydrology 659:133117.
DOI: 10.1016/j.jhydrol.2025.133117 | Repo oficial: github.com/holmescao/U-RNN

Esta versão reproduz os COMPONENTES CONCEITUAIS do artigo em escala reduzida,
para rodar em CPU/Colab sem GPU dedicada:
  - Skip-ConvGRU (Eq. 1 do artigo): reset gate, update gate e estado candidato,
    fundindo a entrada do bloco I_t, o estado de codificação He_t (skip) e o
    estado de decodificação anterior Hd_{t-1}.
  - Backbone U-like (encoder-decoder) com 3 níveis de resolução.
  - Heads desacopladas: classificação (wet/dry) + regressão (profundidade),
    usando a máscara de classificação para focar a regressão nas células molhadas.

Autora da adaptação: Rosemeri Borges (RoseBorges44) — Americas TechGuard, SENAI/SC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipConvGRUCell(nn.Module):
    """Célula Skip-ConvGRU (Eq. 1 do artigo).

    Diferentemente da ConvGRU clássica (Shi et al., 2017), a Skip-ConvGRU
    determina o estado de decodificação atual de cada célula a partir de TRÊS
    fontes: a saída do bloco anterior (I_t), o estado histórico de decodificação
    da própria célula (Hd_{t-1}, recorrência temporal) e o estado de codificação
    vindo do encoder no mesmo nível (He_t, skip connection).

        r_t  = sigma(W_r * [I_t, He_t, Hd_{t-1}] + b_r)        # reset gate
        z_t  = sigma(W_z * [I_t, He_t, Hd_{t-1}] + b_z)        # update gate
        d~_t = tanh (W_c * [I_t, He_t, (r_t . Hd_{t-1})] + b_h)  # candidato
        d_t  = (1 - z_t) . Hd_{t-1} + z_t . d~_t               # novo estado
    """

    def __init__(self, in_ch, skip_ch, hid_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        cat = in_ch + skip_ch + hid_ch
        # Gates de reset e update (concatenam I_t, He_t, Hd_{t-1})
        self.conv_r = nn.Conv2d(cat, hid_ch, kernel, padding=pad)
        self.conv_z = nn.Conv2d(cat, hid_ch, kernel, padding=pad)
        # Estado candidato (usa r_t . Hd_{t-1} no lugar de Hd_{t-1})
        self.conv_c = nn.Conv2d(in_ch + skip_ch + hid_ch, hid_ch, kernel, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x, skip, h):
        # x: I_t (entrada do bloco) | skip: He_t (0 no encoder) | h: Hd_{t-1}
        if skip is None:
            skip = torch.zeros(x.size(0), 0, x.size(2), x.size(3), device=x.device)
        comb = torch.cat([x, skip, h], dim=1)
        r = torch.sigmoid(self.conv_r(comb))
        z = torch.sigmoid(self.conv_z(comb))
        cand_in = torch.cat([x, skip, r * h], dim=1)
        d_tilde = torch.tanh(self.conv_c(cand_in))
        h_new = (1.0 - z) * h + z * d_tilde
        return h_new


class EncoderBlock(nn.Module):
    """Bloco de encoder (E1/E2 do artigo): Conv -> LeakyReLU -> [AvgPool] -> Skip-ConvGRU.

    No encoder, He_t = 0 (não há skip de níveis superiores). O downsampling usa
    average pooling, como no artigo (LeCun et al., 1998)."""

    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.gru = SkipConvGRUCell(in_ch=out_ch, skip_ch=0, hid_ch=out_ch)

    def forward(self, x, h):
        x = self.act(self.conv(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        h = self.gru(x, None, h)
        return h  # estado de codificação (também é o output do bloco)


class DecoderBlock(nn.Module):
    """Bloco de decoder (D1/D2 do artigo): Skip-ConvGRU -> [TransposeConv] -> LeakyReLU.

    O decoder recebe o estado de codificação do mesmo nível como He_t (skip) e
    faz upsampling via convolução transposta."""

    def __init__(self, in_ch, skip_ch, hid_ch, out_ch, upsample=False):
        super().__init__()
        self.gru = SkipConvGRUCell(in_ch=in_ch, skip_ch=skip_ch, hid_ch=hid_ch)
        self.upsample = upsample
        if upsample:
            self.up = nn.ConvTranspose2d(hid_ch, out_ch, 2, stride=2)
        else:
            self.up = nn.Conv2d(hid_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, skip, h):
        h = self.gru(x, skip, h)         # novo estado de decodificação
        out = self.act(self.up(h))       # saída para o próximo nível
        return h, out


class DecoupledHeads(nn.Module):
    """Heads desacopladas (Fig. 2b): classificação wet/dry + regressão de profundidade.

    A saída de classificação (probabilidade de célula molhada) vira MÁSCARA que
    multiplica a saída de regressão, focando a profundidade nas áreas molhadas."""

    def __init__(self, in_ch, mid=16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(1, mid),  # ~ LayerNorm sobre canais
            nn.SiLU(),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.SiLU(),
            nn.Conv2d(mid, 1, 3, padding=1),
        )
        self.reg = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.SiLU(),
            nn.Conv2d(mid, 1, 3, padding=1),
        )

    def forward(self, x):
        feat = self.shared(x)
        logit = self.cls(feat)                 # logit wet/dry
        prob = torch.sigmoid(logit)            # probabilidade (máscara)
        depth = self.reg(feat)                 # profundidade bruta (>=0 via softplus)
        depth = F.softplus(depth)
        masked_depth = depth * prob            # foca nas células molhadas
        return logit, prob, masked_depth


class URNN(nn.Module):
    """U-RNN reduzido: backbone U-like (3 níveis) + heads desacopladas.

    Mantém estados recorrentes (latent autoregression) entre passos de tempo:
    3 estados de encoder + 3 de decoder. O método `init_states` zera/recria os
    estados; `forward_step` avança um passo de tempo."""

    def __init__(self, in_ch, base=24):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3   # ex.: 24, 48, 72
        self.c1, self.c2, self.c3 = c1, c2, c3
        # Encoder (E1, E2, E2)
        self.enc1 = EncoderBlock(in_ch, c1, downsample=False)
        self.enc2 = EncoderBlock(c1, c2, downsample=True)
        self.enc3 = EncoderBlock(c2, c3, downsample=True)
        # Decoder (D1, D2, D2) — sobe de /4 -> /2 -> full
        self.dec3 = DecoderBlock(in_ch=c3, skip_ch=c3, hid_ch=c3, out_ch=c2, upsample=True)
        self.dec2 = DecoderBlock(in_ch=c2, skip_ch=c2, hid_ch=c2, out_ch=c1, upsample=True)
        self.dec1 = DecoderBlock(in_ch=c1, skip_ch=c1, hid_ch=c1, out_ch=16, upsample=False)
        self.heads = DecoupledHeads(16)

    def init_states(self, batch, H, W, device):
        z = lambda c, h, w: torch.zeros(batch, c, h, w, device=device)
        return {
            "e1": z(self.c1, H, W),   "e2": z(self.c2, H // 2, W // 2),   "e3": z(self.c3, H // 4, W // 4),
            "d3": z(self.c3, H // 4, W // 4), "d2": z(self.c2, H // 2, W // 2), "d1": z(self.c1, H, W),
        }

    def forward_step(self, x, states):
        # ----- Encoder -----
        e1 = self.enc1(x, states["e1"])
        e2 = self.enc2(e1, states["e2"])
        e3 = self.enc3(e2, states["e3"])
        # ----- Decoder (skip = estado de codificação do mesmo nível) -----
        d3, up3 = self.dec3(e3, e3, states["d3"])
        d2, up2 = self.dec2(up3, e2, states["d2"])
        d1, out = self.dec1(up2, e1, states["d1"])
        # Atualiza estados recorrentes
        new_states = {"e1": e1, "e2": e2, "e3": e3, "d3": d3, "d2": d2, "d1": d1}
        logit, prob, depth = self.heads(out)
        return logit, prob, depth, new_states

    @staticmethod
    def detach_states(states):
        return {k: v.detach() for k, v in states.items()}


if __name__ == "__main__":
    # Teste rápido de dimensões
    torch.manual_seed(0)
    B, C, H, W, T = 2, 12, 32, 32, 5
    model = URNN(in_ch=C, base=16)
    st = model.init_states(B, H, W, "cpu")
    for t in range(T):
        x = torch.randn(B, C, H, W)
        logit, prob, depth, st = model.forward_step(x, st)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK | logit {tuple(logit.shape)} prob {tuple(prob.shape)} depth {tuple(depth.shape)}")
    print(f"Parametros treinaveis: {n_params:,}")
