# Período 7 — Nowcasting Espaço-Temporal de Inundações Urbanas com U-RNN

**Americas TechGuard · Campus Florianópolis**

**Estudante:** Rosemeri Borges (GitHub: [RoseBorges44](https://github.com/RoseBorges44))

**Eixo:** Aprendizado Profundo · Hidrologia Urbana · Modelagem Espaço-Temporal · Python

---

## 0. O que é esta entrega

Uma **reimplementação didática e reduzida** do modelo **U-RNN** (Cao et al., 2025) para
*nowcasting* espaço-temporal de inundações urbanas, em PyTorch, que **executa em Colab (GPU ou CPU)**,
sem depender de infraestrutura pesada. O pipeline treina, infere e gera saídas verificáveis (mapas, hidrogramas,
métricas), contextualizado no **Vale do Itajaí / Blumenau-SC** — dando continuidade às
Semanas 5 (NDVI Blumenau) e 6 (HAND Blumenau).

> **Importante (precisão conceitual exigida pelo enunciado):** isto é **nowcasting**, não
> mapa de suscetibilidade nem modelo de risco. *Nowcasting* prevê a **evolução no tempo** da
> inundação a partir de chuva + terreno + estados hidrológicos. A entrega **não** integra
> exposição, vulnerabilidade ou população, logo **não** constitui avaliação de risco completa.

### Resultados em uma linha (conjunto de teste, n=6 eventos)
| Métrica | Valor (média ± dp) | O que mede |
|---|---|---|
| **CSI** | **0,68 ± 0,07** | Acerto da extensão máxima da cheia (wet/dry) |
| **PR²** | **0,69 ± 0,21** | Consistência da profundidade de pico |
| **MAE** | **0,010 m** | Erro absoluto médio de profundidade |
| **RMSE** | **0,024 m** | Erro quadrático médio |
| **Inferência** | **~0,36 s** | Para um evento completo de **6 h** (48 passos) |

---

## 1. ETAPA 1 — Estudo técnico do U-RNN

**Problema resolvido.** Modelos hidrodinâmicos (MIKE+, SWMM, LISFLOOD-FP) simulam cheias
urbanas com alta fidelidade, mas são **lentos** (passos de tempo pequenos por estabilidade
numérica). Para *early warning*, precisa-se de previsão **rápida, de curto prazo (até ~6 h)
e em alta resolução** O(m) no espaço e O(min) no tempo. O U-RNN aprende a **emular** o modelo
hidrodinâmico, entregando *nowcasts* de 6 h em ~7 s (>100× mais rápido que o MIKE+).

**Entradas e saídas.**
- *Entradas:* sequência de **intensidade de chuva** (janela W), **chuva acumulada** (capta o
  "efeito integral"), e fatores espaciais  **DSM/DEM**, **impermeabilização**, **área de bueiros**.
- *Saídas:* a cada passo, **mapa wet/dry** (classificação) e **profundidade da lâmina d'água**
  (regressão).

**Arquitetura (ideia geral).** Backbone **U-like** (encoder-decoder) empilhado com blocos
**Skip-ConvGRU**  uma ConvGRU com *skip connections* que funde, para cada célula: a saída do
bloco anterior `I_t`, o estado de codificação do mesmo nível `He_t` e o estado de decodificação
anterior `Hd_{t-1}` (recorrência → *latent autoregression*). Duas **heads desacopladas**:
classificação (wet/dry, vira **máscara**) e regressão (profundidade, focada nas células molhadas).

**Paradigma SWP (Sliding Window-based Pre-warming).** Treinar BPTT sobre sequências longas
estoura a memória de GPU. O SWP divide a sequência em janelas de **K** passos; só esses K passos
guardam gradiente, e um **pre-warming gradient-free** (inferência até o início da janela) fornece
o estado inicial. Resultado no artigo: até **~360× menos memória** de GPU.

**O que usei de cada material de referência:**
- *Artigo* — Eq. 1 (Skip-ConvGRU), Fig. 2 (backbone + heads), Fig. 5 (SWP), Seção 2.6
  (hiperparâmetros: α=0,25→ajustei, γ=2, λ=20→30, W=30→6, K), métricas (Seção 2.7: CSI, PR², MAE, RMSE).
- *GitHub oficial* (holmescao/U-RNN) — estrutura do pipeline, papel do MIKE+ como ground truth,
  separação treino/teste, otimização Adam + cosine annealing.
- *Material suplementar* — entendimento do formato dos dados (grade 500×500 @ 2 m, 360 passos @ 1 min,
  56 eventos). **Não baixei** o dataset completo (volumoso, GPU); ver justificativa na Etapa 2.

---

## 2. ETAPA 2 — Ambiente e estratégia de execução

**Por que não usei a base oficial (dataset + pesos) diretamente.** Inspecionei o repositório
oficial (`holmescao/U-RNN`, tutoriais 02-04). A inferência oficial exige, simultaneamente:
(i) o **dataset UrbanFlood24**  **~115 GB** (a versão *Lite* de 128×128 ainda precisa ser baixada);
(ii) **pesos pré-treinados** de 11-300 MB hospedados em **Google Drive / Baidu / Hugging Face**; e
(iii) **uma GPU** (o próprio tutorial de inferência lista "any GPU" como requisito; a instalação é
feita para CUDA em RTX 4090). No ambiente de execução que usei **não há GPU** e a rede só alcança
um conjunto restrito de domínios que **não inclui** Google Drive, Hugging Face nem Figshare — ou
seja, era **tecnicamente impossível** baixar os pesos/dados oficiais aqui. O enunciado autoriza
explicitamente **versão reduzida / dados sintéticos justificados / pesos pré-treinados**, desde que
a solução **execute e gere saídas verificáveis**.

**Minha estratégia.** Reconstruí os **componentes conceituais** do U-RNN em escala reduzida e
substituí o MIKE+ por um **surrogate**: um **autômato celular de inundação** (chuva → escoamento
→ roteamento por gravidade → drenagem) sobre um **terreno sintético tipo vale do Itajaí**. A
U-RNN aprende a **emular** esse modelo numérico, exatamente o paradigma do artigo, só que com
um "ground truth" que eu posso gerar sem licença nem GPU.

> **Isto está alinhado com os próprios autores.** O `quickstart.ipynb` oficial tem uma *"Architecture
> Demo (Synthetic Data)"* que cria "a tiny **synthetic dataset** (random DEM + rainfall + flood)" e
> roda o modelo  **"No real data needed"**. Ou seja, usar dados sintéticos para demonstrar a
> arquitetura é o caminho acessível recomendado no repositório. Nós **fomos além do quickstart**:
> em vez de um único *forward pass*, montamos um surrogate físico (não aleatório), **treinamos** com
> o paradigma SWP, **avaliamos** (CSI/PR²/MAE/RMSE) e contextualizamos em Blumenau.

**Ambiente (reprodutível).** Testado em Python **3.12.3**, PyTorch **2.12.1**, NumPy **2.4**,
Matplotlib **3.10**  ver `requirements.txt`. O notebook é *device-aware* (usa GPU se houver). A
**execução de referência entregue** foi em **Colab GPU, 56 épocas do zero** (grade 40×40, T=48, K=16);
também roda em CPU via treino incremental por blocos com checkpoint.

### Comparação analítica: U-RNN original × nossa adaptação

| Aspecto | U-RNN original (Cao et al., 2025) | Nossa adaptação (este trabalho) |
|---|---|---|
| Região | Shenzhen (China) + Glasgow (UK) | Vale do Itajaí / Blumenau-SC (estilizado) |
| Grade / resolução | 500×500 @ 2 m | 40×40 @ "2 m sintético" |
| Sequência | 360 passos @ 1 min (6 h) | 48 passos @ 2 min (~1,6 h) |
| Ground truth | MIKE+ 2023 (hidrodinâmico calibrado) | Autômato celular (surrogate físico) |
| Nº de eventos | 56 (20 obs. + 36 projeto) | 18 (12 treino + 6 teste), sintéticos |
| Janela de chuva W | 30 | 6 |
| Janela SWP K | 28 | 16 |
| Hardware / treino | 24× RTX 4090, ~7 dias | 1 GPU (Colab), 56 épocas (~min) |
| Parâmetros | (rede completa, mais canais) | 526 866 |
| CSI (teste) | ~0,89–0,97 | **0,68 ± 0,07** |
| PR² (profundidade) | 0,975–0,996 | **0,69 ± 0,21** |
| Velocidade de inferência | 6 h em ~7 s (>100× MIKE+) | 6 h-equiv. em ~0,36 s |

O que **mantivemos fiel:** Skip-ConvGRU (Eq. 1), backbone U-like encoder-decoder, heads desacopladas
com máscara de classificação, paradigma SWP, perdas Focal BCE + MSE ponderada, e o próprio paradigma
"aprender a emular um modelo numérico". O que **reduzimos:** escala espacial/temporal, nº de eventos,
capacidade da rede e o ground truth (surrogate no lugar do MIKE+) tudo por restrição de recursos, e
explicitamente autorizado pelo enunciado.

---

## 3. ETAPA 3 — Implementação / adaptação funcional

Estrutura do código (em `src/`):
| Arquivo | Conteúdo |
|---|---|
| `model.py` | Skip-ConvGRU (Eq. 1), backbone U-like 3 níveis, heads desacopladas |
| `data.py` | Terreno sintético Blumenau + autômato celular de inundação (surrogate MIKE+) |
| `train.py` | Paradigma SWP, Focal BCE + MSE ponderada, métricas, inferência |
| `run_experiment.py` | Pipeline completo (treino + avaliação + persistência) |
| `train_chunk.py` | Treino incremental por blocos (resiliente a timeout) |
| `evaluate.py` | Avaliação no teste + geração dos arrays para figuras |
| `figures.py` | Geração de todas as figuras de evidência |

O notebook **`URNN_Blumenau_Nowcasting.ipynb`** reúne tudo num fluxo executável célula a célula.

**Decisões técnicas principais:**
- Janela de chuva **W=6** (em vez de 30) e grade **40×40** (em vez de 500×500) para treino rápido.
- **DEM escalado a ~15 m de relevo** para a água concentrar nas cotas baixas (cheia localizada;
  razão seco:molhado realista, como o artigo cita).
- **Surrogate** com contorno aberto e drenagem do canal (o rio escoa) para evitar empoçamento irreal.
- `α` do Focal BCE ajustado de 0,25 → **0,5** (a classe molhada é rara e queremos detectá-la;
  isso elevou o PR² de ~0,52 → ~0,69 e o CSI de ~0,49 → ~0,68).
- **CSI calculado a partir da branch de classificação** (a extensão wet/dry vem dela, Fig. 2b),
  não de um limiar sobre a profundidade regredida.

### Notas de desenvolvimento (erros encontrados e como resolvi)

Registro honesto do caminho, porque foi debugando que entendi de fato o modelo:
- **Inundação cobrindo 98% do domínio.** No começo o autômato alagava tudo o relevo estava
  normalizado em 1 m, muito raso perto das lâminas d'água. **Solução:** escalei o DEM para ~15 m
  de desnível e adicionei contorno aberto + drenagem do canal (o rio escoa). Aí a cheia ficou
  localizada (pico ~34% molhado), como o artigo descreve.
- **Métricas fracas no início (CSI 0,49 / PR² 0,15).** O modelo previa "seco demais". Descobri que
  o `α=0,25` do Focal BCE (padrão do RetinaNet) *subponderava* a classe molhada, que aqui é a rara e
  a que interessa. **Solução:** subi para `α=0,5` → PR² foi de ~0,52 para ~0,69.
- **CSI travado em ~0,52.** Eu estava medindo a extensão da cheia por limiar sobre a profundidade
  regredida. Relendo a Fig. 2b, percebi que a extensão wet/dry vem da **branch de classificação**.
  Passei a calcular o CSI pela máscara de classificação → CSI subiu para ~0,68 (correção conceitual,
  não truque).
- **Timeout no treino longo.** 34 épocas de uma vez estouravam o tempo de execução. **Solução:**
  criei `train_chunk.py` com checkpoint, treinando em blocos (warm restarts) até 56 épocas.
- **Erro de shape nas perdas.** Faltava a dimensão de *batch* nos alvos (`(1,H,W)` vs `(1,1,H,W)`).
  **Solução:** indexação `[t:t+1]` para preservar a dimensão.

### Conexão com as semanas anteriores

Esta entrega fecha um arco: a **Semana 5 (NDVI Blumenau)** mapeou a cobertura vegetal que informa
o **coeficiente de escoamento** (aqui representado pela camada de impermeabilização); a **Semana 6
(HAND Blumenau)** produziu a suscetibilidade estática a inundação — e agora a Semana 7 dá o passo
seguinte: a **dinâmica no tempo**. Numa evolução real, o **DEM/HAND da Semana 6 substituiria o DEM
sintético** como fator espacial de entrada, e o **NDVI da Semana 5** calibraria a impermeabilização —
os três produtos se encaixam no mesmo pipeline.

---

## 4. ETAPA 4 — Evidências, resultados e limitações

Figuras em `outputs/figures/`:
1. `fig1_area_estudo.png` — fatores espaciais (DEM, impermeabilização, bueiros)
2. `fig2_evolucao_temporal.png` — **evolução da cheia: U-RNN vs referência** (4 instantes)
3. `fig3_hidrogramas.png` — hidrogramas em 3 pontos (predito vs referência)
4. `fig4_scatter_pr2.png` — consistência de profundidade de pico (PR²)
5. `fig5_csi_extensao.png` — extensão máxima: TP/FP/FN (CSI)
6. `fig6_curva_perda.png` — convergência do treino
7. `fig7_swp_memoria.png` — princípio do SWP (memória vs K)
8. `fig8_metricas_evento.png` — métricas por evento

**Quais dados foram usados:** 100% **sintéticos** (terreno + chuvas tipo Chicago + autômato celular).
12 eventos de treino, 6 de teste, com picos e lâminas variados.

**O que a saída representa / não permite afirmar:** representa a **capacidade da arquitetura** de
aprender a dinâmica espaço-temporal de uma cheia urbana e emular um modelo numérico rapidamente.
**Não** representa inundação real de Blumenau, **não** está calibrada com cotas observadas e
**não** substitui um modelo hidrodinâmico.

**Limitações:** (i) dados sintéticos, sem validação local; (ii) grade/resolução muito menores que
o artigo (40×40 @ "2 m sintético", não 500×500); (iii) rede de capacidade reduzida → métricas
menores (CSI ~0,68 vs ~0,9+ do artigo); (iv) o surrogate é uma simplificação física, não o MIKE+;
(v) chuva espacialmente uniforme (não variável).

---

## 5. ETAPA 5 — Aplicação ao Americas TechGuard

Um *nowcasting* como este apoiaria o TechGuard em **monitoramento, prevenção, resposta e alerta**:
- **Integração com chuva real:** substituir as chuvas sintéticas por dados de radar/pluviômetros
  (ex.: Defesa Civil SC, INMET) e por *nowcasting* meteorológico de curto prazo.
- **Terreno real:** trocar o DEM sintético pelo **DEM/HAND de Blumenau da Semana 6** e por DSM de
  maior resolução (a melhoria que o professor sugeriu na Semana 6).
- **Sensores e satélite:** assimilar nível de rios (Itajaí-Açu), NDVI (Semana 5) e cotas históricas.
- **Disseminação:** profundidade prevista por minuto alimentando dashboards e alertas a celulares.

**O que já está entregue vs. o que falta para uma PoC robusta:** entregamos a **arquitetura e o
pipeline funcionais**. Para uma prova de conceito real faltam: dados hidrodinâmicos calibrados
(ou MIKE+/HEC-RAS) como ground truth, DEM/DSM reais de alta resolução, GPU para treino em escala,
chuva espacialmente variável e validação com eventos históricos do Vale do Itajaí.

---

## Como executar

```bash
pip install -r requirements.txt
cd src
python3 run_experiment.py     # treino + avaliação + salva artefatos
python3 figures.py            # gera as figuras de evidência
```
Ou abra **`URNN_Blumenau_Nowcasting.ipynb`** no Google Colab e rode célula a célula.

## Materiais de referência (obrigatórios)
- **Artigo:** Cao, X. et al. (2025). *U-RNN high-resolution spatiotemporal nowcasting of urban
  flooding.* **Journal of Hydrology** 659:133117. DOI: [10.1016/j.jhydrol.2025.133117](https://doi.org/10.1016/j.jhydrol.2025.133117)
- **GitHub oficial:** https://github.com/holmescao/U-RNN
- **Material suplementar (Figshare):** https://figshare.com/articles/dataset/28082549

## Licença e uso
Uso **acadêmico**, respeitando as licenças dos materiais de origem. Código próprio de adaptação
sob MIT. Os dados são sintéticos (gerados pelo `data.py`), sem restrição de origem.
