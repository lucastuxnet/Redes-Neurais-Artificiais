#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 03 - Reconhecimento de letras (X e T) usando Perceptrons

Este script implementa:
  (a) um Perceptron de 1 neuronio que discrimina entre as letras X e T;
  (b) uma camada de Perceptrons com 2 neuronios (saida "one-hot" bipolar)
      que faz a mesma discriminacao, um neuronio "especialista" em cada letra.

Os padroes de entrada sao grades 5x5 com pixels bipolares (+1 = aceso,
-1 = apagado), extraidos do enunciado (slide "Trabalho 03") e convertidos
em vetores de 25 posicoes (x1..x25), na ordem linha a linha (row-major).

Autor: gerado com apoio do Claude (Anthropic) para fins didaticos.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reprodutibilidade dos experimentos com ruido
RNG = np.random.default_rng(42)

# Diretorio de saida para figuras e resultados
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Definicao dos padroes (letras X e T em uma grade 5x5, valores bipolares)
# ---------------------------------------------------------------------------

X_GRID = np.array([
    [ 1, -1, -1, -1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1, -1, -1,  1],
], dtype=float)

T_GRID = np.array([
    [ 1,  1,  1,  1,  1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
], dtype=float)

N_LIN, N_COL = X_GRID.shape
N_ENTRADAS = N_LIN * N_COL  # 25

# Vetores de entrada (row-major flatten)
x_X = X_GRID.flatten()
x_T = T_GRID.flatten()

PADROES = np.array([x_X, x_T])          # shape (2, 25)
NOMES_PADROES = ["X", "T"]

# Alvos do item (a): 1 neuronio -> y = 1 se X, y = -1 se T
ALVOS_A = np.array([1.0, -1.0])

# Alvos do item (b): 2 neuronios -> (y1, y2) = (1,-1) se X, (-1,1) se T
ALVOS_B = np.array([
    [1.0, -1.0],   # saida desejada para o padrao X: (y1=1 , y2=-1)
    [-1.0, 1.0],   # saida desejada para o padrao T: (y1=-1, y2=1)
])


# ---------------------------------------------------------------------------
# 2. Funcao de ativacao e Perceptron (regra de aprendizado de Rosenblatt)
# ---------------------------------------------------------------------------

def ativacao_bipolar(net):
    """Funcao degrau bipolar: +1 se net >= 0, -1 caso contrario."""
    return np.where(net >= 0, 1.0, -1.0)


class Perceptron:
    """Perceptron simples (1 neuronio) com ativacao bipolar (+1/-1).

    Implementa a regra de aprendizado de Rosenblatt:
        w <- w + eta * (t - y) * x
        b <- b + eta * (t - y)
    onde t e o alvo, y = f(w.x + b) e a saida atual e eta e a taxa de
    aprendizado.
    """

    def __init__(self, n_entradas, eta=1.0):
        # pesos e bias iniciais em zero (inicializacao classica do
        # perceptron de Rosenblatt). A regra de aprendizado converge a
        # partir de qualquer ponto inicial quando o problema e linearmente
        # separavel, mas comecar do zero torna o processo determinista e
        # deixa claro, no vetor de pesos final, exatamente quais entradas
        # foram ajustadas para separar as duas classes.
        self.w = np.zeros(n_entradas)
        self.b = 0.0
        self.eta = eta
        self.historico_erros = []  # numero de padroes mal-classificados por epoca

    def net(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return ativacao_bipolar(self.net(X))

    def fit(self, X, t, max_epocas=100):
        """Treina o perceptron ate convergencia (erro zero) ou max_epocas."""
        for epoca in range(max_epocas):
            erros = 0
            for xi, ti in zip(X, t):
                yi = self.predict(xi.reshape(1, -1))[0]
                if yi != ti:
                    erro = (ti - yi)
                    self.w += self.eta * erro * xi
                    self.b += self.eta * erro
                    erros += 1
            self.historico_erros.append(erros)
            if erros == 0:
                break
        return self


class CamadaPerceptrons:
    """Camada de M perceptrons independentes (sem camada oculta),
    usada no item (b) com M = 2 neuronios de saida. Cada neuronio da
    camada e treinado de forma independente com a regra do perceptron,
    usando a mesma entrada e sua propria coluna de alvo.
    """

    def __init__(self, n_entradas, n_saidas, eta=1.0):
        self.neuronios = [Perceptron(n_entradas, eta=eta)
                           for _ in range(n_saidas)]

    def predict(self, X):
        saidas = [n.predict(X) for n in self.neuronios]
        return np.stack(saidas, axis=-1)

    def fit(self, X, T, max_epocas=100):
        """T tem shape (n_amostras, n_saidas)."""
        for k, neuronio in enumerate(self.neuronios):
            neuronio.fit(X, T[:, k], max_epocas=max_epocas)
        return self


# ---------------------------------------------------------------------------
# 3. Treinamento
# ---------------------------------------------------------------------------

def treinar_item_a():
    p = Perceptron(N_ENTRADAS, eta=1.0)
    p.fit(PADROES, ALVOS_A, max_epocas=100)
    return p


def treinar_item_b():
    camada = CamadaPerceptrons(N_ENTRADAS, n_saidas=2, eta=1.0)
    camada.fit(PADROES, ALVOS_B, max_epocas=100)
    return camada


# ---------------------------------------------------------------------------
# 4. Teste de robustez a ruido (pixels invertidos aleatoriamente)
# ---------------------------------------------------------------------------

def aplicar_ruido(x, n_flips, rng):
    """Retorna uma copia de x com n_flips posicoes com o sinal invertido."""
    x_ruidoso = x.copy()
    idx = rng.choice(len(x), size=n_flips, replace=False)
    x_ruidoso[idx] *= -1
    return x_ruidoso


def testar_robustez_a(perceptron, n_ensaios=300, max_flips=8):
    """Para cada quantidade de pixels invertidos (0..max_flips), gera
    n_ensaios versoes ruidosas de X e de T e mede a taxa de acerto do
    perceptron do item (a)."""
    taxas = []
    for n_flips in range(0, max_flips + 1):
        acertos = 0
        total = 0
        for x_original, alvo in zip(PADROES, ALVOS_A):
            for _ in range(n_ensaios):
                x_ruido = aplicar_ruido(x_original, n_flips, RNG) if n_flips > 0 else x_original
                y = perceptron.predict(x_ruido.reshape(1, -1))[0]
                acertos += int(y == alvo)
                total += 1
        taxas.append(acertos / total)
    return np.array(taxas)


def testar_robustez_b(camada, n_ensaios=300, max_flips=8):
    taxas = []
    for n_flips in range(0, max_flips + 1):
        acertos = 0
        total = 0
        for x_original, alvo in zip(PADROES, ALVOS_B):
            for _ in range(n_ensaios):
                x_ruido = aplicar_ruido(x_original, n_flips, RNG) if n_flips > 0 else x_original
                y = camada.predict(x_ruido.reshape(1, -1))[0]
                acertos += int(np.array_equal(y, alvo))
                total += 1
        taxas.append(acertos / total)
    return np.array(taxas)


# ---------------------------------------------------------------------------
# 5. Grafico (uma unica imagem consolidada com todos os resultados)
# ---------------------------------------------------------------------------

CMAP_PADRAO = "YlOrBr"
CMAP_PESOS = "RdBu_r"


def gerar_figura_resultados(p, camada, taxas_a, taxas_b, nome_arquivo="resultados.png"):
    """Monta e salva UMA UNICA imagem com todos os resultados do trabalho:
    padroes de entrada, curvas de convergencia, pesos aprendidos e o teste
    de robustez a ruido. E sobrescrita a cada execucao do script, entao o
    arquivo de saida reflete sempre o resultado mais recente.
    """
    # historico de erros do item (b), com as duas listas niveladas no
    # mesmo tamanho (preenchendo com 0 apos a convergencia de cada uma)
    n_epocas_b = max(len(n.historico_erros) for n in camada.neuronios)
    hist_b = np.array([
        n.historico_erros + [0] * (n_epocas_b - len(n.historico_erros))
        for n in camada.neuronios
    ])

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.1], hspace=0.55, wspace=0.4,
                           top=0.93, bottom=0.04, left=0.06, right=0.98)

    # ---- linha 0: padroes de entrada (X e T) ----
    for col, (grid, nome) in enumerate(zip([X_GRID, T_GRID], NOMES_PADROES)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(grid, cmap=CMAP_PADRAO, vmin=-1, vmax=1)
        ax.set_title(f"Padrao de entrada - Letra {nome}")
        ax.set_xticks(range(N_COL)); ax.set_yticks(range(N_LIN))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        for i in range(N_LIN):
            for j in range(N_COL):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=8)
        ax.grid(color="gray", linewidth=0.5)

    # ---- linha 1: convergencia (item a e item b) ----
    ax = fig.add_subplot(gs[1, 0])
    h_a = p.historico_erros
    ax.plot(range(1, len(h_a) + 1), h_a, marker="o", color="tab:blue")
    ax.set_title("Convergencia - item (a)")
    ax.set_xlabel("Epoca"); ax.set_ylabel("Padroes mal classificados")
    ax.set_xticks(range(1, len(h_a) + 1))
    ax.set_ylim(bottom=-0.2)

    ax = fig.add_subplot(gs[1, 1])
    for h, r in zip(hist_b, ["Neuronio 1 (X)", "Neuronio 2 (T)"]):
        ax.plot(range(1, len(h) + 1), h, marker="o", label=r)
    ax.set_title("Convergencia - item (b)")
    ax.set_xlabel("Epoca")
    ax.set_xticks(range(1, n_epocas_b + 1))
    ax.set_ylim(bottom=-0.2)
    ax.legend(fontsize=8)

    # ---- linha 2: pesos aprendidos (item a, e os 2 neuronios do item b) ----
    pesos = [p.w, camada.neuronios[0].w, camada.neuronios[1].w]
    titulos_pesos = ["Pesos w (item a)", "Pesos w1 (item b)", "Pesos w2 (item b)"]
    vlim = max((np.max(np.abs(w)) for w in pesos), default=1) or 1
    for col, (w, titulo) in enumerate(zip(pesos, titulos_pesos)):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(w.reshape(N_LIN, N_COL), cmap=CMAP_PESOS, vmin=-vlim, vmax=vlim)
        ax.set_title(titulo)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- linha 3: robustez a ruido (ocupando as 3 colunas) ----
    ax = fig.add_subplot(gs[3, :])
    flips = range(len(taxas_a))
    ax.plot(flips, taxas_a * 100, marker="o", label="Item (a) - 1 neuronio")
    ax.plot(flips, taxas_b * 100, marker="s", label="Item (b) - 2 neuronios")
    ax.set_xlabel("Pixels invertidos (ruido)")
    ax.set_ylabel("Taxa de acerto (%)")
    ax.set_title("Robustez a ruido")
    ax.set_ylim(0, 105)
    ax.legend()

    fig.suptitle("Trabalho 03 - Reconhecimento de letras X e T usando Perceptrons",
                 fontsize=14, y=0.98)

    caminho = os.path.join(OUT_DIR, nome_arquivo)
    fig.savefig(caminho, dpi=180)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# 6. Execucao principal
# ---------------------------------------------------------------------------

def main():
    linhas_log = []

    def log(msg=""):
        print(msg)
        linhas_log.append(str(msg))

    log("=" * 70)
    log("Trabalho 03 - Reconhecimento de letras X e T usando Perceptrons")
    log("=" * 70)

    log("\nPadroes de entrada (vetores bipolares de 25 posicoes):")
    log(f"  x_X = {x_X.astype(int).tolist()}")
    log(f"  x_T = {x_T.astype(int).tolist()}")

    # ---------------- Item (a): 1 neuronio ----------------
    log("\n" + "-" * 70)
    log("ITEM (a): Perceptron de 1 neuronio")
    log("-" * 70)
    p = treinar_item_a()
    n_epocas_a = len(p.historico_erros)
    log(f"Convergiu em {n_epocas_a} epoca(s). Historico de erros por epoca: {p.historico_erros}")
    log(f"Pesos finais (w): {np.round(p.w, 3).tolist()}")
    log(f"Bias final (b): {round(p.b, 3)}")

    saida_X = p.predict(x_X.reshape(1, -1))[0]
    saida_T = p.predict(x_T.reshape(1, -1))[0]
    log(f"Saida para o padrao X: y = {saida_X:+.0f}  (esperado +1)")
    log(f"Saida para o padrao T: y = {saida_T:+.0f}  (esperado -1)")

    # ---------------- Item (b): 2 neuronios ----------------
    log("\n" + "-" * 70)
    log("ITEM (b): Camada com 2 neuronios (Perceptrons independentes)")
    log("-" * 70)
    camada = treinar_item_b()
    for k, n in enumerate(camada.neuronios, start=1):
        log(f"Neuronio {k}: convergiu em {len(n.historico_erros)} epoca(s); "
            f"erros por epoca = {n.historico_erros}")
        log(f"  Pesos w{k}: {np.round(n.w, 3).tolist()}")
        log(f"  Bias b{k}: {round(n.b, 3)}")

    saida_X_b = camada.predict(x_X.reshape(1, -1))[0]
    saida_T_b = camada.predict(x_T.reshape(1, -1))[0]
    log(f"Saida (y1,y2) para o padrao X: {tuple(saida_X_b)}  (esperado (+1,-1))")
    log(f"Saida (y1,y2) para o padrao T: {tuple(saida_T_b)}  (esperado (-1,+1))")

    # ---------------- Robustez a ruido ----------------
    log("\n" + "-" * 70)
    log("Teste de robustez a ruido (inversao aleatoria de pixels)")
    log("-" * 70)
    taxas_a = testar_robustez_a(p)
    taxas_b = testar_robustez_b(camada)
    for k in range(len(taxas_a)):
        log(f"  {k} pixel(s) invertido(s): acerto item (a) = {taxas_a[k]*100:5.1f}% | "
            f"item (b) = {taxas_b[k]*100:5.1f}%")

    # ---------------- Figura unica com todos os resultados ----------------
    caminho_fig = gerar_figura_resultados(p, camada, taxas_a, taxas_b, "resultados.png")

    # ---------------- Salvar log ----------------
    with open(os.path.join(OUT_DIR, "resultados.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_log))

    log(f"\nImagem consolidada salva em: {caminho_fig}")
    log(f"Log salvo em: {os.path.join(OUT_DIR, 'resultados.txt')}")


if __name__ == "__main__":
    main()
