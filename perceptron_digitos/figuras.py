"""
Gera as figuras estaticas do relatorio e dos slides.
Trabalho 04 - Disciplina de Redes Neurais

Rode:  python3 figuras.py
Saida: pasta figuras/ com 5 PNGs
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # nao precisa de tela para gerar os arquivos
import matplotlib.pyplot as plt
import numpy as np

from perceptron import (LADO, Perceptron, adicionar_ruido, carregar_digitos,
                        para_imagem, testar_com_ruido)

PASTA = Path(__file__).parent / "figuras"
AZUL = "#1f4e79"
VERDE = "#1e8449"
VERMELHO = "#c0392b"


def preparar():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linestyle": ":", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    PASTA.mkdir(exist_ok=True)


def grade(ax, matriz, cmap="gray_r", vmin=-1, vmax=1):
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, LADO, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, LADO, 1), minor=True)
    ax.grid(which="minor", color="#aaaaaa", linewidth=0.5)
    ax.grid(which="major", visible=False)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# --------------------------------------------------------------------------
def fig1_digitos(X, rotulos):
    """Os dez padroes de treinamento."""
    fig, eixos = plt.subplots(2, 5, figsize=(10, 4.4))
    for k, ax in zip(rotulos, eixos.ravel()):
        grade(ax, para_imagem(X[k]))
        ax.set_title(f"digito {k}", fontsize=10)
    fig.suptitle("Os dez digitos em matriz 10 x 10   "
                 "(preto = +1, branco = -1)", y=1.0)
    fig.tight_layout()
    fig.savefig(PASTA / "fig1_digitos.png")
    plt.close(fig)


def fig2_convergencia(rede):
    """Atualizacoes de peso e acertos por epoca."""
    ep = [h[0] for h in rede.historico]
    atualiz = [h[1] for h in rede.historico]
    acertos = [h[2] for h in rede.historico]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))

    a1.bar(ep, atualiz, color=AZUL, width=0.55)
    for x, v in zip(ep, atualiz):
        a1.text(x, v + 0.8, str(v), ha="center", fontsize=9)
    a1.set_xlabel("epoca"); a1.set_ylabel("atualizacoes de peso")
    a1.set_xticks(ep)
    a1.set_title("(a) Correcoes por epoca\no treino para quando chega a zero")
    a1.set_ylim(0, max(atualiz) * 1.18)

    a2.plot(ep, acertos, "o-", color=VERDE, markersize=7)
    a2.axhline(10, color="#999999", ls="--", lw=1)
    a2.set_xlabel("epoca"); a2.set_ylabel("digitos corretos (de 10)")
    a2.set_xticks(ep); a2.set_ylim(0, 10.8)
    a2.set_title("(b) Acertos por epoca")

    fig.suptitle("Treinamento do perceptron", y=1.03)
    fig.tight_layout()
    fig.savefig(PASTA / "fig2_convergencia.png")
    plt.close(fig)


def fig3_pesos(rede):
    """Os pesos aprendidos, vistos como imagem."""
    vmax = float(np.percentile(np.abs(rede.W), 98))
    fig, eixos = plt.subplots(2, 5, figsize=(11, 4.8))
    for j, ax in enumerate(eixos.ravel()):
        im = grade(ax, rede.mapa_de_pesos(j), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
        ax.set_title(f"neuronio {j}", fontsize=10)
    fig.colorbar(im, ax=eixos, shrink=0.8, label="peso", pad=0.02)
    fig.suptitle("O que cada neuronio aprendeu\n"
                 "vermelho = evidencia a favor do digito ; "
                 "azul = evidencia contra", y=1.05)
    fig.savefig(PASTA / "fig3_pesos.png")
    plt.close(fig)


def fig4_exemplos_ruido(X):
    """O mesmo digito com ruido crescente."""
    niveis = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    rng = np.random.default_rng(7)
    fig, eixos = plt.subplots(1, len(niveis), figsize=(1.7 * len(niveis), 2.4))
    for p, ax in zip(niveis, eixos):
        grade(ax, para_imagem(adicionar_ruido(X[3], p, rng)))
        ax.set_title(f"p = {p:g}", fontsize=10)
    fig.suptitle("O digito 3 com ruido crescente "
                 "(p = fracao de pixels invertidos)", y=1.06)
    fig.tight_layout()
    fig.savefig(PASTA / "fig4_exemplos_ruido.png")
    plt.close(fig)


def fig5_curva_ruido(rede, X, rotulos):
    """Acuracia em funcao do ruido."""
    ps = np.arange(0.0, 0.55, 0.05)
    accs = [testar_com_ruido(rede, X, rotulos, p, repeticoes=300) * 100
            for p in ps]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ps, accs, "o-", color=AZUL, markersize=6, lw=2)
    ax.axhline(10, color=VERMELHO, ls=":", lw=1.4)
    ax.text(0.015, 13, "acertar no chute = 10%", ha="left", fontsize=9,
            color=VERMELHO)
    ax.axhline(90, color="#999999", ls="--", lw=1)

    # marca onde a curva cruza os 90%. Interpolamos entre os dois pontos
    # vizinhos: pegar so o primeiro ponto abaixo de 90% daria um valor
    # grosseiro, preso ao passo da grade (0.05).
    cruz = None
    for i in range(1, len(ps)):
        if accs[i] < 90 <= accs[i - 1]:
            x0, x1, y0, y1 = ps[i - 1], ps[i], accs[i - 1], accs[i]
            cruz = x0 + (90 - y0) * (x1 - x0) / (y1 - y0)
            break
    if cruz is not None:
        ax.plot([cruz], [90], "o", color=VERMELHO, markersize=8, zorder=5)
        ax.annotate(f"cai abaixo de 90%\nem p $\\approx$ {cruz:.2f}",
                    xy=(cruz, 90), xytext=(0.04, 62),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    connectionstyle="arc3,rad=-0.2"),
                    fontsize=9.5, color="#555555")

    ax.set_xlabel("p  (probabilidade de inverter cada pixel)")
    ax.set_ylabel("acuracia (%)")
    ax.set_title("Ate quanto ruido o perceptron aguenta?\n"
                 "media de 300 sorteios por ponto")
    ax.set_ylim(0, 104)
    fig.tight_layout()
    fig.savefig(PASTA / "fig5_curva_ruido.png")
    plt.close(fig)
    return ps, accs


# --------------------------------------------------------------------------
def main():
    preparar()
    X, T, rotulos = carregar_digitos()

    rede = Perceptron()
    rede.treinar(X, T, mostrar=False)

    print("Gerando figuras em figuras/ ...")
    fig1_digitos(X, rotulos);            print("  [ok] fig1_digitos.png")
    fig2_convergencia(rede);             print("  [ok] fig2_convergencia.png")
    fig3_pesos(rede);                    print("  [ok] fig3_pesos.png")
    fig4_exemplos_ruido(X);              print("  [ok] fig4_exemplos_ruido.png")
    ps, accs = fig5_curva_ruido(rede, X, rotulos)
    print("  [ok] fig5_curva_ruido.png")

    print("\nValores da curva de ruido (para citar no relatorio):")
    for p, a in zip(ps, accs):
        print(f"   p = {p:.2f}  ->  {a:5.1f}%")


if __name__ == "__main__":
    main()
