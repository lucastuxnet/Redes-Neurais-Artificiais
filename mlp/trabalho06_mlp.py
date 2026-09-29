#!/usr/bin/env python3
"""
Trabalho 06 - Aproximação Funcional com MLP
EL056 - Redes Neurais Artificiais (PPGEELT/UFU) - Prof. Dr. Keiji Yamanaka
Aluno: Lucas A. Martins (12622EEL006)

MLP 1-N-1 (oculta tanh, saída linear) treinada com backpropagation
implementado do zero em NumPy: gradiente descendente com momento,
modos em lote (batch) e padrão a padrão (online), inicialização
uniforme ou Nguyen-Widrow.

Uso:
    python trabalho06_mlp.py                       # treinamento base + figura
    python trabalho06_mlp.py --neurons 8 --lr 0.1  # outra configuração
    python trabalho06_mlp.py --experimentos        # todos os experimentos do relatório
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dados do enunciado
# ---------------------------------------------------------------------------
X_DADOS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
T_DADOS = np.array([-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600,
                    0.4609, 0.1336, -0.2013, -0.4344, -0.5000])

# Configuração base (escolhida a partir dos experimentos)
BASE = dict(neurons=5, lr=0.05, momentum=0.9, epochs=20000, tol=1e-3,
            seed=42, init="nguyen-widrow", mode="batch")


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """Pesos da rede 1-N-1."""
    W1: np.ndarray  # (N, 1)  entrada -> oculta
    b1: np.ndarray  # (N, 1)  bias da oculta
    W2: np.ndarray  # (1, N)  oculta -> saída
    b2: np.ndarray  # (1, 1)  bias da saída

    def copy(self) -> "Params":
        return Params(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def as_list(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


@dataclass
class Resultado:
    params: Params
    historico: list[float] = field(default_factory=list)  # MSE por época
    epocas: int = 0
    convergiu: bool = False
    divergiu: bool = False

    @property
    def mse_final(self) -> float:
        return self.historico[-1] if self.historico else float("nan")


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------
def init_weights(n_hidden: int, metodo: str, rng: np.random.Generator) -> Params:
    """Inicializa os pesos.

    - 'uniforme': todos os pesos e biases em U(-0.5, 0.5).
    - 'nguyen-widrow': pesos da oculta reescalados para norma beta = 0.7 * N^(1/n)
      (n = nº de entradas = 1) e biases em U(-beta, beta), espalhando as regiões
      ativas das tanh pelo intervalo de entrada (Nguyen & Widrow, 1990).
    """
    n_in = 1
    W1 = rng.uniform(-0.5, 0.5, (n_hidden, n_in))
    b1 = rng.uniform(-0.5, 0.5, (n_hidden, 1))
    if metodo == "nguyen-widrow":
        beta = 0.7 * n_hidden ** (1.0 / n_in)
        W1 = beta * W1 / np.linalg.norm(W1, axis=1, keepdims=True)
        b1 = rng.uniform(-beta, beta, (n_hidden, 1))
    elif metodo != "uniforme":
        raise ValueError(f"método de inicialização desconhecido: {metodo}")
    W2 = rng.uniform(-0.5, 0.5, (1, n_hidden))
    b2 = rng.uniform(-0.5, 0.5, (1, 1))
    return Params(W1, b1, W2, b2)


# ---------------------------------------------------------------------------
# Propagação direta e retropropagação
# ---------------------------------------------------------------------------
def forward(x: np.ndarray, p: Params) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Propagação direta. x: (P,) ou (1, P). Retorna y (1, P) e cache."""
    X = np.atleast_2d(x).reshape(1, -1)      # (1, P)
    net1 = p.W1 @ X + p.b1                   # (N, P)
    h = np.tanh(net1)                        # (N, P)  ativação da oculta
    y = p.W2 @ h + p.b2                      # (1, P)  saída linear
    return y, {"X": X, "h": h}


def backward(t: np.ndarray, y: np.ndarray, cache: dict[str, np.ndarray],
             p: Params) -> list[np.ndarray]:
    """Retropropagação do erro E = 1/(2P) * sum (t - y)^2.

    delta2 = (y - t)                 (saída linear: f'(net) = 1)
    delta1 = (W2^T delta2) * (1 - h^2)  (derivada da tanh)
    """
    X, h = cache["X"], cache["h"]
    P = X.shape[1]
    T = np.atleast_2d(t).reshape(1, -1)
    delta2 = (y - T)                                 # (1, P)
    delta1 = (p.W2.T @ delta2) * (1.0 - h ** 2)      # (N, P)
    dW2 = delta2 @ h.T / P
    db2 = delta2.sum(axis=1, keepdims=True) / P
    dW1 = delta1 @ X.T / P
    db1 = delta1.sum(axis=1, keepdims=True) / P
    return [dW1, db1, dW2, db2]


def mse(x: np.ndarray, t: np.ndarray, p: Params) -> float:
    y, _ = forward(x, p)
    return float(np.mean((np.ravel(t) - np.ravel(y)) ** 2))


def predict(x: np.ndarray, p: Params) -> np.ndarray:
    return np.ravel(forward(x, p)[0])


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
def train(x: np.ndarray, t: np.ndarray, neurons: int = 5, lr: float = 0.05,
          momentum: float = 0.9, epochs: int = 20000, tol: float = 1e-3,
          seed: int = 42, init: str = "nguyen-widrow", mode: str = "batch",
          params0: Params | None = None) -> Resultado:
    """Gradiente descendente com momento:
        dw(k) = -lr * grad + momentum * dw(k-1);   w <- w + dw(k)
    mode='batch': uma atualização por época com o gradiente médio.
    mode='online': uma atualização por padrão, em ordem embaralhada a cada época.
    Para quando MSE < tol (tol=0 desativa) ou ao atingir `epochs`.
    """
    rng = np.random.default_rng(seed)
    p = params0.copy() if params0 is not None else init_weights(neurons, init, rng)
    vel = [np.zeros_like(w) for w in p.as_list()]
    res = Resultado(params=p)

    for ep in range(1, epochs + 1):
        if mode == "batch":
            lotes = [np.arange(len(x))]
        elif mode == "online":
            lotes = [[i] for i in rng.permutation(len(x))]
        else:
            raise ValueError(f"modo desconhecido: {mode}")

        for idx in lotes:
            y, cache = forward(x[idx], p)
            grads = backward(t[idx], y, cache, p)
            for w, v, g in zip(p.as_list(), vel, grads):
                v *= momentum
                v -= lr * g
                w += v

        e = mse(x, t, p)
        res.historico.append(e)
        if not np.isfinite(e) or e > 1e6:
            res.divergiu = True
            break
        if tol > 0 and e < tol:
            res.convergiu = True
            break
    res.epocas = len(res.historico)
    return res


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _eixos_estilo_slide(ax: plt.Axes) -> None:
    """Eixos cruzando na origem, limites [-1, 1] e título 'Dados', como no slide."""
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks(np.arange(-1, 1.01, 0.2))
    ax.set_yticks(np.arange(-1, 1.01, 0.2))
    ax.tick_params(labelsize=7)
    ax.set_title("Dados", fontsize=10, fontweight="bold")


def plot(p: Params | None, caminho: str, x: np.ndarray = X_DADOS,
         t: np.ndarray = T_DADOS, faixa: tuple[float, float] = (0.0, 1.0)) -> None:
    """Pontos amostrados (losangos azuis) e curva aprendida (vermelho grosso)."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    _eixos_estilo_slide(ax)
    if p is not None:
        xs = np.linspace(*faixa, 400)
        ax.plot(xs, predict(xs, p), color="red", lw=4, zorder=1)
    ax.plot(x, t, "D", mfc="none", mec="blue", ms=5, zorder=2)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experimentos do relatório
# ---------------------------------------------------------------------------
def fmt_params(p: Params) -> dict[str, list]:
    return {k: np.round(v, 4).tolist() for k, v in
            zip(["W1", "b1", "W2", "b2"], p.as_list())}


def experimentos(pasta_fig: str = "figuras", saida_json: str = "resultados.json") -> dict:
    os.makedirs(pasta_fig, exist_ok=True)
    R: dict = {"base": BASE}
    x, t = X_DADOS, T_DADOS

    # 0) Dados brutos (figura esquerda do slide)
    plot(None, f"{pasta_fig}/dados.png")

    # 1) Treinamento base
    print("[1] Treinamento base")
    base = train(x, t, **BASE)
    plot(base.params, f"{pasta_fig}/aproximacao_base.png")
    R["treino_base"] = {"epocas": base.epocas, "mse": base.mse_final,
                        "convergiu": base.convergiu, "pesos": fmt_params(base.params),
                        "saidas": np.round(predict(x, base.params), 4).tolist()}
    print(f"    épocas={base.epocas}  MSE={base.mse_final:.2e}")

    # Treino "até o fim" (sem tolerância) para ver até onde o erro cai
    longo = train(x, t, **{**BASE, "tol": 0.0, "epochs": 50000})
    R["treino_longo"] = {"epocas": longo.epocas, "mse": longo.mse_final}

    # 2) Curva de aprendizado (base e longo)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.semilogy(longo.historico, color="gray", lw=1, label="sem parada antecipada")
    ax.semilogy(base.historico, color="red", lw=2, label=f"base (para em MSE < {BASE['tol']:g})")
    ax.axhline(BASE["tol"], ls="--", color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Época (escala log)"); ax.set_ylabel("MSE"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{pasta_fig}/curva_aprendizado.png", dpi=200); plt.close(fig)

    # 3) Número de neurônios ocultos (treino sem tolerância, 50000 épocas)
    print("[3] Variação de neurônios")
    Ns = [1, 2, 3, 5, 10, 20]
    xs = np.linspace(0, 1, 400)
    R["neuronios"] = []
    fig, axs = plt.subplots(2, 3, figsize=(9, 5.4), sharex=True, sharey=True)
    for ax, N in zip(axs.ravel(), Ns):
        r = train(x, t, **{**BASE, "neurons": N, "tol": 0.0, "epochs": 50000})
        R["neuronios"].append({"N": N, "mse": r.mse_final})
        ax.plot(xs, predict(xs, r.params), "r", lw=2)
        ax.plot(x, t, "D", mfc="none", mec="blue", ms=4)
        ax.set_title(f"N = {N}   MSE = {r.mse_final:.1e}", fontsize=9)
        ax.set_ylim(-1.1, 1.0); ax.grid(alpha=0.3)
        print(f"    N={N:2d}  MSE={r.mse_final:.2e}")
    fig.tight_layout(); fig.savefig(f"{pasta_fig}/neuronios.png", dpi=200); plt.close(fig)

    # 4) Taxa de aprendizado x momento (épocas até MSE < tol; mediana de 10 sementes)
    print("[4] Taxa de aprendizado x momento")
    lrs, moms, seeds = [0.01, 0.05, 0.1, 0.3], [0.0, 0.5, 0.9], range(10)
    R["lr_momento"] = []
    for lr in lrs:
        for m in moms:
            eps, ok, div = [], 0, 0
            for s in seeds:
                r = train(x, t, **{**BASE, "lr": lr, "momentum": m, "seed": s, "epochs": 50000})
                div += r.divergiu
                if r.convergiu:
                    ok += 1; eps.append(r.epocas)
            R["lr_momento"].append({"lr": lr, "momento": m, "convergiram": ok,
                                    "divergiram": div, "total": len(seeds),
                                    "epocas_mediana": float(np.median(eps)) if eps else None})
            print(f"    lr={lr:<5} mom={m:<4} conv={ok}/10 div={div} "
                  f"mediana={np.median(eps) if eps else float('nan'):.0f}")

    # 4b) Inicialização e modo de treinamento (10 sementes)
    print("[4b] Inicialização e modo")
    R["init_modo"] = []
    for init in ["uniforme", "nguyen-widrow"]:
        for mode in ["batch", "online"]:
            eps, ok = [], 0
            for s in range(10):
                r = train(x, t, **{**BASE, "init": init, "mode": mode, "seed": s, "epochs": 50000})
                if r.convergiu:
                    ok += 1; eps.append(r.epocas)
            R["init_modo"].append({"init": init, "modo": mode, "convergiram": ok, "total": 10,
                                   "epocas_mediana": float(np.median(eps)) if eps else None})
            print(f"    {init:14s} {mode:7s} conv={ok}/10 mediana={np.median(eps) if eps else float('nan'):.0f}")

    # 5) Generalização: leave-one-out (LOO) para cada N
    print("[5] Leave-one-out")
    R["loo"] = []
    for N in Ns:
        erros = []
        for i in range(len(x)):
            m = np.ones(len(x), bool); m[i] = False
            r = train(x[m], t[m], **{**BASE, "neurons": N, "tol": 0.0, "epochs": 20000})
            erros.append((predict(x[i:i + 1], r.params)[0] - t[i]) ** 2)
        R["loo"].append({"N": N, "mse_loo": float(np.mean(erros)),
                         "max_erro": float(np.sqrt(np.max(erros)))})
        print(f"    N={N:2d}  MSE_LOO={np.mean(erros):.2e}")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    mse_tr = [max(d["mse"], 1e-7) for d in R["neuronios"]]  # piso só para visualização
    ax.semilogy(Ns, mse_tr, "o-", label="MSE de treino")
    ax.semilogy(Ns, [d["mse_loo"] for d in R["loo"]], "s-", color="red", label="MSE leave-one-out")
    ax.set_xlabel("Neurônios na camada oculta (N)"); ax.set_ylabel("MSE")
    ax.set_ylim(5e-8, 1); ax.set_xticks(Ns); ax.grid(True, which="both", alpha=0.3)
    ax.annotate("treino ≈ 0\n(interpolação)", (20, 1e-7), (14, 1e-5), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{pasta_fig}/loo.png", dpi=200); plt.close(fig)

    # 6) Extrapolação em [-1, 1]
    print("[6] Extrapolação")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    _eixos_estilo_slide(ax); ax.set_ylim(-1.5, 1.0)
    ax.set_yticks(np.arange(-1.5, 1.01, 0.5))
    xe = np.linspace(-1, 1, 400)
    for N, cor in [(3, "tab:green"), (5, "red"), (20, "tab:purple")]:
        r = train(x, t, **{**BASE, "neurons": N, "tol": 0.0, "epochs": 50000})
        ax.plot(xe, predict(xe, r.params), color=cor, lw=2, label=f"N = {N}")
    ax.axvspan(0, 1, color="yellow", alpha=0.15, lw=0)
    ax.plot(x, t, "D", mfc="none", mec="blue", ms=5)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(f"{pasta_fig}/extrapolacao.png", dpi=200); plt.close(fig)

    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"Resultados salvos em {saida_json}; figuras em {pasta_fig}/")
    return R


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trabalho 06 - Aproximação funcional com MLP")
    ap.add_argument("--neurons", type=int, default=BASE["neurons"])
    ap.add_argument("--lr", type=float, default=BASE["lr"])
    ap.add_argument("--momentum", type=float, default=BASE["momentum"])
    ap.add_argument("--epochs", type=int, default=BASE["epochs"])
    ap.add_argument("--tol", type=float, default=BASE["tol"])
    ap.add_argument("--seed", type=int, default=BASE["seed"])
    ap.add_argument("--init", choices=["uniforme", "nguyen-widrow"], default=BASE["init"])
    ap.add_argument("--mode", choices=["batch", "online"], default=BASE["mode"])
    ap.add_argument("--saida", default="aproximacao.png", help="arquivo da figura")
    ap.add_argument("--experimentos", action="store_true",
                    help="roda todos os experimentos do relatório")
    a = ap.parse_args()

    if a.experimentos:
        experimentos()
        return

    r = train(X_DADOS, T_DADOS, a.neurons, a.lr, a.momentum, a.epochs,
              a.tol, a.seed, a.init, a.mode)
    status = "convergiu" if r.convergiu else ("DIVERGIU" if r.divergiu else "máx. de épocas")
    print(f"MLP 1-{a.neurons}-1 | {status} em {r.epocas} épocas | MSE = {r.mse_final:.4e}")
    print(" x     alvo     saída")
    for xi, ti, yi in zip(X_DADOS, T_DADOS, predict(X_DADOS, r.params)):
        print(f"{xi:.1f}  {ti:+.4f}  {yi:+.4f}")
    plot(r.params, a.saida)
    print(f"Figura salva em {a.saida}")


if __name__ == "__main__":
    main()
