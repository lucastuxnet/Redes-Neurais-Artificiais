#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Trabalho 05 - Classificacao de Padroes com a Rede Neural Adaline
 Disciplina EL056 - Redes Neurais Artificiais - PPGEELT/UFU
-------------------------------------------------------------------------------
 Treina uma rede Adaline (ADAptive LInear NEuron, Widrow & Hoff, 1960) sobre a
 base de dados B2, plota o erro quadratico total (EQT) durante o treinamento e
 realiza o teste da rede treinada (holdout estratificado + leave-one-out).

 Uso:
     python adaline_b2.py                      # execucao completa (padrao)
     python adaline_b2.py --eta 0.01 --eps 1e-7
     python adaline_b2.py --base Basedados_B2.xlsx --figs figs --out resultados
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib

if __name__ == "__main__":          # em notebook, preserva o backend interativo
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Padronizacao visual das figuras
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

COR_C1 = "#1b6ca8"   # classe +1
COR_C2 = "#c0392b"   # classe -1
COR_LIN = "#2c3e50"


# =============================================================================
# 1. REDE NEURAL ADALINE
# =============================================================================
@dataclass
class Adaline:
    """Adaline de camada unica com regra Delta (Widrow-Hoff / LMS).

    Convencao adotada (Silva, Spatti & Flauzino, 2016): a entrada e' aumentada
    com x0 = -1, de modo que w[0] representa o limiar de ativacao (theta) e o
    potencial de ativacao e' u = W . X = sum_i(w_i*x_i) - theta.

    Treinamento  -> funcao de ativacao LINEAR  (g(u) = u)
    Operacao     -> funcao de ativacao SIGNAL  (y = sgn(u) em {-1, +1})
    """

    eta: float = 0.0025          # taxa de aprendizagem
    epsilon: float = 1e-6        # precisao requerida para |EQM(k) - EQM(k-1)|
    max_epocas: int = 10_000     # limite de seguranca
    seed: int = 42               # semente dos pesos iniciais aleatorios

    # --- atributos aprendidos ------------------------------------------------
    w: np.ndarray | None = None
    w_inicial: np.ndarray | None = None
    hist_eqt: list[float] = field(default_factory=list)   # erro quadratico TOTAL
    hist_eqm: list[float] = field(default_factory=list)   # erro quadratico MEDIO
    epocas: int = 0
    convergiu: bool = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _aumenta(X: np.ndarray) -> np.ndarray:
        """Concatena a entrada fixa x0 = -1 (associada ao limiar theta)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([-np.ones((X.shape[0], 1)), X])

    def potencial(self, X: np.ndarray) -> np.ndarray:
        """Potencial de ativacao u = W . X (saida linear, usada no treino)."""
        return self._aumenta(X) @ self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Saida da rede em operacao: y = sgn(u) in {-1, +1}."""
        u = self.potencial(X)
        return np.where(u >= 0.0, 1.0, -1.0)

    def eqm(self, X: np.ndarray, d: np.ndarray) -> float:
        """Erro quadratico medio E(w) = (1/p) * sum_p (d_p - u_p)^2."""
        return float(np.mean((np.asarray(d, float) - self.potencial(X)) ** 2))

    def eqt(self, X: np.ndarray, d: np.ndarray) -> float:
        """Erro quadratico total EQT(w) = sum_p (d_p - u_p)^2."""
        return float(np.sum((np.asarray(d, float) - self.potencial(X)) ** 2))

    # ----------------------------------------------------------------- treino
    def fit(self, X: np.ndarray, d: np.ndarray, verbose: bool = False) -> "Adaline":
        """Treinamento supervisionado sequencial (padrao a padrao) pela regra Delta.

            w <- w + eta * (d_p - u_p) * x_p

        Criterio de parada: |EQM(atual) - EQM(anterior)| <= epsilon.
        """
        Xa = self._aumenta(X)
        d = np.asarray(d, dtype=float).ravel()
        rng = np.random.default_rng(self.seed)

        # Pesos iniciais aleatorios pequenos (evita simetria e satura menos o erro)
        self.w = rng.uniform(-0.5, 0.5, size=Xa.shape[1])
        self.w_inicial = self.w.copy()

        self.hist_eqt, self.hist_eqm = [], []
        eqm_ant = float(np.mean((d - Xa @ self.w) ** 2))
        self.hist_eqm.append(eqm_ant)
        self.hist_eqt.append(eqm_ant * len(d))

        self.epocas = 0
        self.convergiu = False

        while self.epocas < self.max_epocas:
            # ---- uma epoca: percorre todos os p padroes de treinamento ------
            for xa, alvo in zip(Xa, d):
                u = float(xa @ self.w)          # ativacao LINEAR
                self.w += self.eta * (alvo - u) * xa

            self.epocas += 1
            eqm_atual = float(np.mean((d - Xa @ self.w) ** 2))
            self.hist_eqm.append(eqm_atual)
            self.hist_eqt.append(eqm_atual * len(d))

            if verbose and self.epocas % 50 == 0:
                print(f"  epoca {self.epocas:5d} | EQM = {eqm_atual:.8f}")

            if abs(eqm_atual - eqm_ant) <= self.epsilon:
                self.convergiu = True
                break
            eqm_ant = eqm_atual

        return self

    # --------------------------------------------------------------- reta/ODE
    def reta_separacao(self, s1: np.ndarray) -> np.ndarray:
        """Fronteira de decisao u = 0 no plano (s1, s2):
        -theta + w1*s1 + w2*s2 = 0  =>  s2 = (theta - w1*s1) / w2
        """
        theta, w1, w2 = self.w[0], self.w[1], self.w[2]
        return (theta - w1 * np.asarray(s1, float)) / w2

    def __str__(self) -> str:
        return (f"Adaline(theta={self.w[0]:+.4f}, w1={self.w[1]:+.4f}, "
                f"w2={self.w[2]:+.4f}, epocas={self.epocas})")


# =============================================================================
# 2. PERCEPTRON (apenas para efeito comparativo na discussao)
# =============================================================================
@dataclass
class Perceptron:
    eta: float = 0.01
    max_epocas: int = 10_000
    seed: int = 42
    w: np.ndarray | None = None
    epocas: int = 0
    hist_erros: list[int] = field(default_factory=list)

    def fit(self, X, d):
        Xa = Adaline._aumenta(X)
        d = np.asarray(d, float).ravel()
        rng = np.random.default_rng(self.seed)
        self.w = rng.uniform(-0.5, 0.5, size=Xa.shape[1])
        self.hist_erros, self.epocas = [], 0
        for _ in range(self.max_epocas):
            erros = 0
            for xa, alvo in zip(Xa, d):
                y = 1.0 if xa @ self.w >= 0 else -1.0   # ativacao DEGRAU
                if y != alvo:
                    self.w += self.eta * (alvo - y) * xa
                    erros += 1
            self.epocas += 1
            self.hist_erros.append(erros)
            if erros == 0:
                break
        return self

    def predict(self, X):
        return np.where(Adaline._aumenta(X) @ self.w >= 0, 1.0, -1.0)

    def reta_separacao(self, s1):
        return (self.w[0] - self.w[1] * np.asarray(s1, float)) / self.w[2]


# =============================================================================
# 3. METRICAS
# =============================================================================
def matriz_confusao(d, y):
    """Retorna (VP, FN, FP, VN) considerando +1 como classe positiva."""
    d, y = np.asarray(d).ravel(), np.asarray(y).ravel()
    vp = int(np.sum((d == 1) & (y == 1)))
    fn = int(np.sum((d == 1) & (y == -1)))
    fp = int(np.sum((d == -1) & (y == 1)))
    vn = int(np.sum((d == -1) & (y == -1)))
    return vp, fn, fp, vn


def metricas(d, y) -> dict:
    vp, fn, fp, vn = matriz_confusao(d, y)
    n = vp + fn + fp + vn
    acc = (vp + vn) / n if n else 0.0
    prec = vp / (vp + fp) if (vp + fp) else 0.0
    rec = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acuracia": acc, "precisao": prec, "revocacao": rec, "f1": f1,
            "VP": vp, "FN": fn, "FP": fp, "VN": vn}


def split_estratificado(X, d, frac_treino=0.7, seed=0):
    """Particiona mantendo a proporcao original das duas classes."""
    rng = np.random.default_rng(seed)
    idx_tr, idx_te = [], []
    for c in (-1, 1):
        idx = np.where(np.asarray(d).ravel() == c)[0]
        rng.shuffle(idx)
        k = int(round(frac_treino * len(idx)))
        idx_tr.extend(idx[:k])
        idx_te.extend(idx[k:])
    idx_tr, idx_te = np.array(idx_tr), np.array(idx_te)
    rng.shuffle(idx_tr)
    rng.shuffle(idx_te)
    return X[idx_tr], d[idx_tr], X[idx_te], d[idx_te], idx_tr, idx_te


# =============================================================================
# 4. DADOS
# =============================================================================
def carregar_base(caminho="Basedados_B2.xlsx"):
    """Le a base B2 e devolve (X, d, DataFrame)."""
    df = pd.read_excel(caminho)
    df.columns = [str(c).strip().lower() for c in df.columns]
    X = df[["s1", "s2"]].to_numpy(float)
    d = df["t"].to_numpy(float)
    return X, d, df


# =============================================================================
# 5. FIGURAS
# =============================================================================
def fig_dispersao(X, d, path, modelo=None, titulo="Base de dados B2"):
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    m = d == 1
    ax.scatter(X[m, 0], X[m, 1], c=COR_C1, marker="o", s=45,
               edgecolor="white", linewidth=.7, label="Classe A  ($t=+1$)", zorder=3)
    ax.scatter(X[~m, 0], X[~m, 1], c=COR_C2, marker="s", s=45,
               edgecolor="white", linewidth=.7, label="Classe B  ($t=-1$)", zorder=3)
    for i, (a, b) in enumerate(X, start=1):
        ax.annotate(str(i), (a, b), textcoords="offset points", xytext=(5, 4),
                    fontsize=5.8, color="#566573")
    if modelo is not None:
        s1 = np.linspace(X[:, 0].min() - .35, X[:, 0].max() + .35, 200)
        ax.plot(s1, modelo.reta_separacao(s1), color=COR_LIN, lw=1.7,
                label="Fronteira de decis\u00e3o", zorder=2)
        ax.set_ylim(X[:, 1].min() - .4, X[:, 1].max() + .4)
    ax.set_xlabel("$s_1$"); ax.set_ylabel("$s_2$")
    ax.set_title(titulo, fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_erro(hist_eqt, hist_eqm, eps, path,
             titulo="Erro quadr\u00e1tico total durante o treinamento"):
    """Figura principal exigida no enunciado: EQT x \u00e9pocas + crit\u00e9rio de parada."""
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ep = np.arange(len(hist_eqt))

    axs[0].plot(ep, hist_eqt, color=COR_C1, lw=1.7)
    axs[0].axhline(hist_eqt[-1], color="#95a5a6", ls=":", lw=1.1)
    axs[0].annotate(f"EQT final = {hist_eqt[-1]:.3f}",
                    xy=(len(ep) * 0.45, hist_eqt[-1]), xytext=(0, 9),
                    textcoords="offset points", fontsize=7.5, color="#566573")
    axs[0].set_xlabel("\u00c9poca"); axs[0].set_ylabel("EQT")
    axs[0].set_title("(a) erro quadr\u00e1tico total", fontsize=9)

    delta = np.abs(np.diff(hist_eqm))
    axs[1].semilogy(np.arange(1, len(delta) + 1), delta, color=COR_C2, lw=1.5)
    axs[1].axhline(eps, color=COR_LIN, ls="--", lw=1.2,
                   label=fr"$\varepsilon$ = {eps:g}")
    axs[1].set_xlabel("\u00c9poca"); axs[1].set_ylabel(r"$|EQM(k)-EQM(k-1)|$")
    axs[1].set_title("(b) crit\u00e9rio de parada", fontsize=9)
    axs[1].legend(fontsize=7.5)

    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path); plt.close(fig)


def fig_eta(curvas, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for eta, h in curvas.items():
        ax.semilogy(np.arange(len(h)), h, lw=1.5, label=fr"$\eta$ = {eta:g}")
    ax.set_xlabel("\u00c9poca"); ax.set_ylabel("EQT (log)")
    ax.set_title("Influ\u00eancia da taxa de aprendizagem", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_teste(Xtr, dtr, Xte, dte, modelo, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.scatter(Xtr[dtr == 1, 0], Xtr[dtr == 1, 1], c=COR_C1, marker="o", s=35,
               alpha=.35, label="Treino $t=+1$")
    ax.scatter(Xtr[dtr == -1, 0], Xtr[dtr == -1, 1], c=COR_C2, marker="s", s=35,
               alpha=.35, label="Treino $t=-1$")
    y = modelo.predict(Xte)
    ok = y == dte
    ax.scatter(Xte[ok & (dte == 1), 0], Xte[ok & (dte == 1), 1], c=COR_C1,
               marker="o", s=95, edgecolor="k", linewidth=1.0, label="Teste $t=+1$", zorder=4)
    ax.scatter(Xte[ok & (dte == -1), 0], Xte[ok & (dte == -1), 1], c=COR_C2,
               marker="s", s=95, edgecolor="k", linewidth=1.0, label="Teste $t=-1$", zorder=4)
    if np.any(~ok):
        ax.scatter(Xte[~ok, 0], Xte[~ok, 1], facecolor="none", edgecolor="k",
                   s=180, linewidth=1.6, label="Erro de classifica\u00e7\u00e3o", zorder=5)
    s1 = np.linspace(0, 3.0, 200)
    ax.plot(s1, modelo.reta_separacao(s1), color=COR_LIN, lw=1.7, zorder=2)
    ax.set_xlim(-0.05, 3.0); ax.set_ylim(-0.25, 4.15)
    ax.set_xlabel("$s_1$"); ax.set_ylabel("$s_2$")
    ax.set_title("Teste da rede treinada (holdout 70/30)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_comparacao(X, d, net, perc, path):
    """Adaline (solucao de minimos quadrados) x Perceptron (primeira solucao)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    m = d == 1
    ax.scatter(X[m, 0], X[m, 1], c=COR_C1, marker="o", s=45,
               edgecolor="white", linewidth=.7, label="Classe A ($t=+1$)", zorder=3)
    ax.scatter(X[~m, 0], X[~m, 1], c=COR_C2, marker="s", s=45,
               edgecolor="white", linewidth=.7, label="Classe B ($t=-1$)", zorder=3)
    s1 = np.linspace(0, 3.0, 200)
    ax.plot(s1, net.reta_separacao(s1), color=COR_LIN, lw=1.8, label="Adaline (LMS)")
    ax.plot(s1, perc.reta_separacao(s1), color="#7f8c8d", lw=1.6, ls="--",
            label="Perceptron (degrau)")
    ax.set_xlim(-0.05, 3.0); ax.set_ylim(-0.25, 3.3)
    ax.set_xlabel("$s_1$"); ax.set_ylabel("$s_2$")
    ax.set_title("Fronteiras obtidas por Adaline e Perceptron", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_ruido(sigmas, med_a, dp_a, med_p, dp_p, marg_a, marg_p, path):
    """Degrada\u00e7\u00e3o da acur\u00e1cia sob ru\u00eddo gaussiano nos atributos."""
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(sigmas, med_a, color=COR_C1, lw=1.8, marker="o", ms=3.2, label="Adaline")
    ax.fill_between(sigmas, med_a - dp_a, med_a + dp_a, color=COR_C1, alpha=.18)
    ax.plot(sigmas, med_p, color=COR_C2, lw=1.8, ls="--", marker="s", ms=3.2,
            label="Perceptron")
    ax.fill_between(sigmas, med_p - dp_p, med_p + dp_p, color=COR_C2, alpha=.18)
    ax.axvline(marg_a, color=COR_C1, lw=1.0, ls=":", alpha=.8)
    ax.annotate(f"margem da Adaline = {marg_a:.3f}", xy=(marg_a, 0.845),
                xytext=(5, 0), textcoords="offset points", fontsize=7,
                color=COR_C1, va="center")
    ax.set_xlabel("desvio-padr\u00e3o do ru\u00eddo  $\\sigma$")
    ax.set_ylabel("acur\u00e1cia m\u00e9dia")
    ax.set_ylim(0.83, 1.02)
    ax.set_title("Robustez da fronteira a ru\u00eddo nos atributos", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_superficie(X, d, modelo, path):
    """Regioes de decisao + valor do potencial de ativacao u(s1,s2)."""
    g1, g2 = np.meshgrid(np.linspace(-0.1, 3.1, 320), np.linspace(-0.3, 3.3, 320))
    G = np.c_[g1.ravel(), g2.ravel()]
    U = modelo.potencial(G).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    cf = ax.contourf(g1, g2, U, levels=24, cmap="RdBu", alpha=.85)
    ax.contour(g1, g2, U, levels=[0], colors=[COR_LIN], linewidths=1.8)
    plt.colorbar(cf, ax=ax, label="potencial de ativa\u00e7\u00e3o $u$")
    m = d == 1
    ax.scatter(X[m, 0], X[m, 1], c="w", edgecolor=COR_C1, marker="o", s=45, linewidth=1.4)
    ax.scatter(X[~m, 0], X[~m, 1], c="w", edgecolor=COR_C2, marker="s", s=45, linewidth=1.4)
    ax.set_xlabel("$s_1$"); ax.set_ylabel("$s_2$")
    ax.set_title("Regi\u00f5es de decis\u00e3o da Adaline", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================================
# 6. EXPERIMENTOS
# =============================================================================
def experimento_completo(base, dir_figs, dir_out, eta, eps, seed):
    os.makedirs(dir_figs, exist_ok=True)
    os.makedirs(dir_out, exist_ok=True)
    R: dict = {}

    X, d, df = carregar_base(base)
    R["n_amostras"] = int(len(d))
    R["n_classe_pos"] = int(np.sum(d == 1))
    R["n_classe_neg"] = int(np.sum(d == -1))
    R["estatisticas"] = df[["s1", "s2"]].describe().round(4).to_dict()
    print(f"[1] Base carregada: {len(d)} amostras, 2 atributos, "
          f"{R['n_classe_pos']} (+1) / {R['n_classe_neg']} (-1)")
    fig_dispersao(X, d, f"{dir_figs}/fig1_dispersao.png")

    # --- 2. Treinamento com a base completa ---------------------------------
    net = Adaline(eta=eta, epsilon=eps, seed=seed).fit(X, d)
    print(f"[2] Treinamento (base completa): {net}  convergiu={net.convergiu}")
    R["treino_completo"] = {
        "eta": eta, "epsilon": eps, "seed": seed,
        "w_inicial": net.w_inicial.round(4).tolist(),
        "theta": round(float(net.w[0]), 4),
        "w1": round(float(net.w[1]), 4),
        "w2": round(float(net.w[2]), 4),
        "epocas": net.epocas,
        "eqt_inicial": round(net.hist_eqt[0], 4),
        "eqt_final": round(net.hist_eqt[-1], 6),
        "eqm_final": round(net.hist_eqm[-1], 6),
        "convergiu": net.convergiu,
    }
    fig_erro(net.hist_eqt, net.hist_eqm, eps, f"{dir_figs}/fig2_eqt.png")
    fig_dispersao(X, d, f"{dir_figs}/fig3_fronteira.png", modelo=net,
                  titulo="Fronteira de decis\u00e3o ap\u00f3s o treinamento")
    fig_superficie(X, d, net, f"{dir_figs}/fig5_regioes.png")

    y = net.predict(X)
    R["desempenho_treino"] = metricas(d, y)
    print(f"    acuracia na propria base de treinamento: "
          f"{R['desempenho_treino']['acuracia']*100:.1f}%")

    # --- 3. Sensibilidade a taxa de aprendizagem ----------------------------
    curvas, tab_eta = {}, []
    for e in (0.1, 0.05, 0.01, 0.0025, 0.001):
        m = Adaline(eta=e, epsilon=eps, seed=seed).fit(X, d)
        curvas[e] = m.hist_eqt
        tab_eta.append({"eta": e, "epocas": m.epocas,
                        "eqt_final": round(m.hist_eqt[-1], 4),
                        "acuracia": round(metricas(d, m.predict(X))["acuracia"], 4),
                        "convergiu": m.convergiu})
        print(f"[3] eta={e:<7g} epocas={m.epocas:<5d} EQT={m.hist_eqt[-1]:9.4f} "
              f"acc={tab_eta[-1]['acuracia']*100:5.1f}%")
    R["tabela_eta"] = tab_eta
    fig_eta(curvas, f"{dir_figs}/fig4_eta.png")

    # --- 4. Teste: holdout estratificado 70/30 ------------------------------
    linhas = []
    for s in range(10):
        Xtr, dtr, Xte, dte, itr, ite = split_estratificado(X, d, 0.7, seed=s)
        m = Adaline(eta=eta, epsilon=eps, seed=seed).fit(Xtr, dtr)
        mt = metricas(dte, m.predict(Xte))
        linhas.append({"rodada": s + 1, "epocas": m.epocas,
                       "acc_treino": round(metricas(dtr, m.predict(Xtr))["acuracia"], 4),
                       "acc_teste": round(mt["acuracia"], 4),
                       "eqm_final": round(m.hist_eqm[-1], 6)})
        if s == 0:
            fig_teste(Xtr, dtr, Xte, dte, m, f"{dir_figs}/fig6_teste.png")
            R["holdout_detalhe"] = {
                "idx_treino": itr.tolist(), "idx_teste": ite.tolist(),
                "theta": round(float(m.w[0]), 4), "w1": round(float(m.w[1]), 4),
                "w2": round(float(m.w[2]), 4), "epocas": m.epocas,
                "u_teste": np.round(m.potencial(Xte), 4).tolist(),
                "y_teste": m.predict(Xte).astype(int).tolist(),
                "d_teste": dte.astype(int).tolist(),
                "metricas": mt}
    R["holdout"] = linhas
    accs = np.array([l["acc_teste"] for l in linhas])
    R["holdout_resumo"] = {"media": round(float(accs.mean()), 4),
                           "desvio": round(float(accs.std(ddof=1)), 4),
                           "min": float(accs.min()), "max": float(accs.max())}
    print(f"[4] Holdout 70/30 (10 rodadas): acuracia media = "
          f"{accs.mean()*100:.1f}% +/- {accs.std(ddof=1)*100:.1f}%")

    # --- 5. Teste: validacao cruzada leave-one-out --------------------------
    acertos, erros_loo = 0, []
    for i in range(len(d)):
        mask = np.ones(len(d), bool); mask[i] = False
        m = Adaline(eta=eta, epsilon=eps, seed=seed).fit(X[mask], d[mask])
        yi = float(m.predict(X[i:i+1])[0])
        if yi == d[i]:
            acertos += 1
        else:
            erros_loo.append(int(i))
    R["loocv"] = {"acuracia": round(acertos / len(d), 4),
                  "acertos": acertos, "total": int(len(d)),
                  "indices_erro": erros_loo}
    print(f"[5] Leave-one-out ({len(d)} particoes): acuracia = "
          f"{acertos}/{len(d)} = {acertos/len(d)*100:.1f}%")

    # --- 6. Operacao sobre padroes ineditos ---------------------------------
    novos = np.array([[0.50, 0.50], [0.90, 1.10], [1.50, 1.20],
                      [1.60, 1.90], [2.40, 2.40], [2.90, 0.80]])
    R["novos_padroes"] = [
        {"s1": float(a), "s2": float(b), "u": round(float(u), 4), "y": int(yy)}
        for (a, b), u, yy in zip(novos, net.potencial(novos), net.predict(novos))]
    print("[6] Operacao sobre 6 padroes ineditos concluida")

    # --- 7. Comparacao com o Perceptron -------------------------------------
    p = Perceptron(eta=eta, seed=seed).fit(X, d)

    def margens(w):
        """Distancia euclidiana com sinal de cada padrao a fronteira u = 0."""
        return (Adaline._aumenta(X) @ w) / np.linalg.norm(w[1:])

    marg_a, marg_p = margens(net.w), margens(p.w)
    va = net.w[1:] / np.linalg.norm(net.w[1:])
    vp = p.w[1:] / np.linalg.norm(p.w[1:])
    R["perceptron"] = {"epocas": p.epocas,
                       "acuracia": round(metricas(d, p.predict(X))["acuracia"], 4),
                       "theta": round(float(p.w[0]), 4),
                       "w1": round(float(p.w[1]), 4),
                       "w2": round(float(p.w[2]), 4)}
    R["margens"] = {
        "adaline_min": round(float(np.min(np.abs(marg_a))), 4),
        "perceptron_min": round(float(np.min(np.abs(marg_p))), 4),
        "padrao_critico": int(np.argmin(np.abs(marg_a)) + 1),
        "angulo_graus": round(float(np.degrees(np.arccos(np.clip(va @ vp, -1, 1)))), 2)}
    fig_comparacao(X, d, net, p, f"{dir_figs}/fig7_comparacao.png")
    print(f"[7] Perceptron: {p.epocas} epocas, acuracia = "
          f"{R['perceptron']['acuracia']*100:.1f}% | margem minima: "
          f"Adaline {R['margens']['adaline_min']:.4f} x "
          f"Perceptron {R['margens']['perceptron_min']:.4f}")

    # --- 8. Estabilidade dos pesos frente a inicializacao --------------------
    ws = []
    for s in range(20):
        m = Adaline(eta=eta, epsilon=eps, seed=s).fit(X, d)
        ws.append([m.w[0], m.w[1], m.w[2], m.epocas, m.hist_eqm[-1]])
    ws = np.array(ws)
    R["estabilidade"] = {
        "theta_media": round(float(ws[:, 0].mean()), 4),
        "theta_desvio": round(float(ws[:, 0].std(ddof=1)), 5),
        "w1_media": round(float(ws[:, 1].mean()), 4),
        "w1_desvio": round(float(ws[:, 1].std(ddof=1)), 5),
        "w2_media": round(float(ws[:, 2].mean()), 4),
        "w2_desvio": round(float(ws[:, 2].std(ddof=1)), 5),
        "epocas_media": round(float(ws[:, 3].mean()), 1),
        "epocas_desvio": round(float(ws[:, 3].std(ddof=1)), 1),
        "eqm_media": round(float(ws[:, 4].mean()), 6),
        "eqm_desvio": round(float(ws[:, 4].std(ddof=1)), 8)}
    print(f"[8] 20 inicializacoes distintas: EQM = {ws[:,4].mean():.6f} "
          f"+/- {ws[:,4].std(ddof=1):.2e}")

    # --- 9. Solucao analitica de minimos quadrados (referencia) -------------
    w_ot = np.linalg.pinv(Adaline._aumenta(X)) @ d
    eqm_ot = float(np.mean((d - Adaline._aumenta(X) @ w_ot) ** 2))
    R["solucao_analitica"] = {
        "theta": round(float(w_ot[0]), 4), "w1": round(float(w_ot[1]), 4),
        "w2": round(float(w_ot[2]), 4), "eqm": round(eqm_ot, 6),
        "eqm_gradiente": round(net.hist_eqm[-1], 6),
        "erro_relativo_pct": round(abs(net.hist_eqm[-1] - eqm_ot) / eqm_ot * 100, 4)}
    print(f"[9] Solucao analitica: EQM = {eqm_ot:.6f} | obtida por gradiente: "
          f"{net.hist_eqm[-1]:.6f} (desvio de "
          f"{R['solucao_analitica']['erro_relativo_pct']:.3f}%)")

    # --- 10. Robustez da fronteira a ruido nos atributos ---------------------
    # As redes ja treinadas (dados limpos) sao submetidas a entradas
    # corrompidas por ruido gaussiano; mede-se a queda da acuracia.
    sigmas = np.linspace(0.0, 0.60, 13)
    rep = 200
    rng = np.random.default_rng(seed)
    med_a, dp_a, med_p, dp_p = [], [], [], []
    for sg in sigmas:
        aa, pp = [], []
        for _ in range(rep):
            Xr = X + rng.normal(0.0, sg, size=X.shape) if sg > 0 else X
            aa.append(metricas(d, net.predict(Xr))["acuracia"])
            pp.append(metricas(d, p.predict(Xr))["acuracia"])
        med_a.append(np.mean(aa)); dp_a.append(np.std(aa))
        med_p.append(np.mean(pp)); dp_p.append(np.std(pp))
    med_a, dp_a = np.array(med_a), np.array(dp_a)
    med_p, dp_p = np.array(med_p), np.array(dp_p)
    fig_ruido(sigmas, med_a, dp_a, med_p, dp_p,
              R["margens"]["adaline_min"], R["margens"]["perceptron_min"],
              f"{dir_figs}/fig8_ruido.png")

    def sigma_ate(med, alvo=0.95):
        """Maior sigma em que a acuracia media ainda se mantem acima de alvo."""
        ok = sigmas[med >= alvo]
        return round(float(ok.max()), 3) if len(ok) else 0.0

    R["ruido"] = {
        "sigmas": [round(float(v), 3) for v in sigmas],
        "adaline_media": [round(float(v), 4) for v in med_a],
        "perceptron_media": [round(float(v), 4) for v in med_p],
        "repeticoes": rep,
        "sigma_95_adaline": sigma_ate(med_a),
        "sigma_95_perceptron": sigma_ate(med_p),
        "acc_sigma_010": {"adaline": round(float(med_a[2]), 4),
                          "perceptron": round(float(med_p[2]), 4)},
        "acc_sigma_030": {"adaline": round(float(med_a[6]), 4),
                          "perceptron": round(float(med_p[6]), 4)}}
    print(f"[10] Ruido: acuracia >= 95% ate sigma = {R['ruido']['sigma_95_adaline']} "
          f"(Adaline) x {R['ruido']['sigma_95_perceptron']} (Perceptron)")

    with open(f"{dir_out}/resultados.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)

    tab = df.copy()
    tab["u"] = np.round(net.potencial(X), 4)
    tab["y"] = net.predict(X).astype(int)
    tab["acertou"] = (tab["y"] == tab["t"])
    tab.to_csv(f"{dir_out}/saidas_treinamento.csv", index=False)

    print(f"\nFiguras salvas em '{dir_figs}/', resultados em '{dir_out}/'.")
    return R


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Adaline aplicado a base B2")
    ap.add_argument("--base", default="Basedados_B2.xlsx")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--eta", type=float, default=0.0025)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print("=" * 72)
    print(" Trabalho 05 - Adaline aplicado a classificacao de padroes (base B2)")
    print("=" * 72)
    experimento_completo(a.base, a.figs, a.out, a.eta, a.eps, a.seed)


if __name__ == "__main__":
    main()
