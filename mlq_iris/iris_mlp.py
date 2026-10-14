#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 08 -- Reconhecimento de Flores Iris com Perceptron Multicamadas (MLP)
==============================================================================

Disciplina : EL056 - Redes Neurais Artificiais (PPGEELT/UFU)
Professor  : Prof. Dr. Keiji Yamanaka
Aluno      : Lucas Albino Martins (12622EEL006)

MLP implementada do zero em NumPy: camada de saida Softmax (estavel),
funcao de custo Entropia Cruzada categorial e retropropagacao manual com
gradiente descendente em mini-batch (+ momentum) e early stopping.
O scikit-learn e' usado APENAS para os baselines de comparacao.

Exemplos de execucao
--------------------
    pip install -r requirements.txt

    # configuracao padrao (identica ao notebook iris_mlp.ipynb)
    python iris_mlp.py --data iris_data.xlsx

    # variacoes
    python iris_mlp.py --hidden 10,6 --activation relu --lr 0.01 --kfold 5
    python iris_mlp.py --no-corrigir-uci --sem-experimentos --outdir figs/

Saidas
------
    * metricas no terminal (logging);
    * figuras em ``--outdir`` (PDF vetorial + PNG 300 dpi);
    * ``resultados.json`` com hiperparametros e metricas (alimenta o LaTeX
      por meio de ``gerar_macros_latex.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # backend nao interativo (execucao em terminal)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger("iris_mlp")

# --------------------------------------------------------------------------
# Constantes do problema
# --------------------------------------------------------------------------
SEED: int = 42
ATRIBUTOS: List[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES: List[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
CORES: List[str] = ["#0072B2", "#E69F00", "#009E73"]  # paleta Okabe-Ito
# Correcoes documentadas no iris_names.txt (indices 0-based das amostras 35 e 38)
CORRECOES_UCI: Dict[int, List[float]] = {
    34: [4.9, 3.1, 1.5, 0.2],
    37: [4.9, 3.6, 1.4, 0.1],
}
ESTATISTICAS_UCI: Dict[str, Tuple[float, float, float, float, float]] = {
    # min, max, media, desvio-padrao, correlacao com a classe
    "sepal_length": (4.3, 7.9, 5.84, 0.83, 0.7826),
    "sepal_width": (2.0, 4.4, 3.05, 0.43, -0.4194),
    "petal_length": (1.0, 6.9, 3.76, 1.76, 0.9490),
    "petal_width": (0.1, 2.5, 1.20, 0.76, 0.9565),
}

# Configuracao padrao (compartilhada com o notebook)
CONFIG_PADRAO: Dict[str, object] = {
    "hidden": [8],
    "activation": "tanh",
    "lr": 0.05,
    "momentum": 0.9,
    "epochs": 2000,
    "batch_size": 16,
    "patience": 100,
}


# ==========================================================================
# 1. Dados
# ==========================================================================
def carregar_dados(caminho: str | Path, corrigir_uci: bool = True) -> pd.DataFrame:
    """Carrega o ``iris_data.xlsx`` (sem cabecalho) e converte atributos para float.

    Parameters
    ----------
    caminho : str or Path
        Caminho da planilha (150 linhas x 5 colunas, sem cabecalho).
    corrigir_uci : bool, default True
        Se ``True``, aplica as correcoes das amostras 35 e 38 documentadas no
        ``iris_names.txt`` (valores do artigo original de Fisher, 1936).

    Returns
    -------
    pandas.DataFrame
        Colunas ``ATRIBUTOS`` (float64) + ``classe`` (str) + ``y`` (int 0..2).
    """
    df = pd.read_excel(caminho, header=None, names=ATRIBUTOS + ["classe"], engine="openpyxl")
    for col in ATRIBUTOS:  # celulas armazenadas como TEXTO -> float
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="raise"
        ).astype(np.float64)
    df["classe"] = df["classe"].astype(str).str.strip()
    if df.shape != (150, 5) or df.isna().any().any():
        raise ValueError(f"Planilha inesperada: shape={df.shape}, NaN={df.isna().sum().sum()}")
    if sorted(df["classe"].unique()) != CLASSES:
        raise ValueError(f"Rotulos inesperados: {df['classe'].unique()}")
    if corrigir_uci:
        for idx, valores in CORRECOES_UCI.items():
            antes = df.loc[idx, ATRIBUTOS].to_numpy(dtype=float)
            df.loc[idx, ATRIBUTOS] = valores
            LOGGER.info("Amostra %d corrigida: %s -> %s", idx + 1, antes.tolist(), valores)
    df["y"] = df["classe"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return df


def one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    r"""Codificacao one-hot: :math:`Y_{nk} = 1` se :math:`y_n = k`, 0 caso contrario."""
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def split_estratificado(
    y: np.ndarray, fracoes: Tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divisao estratificada treino/validacao/teste com semente fixa.

    Para 50 amostras por classe e fracoes 70/15/15, resultam 35/8/7 por classe
    (105/24/21 no total); o arredondamento e' feito por classe para garantir
    proporcoes identicas de cada especie em todos os subconjuntos.

    Returns
    -------
    tuple of numpy.ndarray
        Indices de treino, validacao e teste.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr = int(np.floor(fracoes[0] * idx.size + 0.5))
        n_va = int(np.floor(fracoes[1] * idx.size + 0.5))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (rng.permutation(np.concatenate(tr)), rng.permutation(np.concatenate(va)),
            rng.permutation(np.concatenate(te)))


def kfold_estratificado(y: np.ndarray, k: int = 5, seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Gera ``k`` pares (indices_treino, indices_teste) estratificados por classe."""
    rng = np.random.default_rng(seed)
    partes_por_classe = [np.array_split(rng.permutation(np.flatnonzero(y == c)), k) for c in np.unique(y)]
    folds = []
    for i in range(k):
        te = np.concatenate([p[i] for p in partes_por_classe])
        tr = np.concatenate([np.concatenate([p[j] for j in range(k) if j != i]) for p in partes_por_classe])
        folds.append((rng.permutation(tr), rng.permutation(te)))
    return folds


class ZScore:
    r"""Padronizacao z-score :math:`x' = (x-\mu_{treino})/\sigma_{treino}`.

    Os parametros sao estimados SOMENTE no conjunto de treino, evitando
    vazamento de informacao (data leakage) da validacao/teste.
    """

    def __init__(self) -> None:
        self.media: Optional[np.ndarray] = None
        self.desvio: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScore":
        self.media = X.mean(axis=0)
        self.desvio = X.std(axis=0)
        self.desvio[self.desvio == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.media) / self.desvio

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def preprocessar(
    X: np.ndarray, y: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray, idx_te: np.ndarray
) -> Dict[str, object]:
    """Aplica z-score (ajustado no treino) e one-hot aos tres subconjuntos."""
    scaler = ZScore().fit(X[idx_tr])
    return {
        "scaler": scaler,
        "Xtr": scaler.transform(X[idx_tr]), "Ytr": one_hot(y[idx_tr]), "ytr": y[idx_tr],
        "Xva": scaler.transform(X[idx_va]), "Yva": one_hot(y[idx_va]), "yva": y[idx_va],
        "Xte": scaler.transform(X[idx_te]), "Yte": one_hot(y[idx_te]), "yte": y[idx_te],
    }


# ==========================================================================
# 2. Funcoes de ativacao, Softmax e Entropia Cruzada
# ==========================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    r"""Sigmoide logistica :math:`\sigma(z) = 1/(1+e^{-z})` (com clipping numerico)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    r""":math:`\sigma'(z) = \sigma(z)\,[1-\sigma(z)]`."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    r"""Tangente hiperbolica :math:`\tanh(z)`."""
    return np.tanh(z)


def tanh_deriv(z: np.ndarray) -> np.ndarray:
    r""":math:`\tanh'(z) = 1-\tanh^2(z)`."""
    return 1.0 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    r"""ReLU :math:`\max(0, z)`."""
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    r""":math:`\mathrm{ReLU}'(z) = \mathbb{1}[z>0]` (subgradiente 0 em z=0)."""
    return (z > 0).astype(z.dtype)


ATIVACOES: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
}


def softmax(z: np.ndarray) -> np.ndarray:
    r"""Softmax numericamente estavel, aplicada por linha.

    .. math:: \hat y_k = \frac{e^{z_k - \max_j z_j}}{\sum_{i=1}^{K} e^{z_i - \max_j z_j}}

    A subtracao do maximo nao altera o resultado (invariancia a translacao)
    e evita overflow de ``exp`` para logits grandes.
    """
    z_desl = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z_desl)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-12) -> float:
    r"""Entropia cruzada categorial media com rotulos one-hot.

    .. math:: L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} y_{nk}\,\ln \hat y_{nk}

    ``y_hat`` e' limitado a ``[eps, 1]`` para evitar ``log(0)``.
    """
    return float(-np.mean(np.sum(y * np.log(np.clip(y_hat, eps, 1.0)), axis=1)))


# ==========================================================================
# 3. MLP
# ==========================================================================
class MLP:
    r"""Perceptron Multicamadas com saida Softmax e custo de entropia cruzada.

    Parameters
    ----------
    n_entradas : int
        Numero de atributos de entrada (4 no Iris).
    ocultas : sequence of int
        Neuronios de cada camada oculta, ex.: ``[8]`` ou ``[10, 6]``.
    n_saidas : int
        Numero de classes (3).
    ativacao : {'sigmoid', 'tanh', 'relu'}
        Ativacao das camadas ocultas.
    seed : int
        Semente da inicializacao dos pesos.

    Notes
    -----
    Inicializacao: He, :math:`W\sim\mathcal N(0, 2/n_{in})`, para ReLU;
    Xavier/Glorot, :math:`W\sim\mathcal N(0, 2/(n_{in}+n_{out}))`, para
    sigmoid/tanh e para a camada Softmax. Vieses iniciam em zero.
    """

    def __init__(self, n_entradas: int, ocultas: Sequence[int], n_saidas: int,
                 ativacao: str = "tanh", seed: int = SEED) -> None:
        if ativacao not in ATIVACOES:
            raise ValueError(f"Ativacao desconhecida: {ativacao}")
        self.ativacao = ativacao
        self.f, self.df = ATIVACOES[ativacao]
        self.dims = [n_entradas, *ocultas, n_saidas]
        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        n_camadas = len(self.dims) - 1
        for l in range(n_camadas):
            fan_in, fan_out = self.dims[l], self.dims[l + 1]
            if ativacao == "relu" and l < n_camadas - 1:
                std = np.sqrt(2.0 / fan_in)  # He
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier/Glorot
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._Z: List[np.ndarray] = []
        self._A: List[np.ndarray] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        r"""Propagacao direta: :math:`z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}`,
        :math:`a^{(l)} = f(z^{(l)})` nas ocultas e :math:`\hat y = \mathrm{softmax}(z^{(L)})`."""
        self._A = [X]
        self._Z = []
        a = X
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.b[l]
            self._Z.append(z)
            a = softmax(z) if l == len(self.W) - 1 else self.f(z)
            self._A.append(a)
        return a

    def backward(self, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r"""Retropropagacao. Na saida, Softmax + CE fornecem
        :math:`\delta^{(L)} = (\hat Y - Y)/N`; nas ocultas,
        :math:`\delta^{(l)} = (\delta^{(l+1)} W^{(l+1)\top}) \odot f'(z^{(l)})`.

        Returns
        -------
        (dW, db) : listas com os gradientes de cada camada.
        """
        N = Y.shape[0]
        delta = (self._A[-1] - Y) / N
        dW: List[np.ndarray] = [np.empty(0)] * len(self.W)
        db: List[np.ndarray] = [np.empty(0)] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            dW[l] = self._A[l].T @ delta
            db[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * self.df(self._Z[l - 1])
        return dW, db

    def update(self, dW: List[np.ndarray], db: List[np.ndarray], lr: float, momentum: float = 0.0) -> None:
        r"""Gradiente descendente com momentum (heavy-ball):
        :math:`v \leftarrow \mu v - \eta \nabla_\theta L`, :math:`\theta \leftarrow \theta + v`."""
        for l in range(len(self.W)):
            self.vW[l] = momentum * self.vW[l] - lr * dW[l]
            self.vb[l] = momentum * self.vb[l] - lr * db[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades a posteriori estimadas (saida Softmax)."""
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classe de maior probabilidade (regra de decisao MAP)."""
        return np.argmax(self.forward(X), axis=1)

    def get_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Copia profunda dos pesos (usada pelo early stopping)."""
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, params: Tuple[List[np.ndarray], List[np.ndarray]]) -> None:
        """Restaura pesos salvos por :meth:`get_params`."""
        self.W = [w.copy() for w in params[0]]
        self.b = [b.copy() for b in params[1]]

    def n_parametros(self) -> int:
        """Numero total de parametros treinaveis."""
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))


# ==========================================================================
# 4. Verificacao do gradiente
# ==========================================================================
def gradient_check(ativacao: str = "tanh", eps: float = 1e-5, seed: int = SEED) -> float:
    r"""Compara o gradiente analitico (backprop) com diferencas finitas centrais.

    .. math:: \frac{\partial L}{\partial\theta} \approx \frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon},
       \qquad e_{rel} = \frac{|g_a-g_n|}{\max(|g_a|+|g_n|,\,10^{-12})}

    Returns
    -------
    float
        Maior erro relativo entre todos os parametros (aceitavel: < 1e-6).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 4))
    Y = one_hot(rng.integers(0, 3, size=6))
    rede = MLP(4, [5, 4], 3, ativacao=ativacao, seed=seed)
    rede.forward(X)
    dW, db = rede.backward(Y)
    erro_max = 0.0
    for params, grads in ((rede.W, dW), (rede.b, db)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                orig = P[i]
                P[i] = orig + eps
                lp = cross_entropy(Y, rede.forward(X))
                P[i] = orig - eps
                lm = cross_entropy(Y, rede.forward(X))
                P[i] = orig
                g_num = (lp - lm) / (2 * eps)
                erro = abs(G[i] - g_num) / max(abs(G[i]) + abs(g_num), 1e-12)
                erro_max = max(erro_max, erro)
    return float(erro_max)


# ==========================================================================
# 5. Treinamento e avaliacao
# ==========================================================================
def acuracia(Y: np.ndarray, y_hat: np.ndarray) -> float:
    """Fracao de acertos entre rotulos one-hot e probabilidades previstas."""
    return float(np.mean(np.argmax(Y, axis=1) == np.argmax(y_hat, axis=1)))


def treinar(
    modelo: MLP, Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray,
    lr: float = 0.05, momentum: float = 0.9, epochs: int = 2000, batch_size: int = 16,
    patience: int = 100, seed: int = SEED, log_every: int = 0,
) -> Dict[str, object]:
    """Treina a MLP com mini-batch SGD + momentum e early stopping.

    O early stopping monitora a perda de validacao; apos ``patience`` epocas
    sem melhora (> 1e-6), o treino e' interrompido e os pesos da melhor epoca
    sao restaurados.

    Returns
    -------
    dict
        Historico (``loss_tr``, ``loss_va``, ``acc_tr``, ``acc_va``),
        ``best_epoch`` e ``epocas_executadas``.
    """
    rng = np.random.default_rng(seed)
    hist: Dict[str, list] = {"loss_tr": [], "loss_va": [], "acc_tr": [], "acc_va": []}
    melhor_loss, melhor_ep, espera = np.inf, 0, 0
    melhores = modelo.get_params()
    n = Xtr.shape[0]
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for ini in range(0, n, batch_size):
            lote = perm[ini:ini + batch_size]
            modelo.forward(Xtr[lote])
            dW, db = modelo.backward(Ytr[lote])
            modelo.update(dW, db, lr, momentum)
        p_tr, p_va = modelo.forward(Xtr), modelo.forward(Xva)
        hist["loss_tr"].append(cross_entropy(Ytr, p_tr))
        hist["loss_va"].append(cross_entropy(Yva, p_va))
        hist["acc_tr"].append(acuracia(Ytr, p_tr))
        hist["acc_va"].append(acuracia(Yva, p_va))
        LOGGER.debug("epoca %4d | loss_tr %.4f loss_va %.4f | acc_tr %.3f acc_va %.3f", ep,
                     hist["loss_tr"][-1], hist["loss_va"][-1], hist["acc_tr"][-1], hist["acc_va"][-1])
        if log_every and ep % log_every == 0:
            LOGGER.info("epoca %4d | loss_tr %.4f loss_va %.4f | acc_tr %.3f acc_va %.3f", ep,
                        hist["loss_tr"][-1], hist["loss_va"][-1], hist["acc_tr"][-1], hist["acc_va"][-1])
        if hist["loss_va"][-1] < melhor_loss - 1e-6:
            melhor_loss, melhor_ep, espera = hist["loss_va"][-1], ep, 0
            melhores = modelo.get_params()
        else:
            espera += 1
            if espera >= patience:
                LOGGER.info("Early stopping na epoca %d (melhor epoca: %d)", ep, melhor_ep)
                break
    modelo.set_params(melhores)
    return {**hist, "best_epoch": melhor_ep, "epocas_executadas": len(hist["loss_tr"]),
            "melhor_loss_va": float(melhor_loss)}


def metricas_classificacao(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> Dict[str, object]:
    """Matriz de confusao, acuracia, precisao, recall e F1 por classe e macro."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "acuracia": float(tp.sum() / cm.sum()),
        "matriz_confusao": cm.tolist(),
        "por_classe": {c: {"precisao": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                           "suporte": int(cm[i].sum())} for i, c in enumerate(CLASSES)},
        "macro": {"precisao": float(prec.mean()), "recall": float(rec.mean()), "f1": float(f1.mean())},
    }


def avaliar(modelo: MLP, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Avalia o modelo: metricas de classificacao + perda de entropia cruzada."""
    proba = modelo.predict_proba(X)
    res = metricas_classificacao(y, np.argmax(proba, axis=1))
    res["loss"] = cross_entropy(one_hot(y), proba)
    return res


def executar_kfold(X: np.ndarray, y: np.ndarray, config: Dict[str, object], k: int = 5,
                   seed: int = SEED) -> Dict[str, object]:
    """Validacao cruzada estratificada k-fold da MLP.

    Em cada fold, a parte de treino e' subdividida (85/15, estratificada)
    para o early stopping; o z-score e' ajustado apenas nessa subparte.
    A matriz de confusao agregada soma as predicoes fora-do-fold das 150 amostras.
    """
    accs, f1s, epocas = [], [], []
    cm_total = np.zeros((3, 3), dtype=int)
    for i, (tr, te) in enumerate(kfold_estratificado(y, k, seed)):
        sub_tr, sub_va, _ = split_estratificado(y[tr], (0.85, 0.15, 0.0), seed + i)
        idx_tr, idx_va = tr[sub_tr], tr[sub_va]
        d = preprocessar(X, y, idx_tr, idx_va, te)
        rede = MLP(X.shape[1], config["hidden"], 3, config["activation"], seed + i)
        h = treinar(rede, d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], config["lr"], config["momentum"],
                    config["epochs"], config["batch_size"], config["patience"], seed + i)
        m = avaliar(rede, d["Xte"], d["yte"])
        cm_total += np.asarray(m["matriz_confusao"])
        accs.append(m["acuracia"])
        f1s.append(m["macro"]["f1"])
        epocas.append(h["best_epoch"])
    return {"acc_folds": accs, "acc_media": float(np.mean(accs)), "acc_dp": float(np.std(accs, ddof=1)),
            "f1_media": float(np.mean(f1s)), "f1_dp": float(np.std(f1s, ddof=1)),
            "best_epoch_medio": float(np.mean(epocas)), "matriz_confusao_total": cm_total.tolist()}


def executar_experimentos(X: np.ndarray, y: np.ndarray, base: Dict[str, object], k: int = 5,
                          seed: int = SEED) -> List[Dict[str, object]]:
    """Varia um fator por vez (neuronios ocultos, taxa de aprendizado, ativacao)
    em torno da configuracao base, avaliando cada variante por k-fold."""
    grade = (
        [("hidden", h) for h in ([2], [4], [8], [16], [32], [10, 6])]
        + [("lr", v) for v in (0.005, 0.01, 0.05, 0.1, 0.5)]
        + [("activation", a) for a in ("sigmoid", "tanh", "relu")]
    )
    linhas = []
    for fator, valor in grade:
        cfg = {**base, fator: valor}
        t0 = time.perf_counter()
        r = executar_kfold(X, y, cfg, k, seed)
        linhas.append({"fator": fator, "valor": valor, "hidden": cfg["hidden"], "activation": cfg["activation"],
                       "lr": cfg["lr"], **{c: r[c] for c in ("acc_media", "acc_dp", "f1_media", "f1_dp",
                                                             "best_epoch_medio")},
                       "tempo_s": round(time.perf_counter() - t0, 2)})
        LOGGER.info("Exp. %-10s = %-8s | acc %.4f +- %.4f | F1 %.4f", fator, valor, r["acc_media"],
                    r["acc_dp"], r["f1_media"])
    return linhas


def baselines(X: np.ndarray, y: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray, k: int = 5,
              seed: int = SEED) -> Dict[str, Dict[str, float]]:
    """Baselines do scikit-learn (APENAS para comparacao): regressao logistica
    multinomial e k-NN (k=5), no mesmo split holdout e nos mesmos folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    modelos = {"Regressao logistica multinomial": lambda: LogisticRegression(max_iter=2000),
               "k-NN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5)}
    res = {}
    for nome, fabrica in modelos.items():
        sc = ZScore().fit(X[idx_tr])
        clf = fabrica().fit(sc.transform(X[idx_tr]), y[idx_tr])
        hold = metricas_classificacao(y[idx_te], clf.predict(sc.transform(X[idx_te])))
        accs = []
        for tr, te in kfold_estratificado(y, k, seed):
            sc = ZScore().fit(X[tr])
            c = fabrica().fit(sc.transform(X[tr]), y[tr])
            accs.append(float(np.mean(c.predict(sc.transform(X[te])) == y[te])))
        res[nome] = {"acc_teste": hold["acuracia"], "f1_macro_teste": hold["macro"]["f1"],
                     "acc_kfold_media": float(np.mean(accs)), "acc_kfold_dp": float(np.std(accs, ddof=1))}
        LOGGER.info("Baseline %-32s | teste %.4f | k-fold %.4f +- %.4f", nome, res[nome]["acc_teste"],
                    res[nome]["acc_kfold_media"], res[nome]["acc_kfold_dp"])
    return res


# ==========================================================================
# 6. Graficos (salvos em PDF vetorial e PNG 300 dpi)
# ==========================================================================
def _salvar(fig: plt.Figure, outdir: Path, nome: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{nome}.png", dpi=300, bbox_inches="tight")


def plotar_scatter_matrix(df: pd.DataFrame, outdir: Path) -> plt.Figure:
    """Matriz de dispersao 4x4 colorida por classe (histogramas na diagonal)."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ai in enumerate(ATRIBUTOS):
        for j, aj in enumerate(ATRIBUTOS):
            ax = axes[i, j]
            for k, c in enumerate(CLASSES):
                s = df[df["y"] == k]
                if i == j:
                    ax.hist(s[ai], bins=12, alpha=0.6, color=CORES[k], label=c)
                else:
                    ax.scatter(s[aj], s[ai], s=12, alpha=0.75, color=CORES[k], label=c)
            if i == 3:
                ax.set_xlabel(f"{aj} (cm)")
            if j == 0:
                ax.set_ylabel(f"{ai} (cm)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _salvar(fig, outdir, "scatter_matrix")
    return fig


def plotar_boxplots(df: pd.DataFrame, outdir: Path) -> plt.Figure:
    """Boxplots de cada atributo por classe."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, a in zip(axes, ATRIBUTOS):
        dados = [df.loc[df["y"] == k, a].to_numpy() for k in range(3)]
        bp = ax.boxplot(dados, patch_artist=True, widths=0.6)
        for patch, cor in zip(bp["boxes"], CORES):
            patch.set_facecolor(cor)
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], ["setosa", "versicolor", "virginica"])
        ax.set_title(f"{a} (cm)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _salvar(fig, outdir, "boxplots")
    return fig


def plotar_correlacao(df: pd.DataFrame, outdir: Path) -> plt.Figure:
    """Mapa de calor da correlacao de Pearson entre atributos e o rotulo (0,1,2)."""
    corr = df[ATRIBUTOS + ["y"]].corr().to_numpy()
    rotulos = ATRIBUTOS + ["classe"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5), rotulos, rotation=45, ha="right")
    ax.set_yticks(range(5), rotulos)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _salvar(fig, outdir, "correlacao")
    return fig


def plotar_curvas(hist: Dict[str, object], outdir: Path) -> plt.Figure:
    """Curvas de perda e acuracia (treino x validacao) com a melhor epoca marcada."""
    ep = np.arange(1, len(hist["loss_tr"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, hist["loss_tr"], color=CORES[0], label="Treino")
    a1.plot(ep, hist["loss_va"], color=CORES[1], label="Validação")
    a1.set_ylabel("Entropia cruzada")
    a1.set_yscale("log")
    a2.plot(ep, hist["acc_tr"], color=CORES[0], label="Treino")
    a2.plot(ep, hist["acc_va"], color=CORES[1], label="Validação")
    a2.set_ylabel("Acurácia")
    for ax in (a1, a2):
        ax.axvline(hist["best_epoch"], color="gray", ls="--", lw=1, label=f"Melhor época ({hist['best_epoch']})")
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    _salvar(fig, outdir, "curvas_treino")
    return fig


def plotar_matriz_confusao(cm: Sequence[Sequence[int]], outdir: Path, nome: str = "matriz_confusao") -> plt.Figure:
    """Heatmap da matriz de confusao (linhas = classe real, colunas = prevista)."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    nomes = ["setosa", "versicolor", "virginica"]
    ax.set_xticks(range(3), nomes)
    ax.set_yticks(range(3), nomes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _salvar(fig, outdir, nome)
    return fig


def plotar_experimentos(linhas: List[Dict[str, object]], outdir: Path) -> plt.Figure:
    """Acuracia media (+- DP) do k-fold para cada fator variado."""
    fatores = [("hidden", "Neurônios ocultos"), ("lr", "Taxa de aprendizado"), ("activation", "Ativação")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (f, titulo) in zip(axes, fatores):
        sel = [r for r in linhas if r["fator"] == f]
        rot = ["-".join(map(str, r["valor"])) if isinstance(r["valor"], list) else str(r["valor"]) for r in sel]
        ax.bar(range(len(sel)), [r["acc_media"] for r in sel], yerr=[r["acc_dp"] for r in sel],
               color=CORES[0], alpha=0.8, capsize=4)
        ax.set_xticks(range(len(sel)), rot)
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Acurácia média (k-fold)")
    axes[0].set_ylim(0.85, 1.02)
    fig.tight_layout()
    _salvar(fig, outdir, "experimentos")
    return fig


def treinar_fronteira(X: np.ndarray, y: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray,
                      idx_te: np.ndarray, config: Dict[str, object], seed: int = SEED
                      ) -> Tuple[MLP, ZScore, Dict[str, object]]:
    """Treina uma MLP auxiliar so com petal length x petal width (para visualizacao 2-D)."""
    X2 = X[:, 2:4]
    d = preprocessar(X2, y, idx_tr, idx_va, idx_te)
    rede = MLP(2, config["hidden"], 3, config["activation"], seed)
    treinar(rede, d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], config["lr"], config["momentum"],
            config["epochs"], config["batch_size"], config["patience"], seed)
    return rede, d["scaler"], avaliar(rede, d["Xte"], d["yte"])


def plotar_fronteira(rede: MLP, scaler: ZScore, X: np.ndarray, y: np.ndarray, idx_te: np.ndarray,
                     outdir: Path) -> plt.Figure:
    """Regioes de decisao da MLP 2-D no plano petal length x petal width (em cm)."""
    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.3, X[:, 3].max() + 0.3
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grade = scaler.transform(np.c_[gx.ravel(), gy.ravel()])
    zz = rede.predict(grade).reshape(gx.shape)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(CORES), alpha=0.25)
    ax.contour(gx, gy, zz, levels=[0.5, 1.5], colors="k", linewidths=0.8)
    eh_teste = np.zeros(len(y), dtype=bool)
    eh_teste[idx_te] = True
    for k, c in enumerate(CLASSES):
        m = y == k
        ax.scatter(X[m & ~eh_teste, 2], X[m & ~eh_teste, 3], color=CORES[k], s=18, label=f"{c} (treino/val.)")
        ax.scatter(X[m & eh_teste, 2], X[m & eh_teste, 3], color=CORES[k], s=55, marker="*",
                   edgecolors="k", linewidths=0.6, label=f"{c} (teste)")
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _salvar(fig, outdir, "fronteira_decisao")
    return fig


# ==========================================================================
# 7. Resultados (JSON)
# ==========================================================================
def comparar_estatisticas_uci(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Estatisticas calculadas (DP amostral, ddof=1) x valores do iris_names.txt."""
    out = {}
    for a in ATRIBUTOS:
        v = df[a]
        out[a] = {"min": float(v.min()), "max": float(v.max()), "media": float(v.mean()),
                  "dp": float(v.std(ddof=1)), "corr_classe": float(np.corrcoef(v, df["y"])[0, 1]),
                  "uci": dict(zip(["min", "max", "media", "dp", "corr_classe"], ESTATISTICAS_UCI[a]))}
    return out


def montar_resultados(config: Dict[str, object], seed: int, corrigir_uci: bool, tamanhos: Dict[str, int],
                      n_parametros: int, grad_check: Dict[str, float], hist: Dict[str, object],
                      teste: Dict[str, object], validacao: Dict[str, object], kfold: Optional[Dict[str, object]],
                      experimentos: Optional[List[Dict[str, object]]], base: Optional[Dict[str, object]],
                      fronteira: Dict[str, object], estatisticas: Dict[str, object]) -> Dict[str, object]:
    """Agrupa hiperparametros e metricas em um dicionario serializavel."""
    return {
        "hiperparametros": {**config, "seed": seed, "corrigir_uci": corrigir_uci,
                            "inicializacao": "He" if config["activation"] == "relu" else "Xavier",
                            "otimizador": "SGD mini-batch + momentum", "n_parametros": n_parametros},
        "tamanhos": tamanhos,
        "gradient_check": grad_check,
        "treinamento": {"best_epoch": hist["best_epoch"], "epocas_executadas": hist["epocas_executadas"],
                        "loss_tr_final": hist["loss_tr"][hist["best_epoch"] - 1],
                        "loss_va_final": hist["loss_va"][hist["best_epoch"] - 1],
                        "acc_tr_final": hist["acc_tr"][hist["best_epoch"] - 1],
                        "acc_va_final": hist["acc_va"][hist["best_epoch"] - 1]},
        "validacao": validacao,
        "teste": teste,
        "kfold": kfold,
        "experimentos": experimentos,
        "baselines": base,
        "fronteira_2d": {"acuracia_teste": fronteira["acuracia"], "matriz_confusao": fronteira["matriz_confusao"]},
        "estatisticas": estatisticas,
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }


# ==========================================================================
# 8. CLI
# ==========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Argumentos de linha de comando."""
    p = argparse.ArgumentParser(description="MLP (Softmax + entropia cruzada) do zero para o Iris.")
    p.add_argument("--data", default="iris_data.xlsx", help="caminho do iris_data.xlsx")
    p.add_argument("--hidden", default="8", help='neuronios ocultos, ex.: "8" ou "10,6"')
    p.add_argument("--activation", choices=sorted(ATIVACOES), default=CONFIG_PADRAO["activation"])
    p.add_argument("--lr", type=float, default=CONFIG_PADRAO["lr"], help="taxa de aprendizado")
    p.add_argument("--momentum", type=float, default=CONFIG_PADRAO["momentum"])
    p.add_argument("--epochs", type=int, default=CONFIG_PADRAO["epochs"], help="maximo de epocas")
    p.add_argument("--batch-size", type=int, default=CONFIG_PADRAO["batch_size"])
    p.add_argument("--patience", type=int, default=CONFIG_PADRAO["patience"], help="paciencia do early stopping")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--corrigir-uci", action=argparse.BooleanOptionalAction, default=True,
                   help="corrige as amostras 35 e 38 (use --no-corrigir-uci para manter)")
    p.add_argument("--kfold", type=int, default=5, help="numero de folds (0 desativa)")
    p.add_argument("--sem-experimentos", action="store_true", help="pula a varredura de hiperparametros")
    p.add_argument("--outdir", default="figs/", help="pasta das figuras")
    p.add_argument("--json", default="resultados.json", help="arquivo de saida com metricas")
    p.add_argument("--log-every", type=int, default=50, help="frequencia (epocas) do log INFO")
    p.add_argument("-v", "--verbose", action="store_true", help="log DEBUG de todas as epocas")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Pipeline completo: dados -> EDA -> treino -> avaliacao -> experimentos -> JSON."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    outdir = Path(args.outdir)
    config = {"hidden": [int(h) for h in str(args.hidden).split(",") if h.strip()],
              "activation": args.activation, "lr": args.lr, "momentum": args.momentum,
              "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience}
    LOGGER.info("Configuracao: %s | seed=%d | corrigir_uci=%s", config, args.seed, args.corrigir_uci)

    # Dados e analise exploratoria
    df = carregar_dados(args.data, args.corrigir_uci)
    X = df[ATRIBUTOS].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy()
    estat = comparar_estatisticas_uci(df)
    for f in (plotar_scatter_matrix(df, outdir), plotar_boxplots(df, outdir), plotar_correlacao(df, outdir)):
        plt.close(f)

    # Verificacao do gradiente
    gc = {a: gradient_check(a, seed=args.seed) for a in ("sigmoid", "tanh", "relu")}
    LOGGER.info("Gradient check (erro relativo maximo): %s", {k: f"{v:.2e}" for k, v in gc.items()})

    # Split, pre-processamento e treinamento
    idx_tr, idx_va, idx_te = split_estratificado(y, seed=args.seed)
    d = preprocessar(X, y, idx_tr, idx_va, idx_te)
    rede = MLP(4, config["hidden"], 3, config["activation"], args.seed)
    LOGGER.info("Arquitetura %s (%d parametros) | treino/val/teste = %d/%d/%d", rede.dims,
                rede.n_parametros(), len(idx_tr), len(idx_va), len(idx_te))
    hist = treinar(rede, d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], config["lr"], config["momentum"],
                   config["epochs"], config["batch_size"], config["patience"], args.seed, args.log_every)
    plt.close(plotar_curvas(hist, outdir))

    # Avaliacao
    res_va = avaliar(rede, d["Xva"], d["yva"])
    res_te = avaliar(rede, d["Xte"], d["yte"])
    plt.close(plotar_matriz_confusao(res_te["matriz_confusao"], outdir))
    LOGGER.info("TESTE: acuracia=%.4f | F1 macro=%.4f | CE=%.4f", res_te["acuracia"], res_te["macro"]["f1"],
                res_te["loss"])
    for c, m in res_te["por_classe"].items():
        LOGGER.info("  %-16s P=%.3f R=%.3f F1=%.3f (n=%d)", c, m["precisao"], m["recall"], m["f1"], m["suporte"])
    LOGGER.info("Matriz de confusao (teste):\n%s", np.array(res_te["matriz_confusao"]))

    # k-fold, experimentos e baselines
    kf = executar_kfold(X, y, config, args.kfold, args.seed) if args.kfold > 1 else None
    if kf:
        LOGGER.info("k-fold (k=%d): acuracia %.4f +- %.4f", args.kfold, kf["acc_media"], kf["acc_dp"])
        LOGGER.info("Matriz de confusao agregada (k-fold):\n%s", np.array(kf["matriz_confusao_total"]))
        plt.close(plotar_matriz_confusao(kf["matriz_confusao_total"], outdir, "matriz_confusao_kfold"))
    exps = None
    if not args.sem_experimentos and args.kfold > 1:
        exps = executar_experimentos(X, y, config, args.kfold, args.seed)
        plt.close(plotar_experimentos(exps, outdir))
    base = baselines(X, y, idx_tr, idx_te, max(args.kfold, 2), args.seed)

    # Fronteira de decisao 2-D
    rede2, sc2, res2 = treinar_fronteira(X, y, idx_tr, idx_va, idx_te, config, args.seed)
    plt.close(plotar_fronteira(rede2, sc2, X, y, idx_te, outdir))
    LOGGER.info("MLP 2-D (petalas): acuracia de teste %.4f", res2["acuracia"])

    resultados = montar_resultados(config, args.seed, args.corrigir_uci,
                                   {"treino": len(idx_tr), "validacao": len(idx_va), "teste": len(idx_te)},
                                   rede.n_parametros(), gc, hist, res_te, res_va, kf, exps, base, res2, estat)
    Path(args.json).write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Resultados salvos em %s e figuras em %s", args.json, outdir)
    return resultados


if __name__ == "__main__":
    main()
