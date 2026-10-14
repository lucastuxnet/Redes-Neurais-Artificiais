#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Converte ``resultados.json`` em ``latex/resultados_macros.tex``.

O relatorio ``latex/relatorio_trabalho08.tex`` define cada resultado como um
marcador (ex.: ``\\textbf{XX,X\\%}``) e, se este arquivo existir, os
marcadores sao substituidos (``\\renewcommand``) pelos valores reais.

Uso::

    python iris_mlp.py                 # gera resultados.json e figs/
    python gerar_macros_latex.py       # gera latex/resultados_macros.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

CLS = {"Iris-setosa": "Set", "Iris-versicolor": "Ver", "Iris-virginica": "Vir"}
LETRA = "ABC"  # A = setosa, B = versicolor, C = virginica (linhas = real, colunas = prevista)


def pct(x: float, casas: int = 2) -> str:
    """0.97333 -> '97,33\\%' (virgula decimal, padrao ABNT)."""
    return f"{100 * x:.{casas}f}".replace(".", ",") + r"\%"


def num(x: float, casas: int = 4) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return rf"${m.replace('.', '{,}')}\times10^{{{int(e)}}}$"


def macros(r: Dict) -> List[str]:
    L: List[str] = []

    def M(nome: str, valor: str) -> None:
        L.append(rf"\renewcommand{{\{nome}}}{{{valor}}}")

    te, tr, kf = r["teste"], r["treinamento"], r["kfold"]
    M("AccTeste", pct(te["acuracia"]))
    M("FUmTeste", pct(te["macro"]["f1"]))
    M("CETeste", num(te["loss"]))
    M("AccVal", pct(r["validacao"]["acuracia"]))
    M("AccTreino", pct(tr["acc_tr_final"]))
    M("CETreino", num(tr["loss_tr_final"]))
    M("CEVal", num(tr["loss_va_final"]))
    M("MelhorEpoca", str(tr["best_epoch"]))
    M("EpocasExec", str(tr["epocas_executadas"]))
    M("NParam", str(r["hiperparametros"]["n_parametros"]))
    gc = r["gradient_check"]
    M("GCSig", sci(gc["sigmoid"]))
    M("GCTanh", sci(gc["tanh"]))
    M("GCRelu", sci(gc["relu"]))
    for c, s in CLS.items():
        m = te["por_classe"][c]
        M(f"P{s}", num(m["precisao"], 3))
        M(f"R{s}", num(m["recall"], 3))
        M(f"F{s}", num(m["f1"], 3))
    M("PMacro", num(te["macro"]["precisao"], 3))
    M("RMacro", num(te["macro"]["recall"], 3))
    M("FMacro", num(te["macro"]["f1"], 3))
    for i in range(3):
        for j in range(3):
            M(f"CMT{LETRA[i]}{LETRA[j]}", str(te["matriz_confusao"][i][j]))
    if kf:
        M("AccKfold", pct(kf["acc_media"]))
        M("DPKfold", pct(kf["acc_dp"]))
        M("FUmKfold", pct(kf["f1_media"]))
        M("DPFUmKfold", pct(kf["f1_dp"]))
        M("FoldsKfold", "; ".join(pct(a) for a in kf["acc_folds"]))
        for i in range(3):
            for j in range(3):
                M(f"CMK{LETRA[i]}{LETRA[j]}", str(kf["matriz_confusao_total"][i][j]))
        cm = kf["matriz_confusao_total"]
        M("ErrosVerVir", str(cm[1][2] + cm[2][1]))
        M("ErrosTotalKfold", str(sum(cm[i][j] for i in range(3) for j in range(3) if i != j)))
    if r.get("experimentos"):
        nomes = {"hidden": "Neurônios ocultos", "lr": r"Taxa $\eta$", "activation": "Ativação"}
        linhas, ultimo = [], None
        for e in r["experimentos"]:
            if ultimo is not None and e["fator"] != ultimo:
                linhas.append(r"\midrule")
            ultimo = e["fator"]
            v = e["valor"]
            v = "-".join(map(str, v)) if isinstance(v, list) else (num(v, 3).rstrip("0").rstrip(",")
                                                                   if isinstance(v, float) else str(v))
            linhas.append(rf"{nomes[e['fator']]} & {v} & {pct(e['acc_media'])} $\pm$ {pct(e['acc_dp'])} & "
                          rf"{pct(e['f1_media'])} & {e['best_epoch_medio']:.0f} \\")
        M("TabelaExperimentos", "\n".join(linhas))
    b = r["baselines"]
    for nome, s in (("Regressao logistica multinomial", "LR"), ("k-NN (k=5)", "KNN")):
        M(f"AccTeste{s}", pct(b[nome]["acc_teste"]))
        M(f"AccKfold{s}", pct(b[nome]["acc_kfold_media"]))
        M(f"DPKfold{s}", pct(b[nome]["acc_kfold_dp"]))
    M("AccFronteira", pct(r["fronteira_2d"]["acuracia_teste"]))
    est = r["estatisticas"]
    rot = {"sepal_length": "sepal length", "sepal_width": "sepal width",
           "petal_length": "petal length", "petal_width": "petal width"}
    linhas = [rf"{rot[a]} & {num(v['min'], 1)} & {num(v['max'], 1)} & {num(v['media'], 3)} & {num(v['dp'], 3)} & "
              rf"{num(v['corr_classe'], 4)} & {num(v['uci']['corr_classe'], 4)} \\" for a, v in est.items()]
    M("TabelaEstatisticas", "\n".join(linhas))
    return L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", default="resultados.json")
    p.add_argument("--saida", default="latex/resultados_macros.tex")
    a = p.parse_args()
    r = json.loads(Path(a.json).read_text(encoding="utf-8"))
    corpo = "\n".join(macros(r))
    Path(a.saida).write_text("% Gerado automaticamente por gerar_macros_latex.py -- nao editar.\n" + corpo + "\n",
                             encoding="utf-8")
    print(f"{a.saida}: {corpo.count(chr(92) + 'renewcommand')} macros")


if __name__ == "__main__":
    main()
