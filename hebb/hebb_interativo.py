"""
Regra de Hebb - programa interativo
------------------------------------
Escolha uma das 16 funcoes logicas de 2 variaveis em um menu.
O programa mostra a tabela-verdade, o calculo passo a passo dos pesos
pela regra de Hebb (representacao bipolar) e o teste da rede nos 4
padroes de entrada. Em seguida, gera e salva o grafico da fronteira
de decisao (ou avisa que a funcao nao e linearmente separavel).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. As 16 funcoes logicas de 2 variaveis (alvo em bipolar)
# ---------------------------------------------------------------------------
# Ordem fixa dos padroes de entrada: (-1,-1), (-1,1), (1,-1), (1,1)
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)

FUNCOES = {
    "1":  ("FALSO (0)",           [-1, -1, -1, -1]),
    "2":  ("AND",                 [-1, -1, -1,  1]),
    "3":  ("x1 AND NOT x2",       [-1, -1,  1, -1]),
    "4":  ("x1",                  [-1, -1,  1,  1]),
    "5":  ("NOT x1 AND x2",       [-1,  1, -1, -1]),
    "6":  ("x2",                  [-1,  1, -1,  1]),
    "7":  ("XOR",                 [-1,  1,  1, -1]),
    "8":  ("OR",                  [-1,  1,  1,  1]),
    "9":  ("NOR",                 [ 1, -1, -1, -1]),
    "10": ("XNOR (equivalencia)", [ 1, -1, -1,  1]),
    "11": ("NOT x2",              [ 1, -1,  1, -1]),
    "12": ("x1 OR NOT x2",        [ 1, -1,  1,  1]),
    "13": ("NOT x1",              [ 1,  1, -1, -1]),
    "14": ("NOT x1 OR x2",        [ 1,  1, -1,  1]),
    "15": ("NAND",                [ 1,  1,  1, -1]),
    "16": ("VERDADEIRO (1)",      [ 1,  1,  1,  1]),
}

PASTA_SAIDA = "figs"  # onde os graficos serao salvos


# ---------------------------------------------------------------------------
# 2. Regra de Hebb
# ---------------------------------------------------------------------------
def regra_hebb(X, t):
    """Calcula w1, w2, b pela regra de Hebb (passagem unica)."""
    w = np.zeros(2)
    b = 0.0
    for x, ti in zip(X, t):
        w += x * ti
        b += ti
    return w[0], w[1], b


def sinal_bipolar(v):
    return 1.0 if v >= 0 else -1.0


def testa_rede(w1, w2, b, X, t):
    """Retorna (acertos, saida_predita, sucesso_total)."""
    saida = []
    acertos = 0
    for x, ti in zip(X, t):
        net = w1 * x[0] + w2 * x[1] + b
        y = sinal_bipolar(net)
        saida.append(y)
        if y == ti:
            acertos += 1
    return acertos, saida, acertos == len(t)


# ---------------------------------------------------------------------------
# 3. Exibicao do menu e da tabela-verdade / calculo dos pesos
# ---------------------------------------------------------------------------
def mostra_menu():
    print("\n" + "=" * 60)
    print("REGRA DE HEBB - FUNCOES LOGICAS DE 2 VARIAVEIS (BIPOLAR)")
    print("=" * 60)
    for chave, (nome, _) in FUNCOES.items():
        print(f"  {chave:>2s}) {nome}")
    print("   0) Sair")
    print("=" * 60)


def mostra_resultado(nome, t):
    t = np.array(t, dtype=float)
    w1, w2, b = regra_hebb(X, t)
    acertos, saida, sucesso = testa_rede(w1, w2, b, X, t)

    print(f"\nFuncao escolhida: {nome}")
    print("-" * 72)
    print(f"{'(x1,x2)':>10s} {'alvo t':>8s}   |   calculo de w1, w2 e b (acumulado)")
    print("-" * 72)

    w1_acum = w2_acum = b_acum = 0.0
    for (x1, x2), ti in zip(X, t):
        w1_acum += x1 * ti
        w2_acum += x2 * ti
        b_acum += ti
        print(f"({x1:+.0f},{x2:+.0f})   {ti:+.0f}       ->  "
              f"w1={w1_acum:+.0f}  w2={w2_acum:+.0f}  b={b_acum:+.0f}")

    print("-" * 72)
    print(f"Pesos finais (regra de Hebb): w1={w1:+.0f}  w2={w2:+.0f}  b={b:+.0f}")
    print(f"Neuronio treinado: y = sinal({w1:+.0f}*x1 {w2:+.0f}*x2 {b:+.0f})")
    print("-" * 72)

    print(f"{'(x1,x2)':>10s} {'net':>8s} {'saida':>7s} {'alvo':>6s} {'OK?':>5s}")
    for (x1, x2), ti, y in zip(X, t, saida):
        net = w1 * x1 + w2 * x2 + b
        ok = "SIM" if y == ti else "NAO"
        print(f"({x1:+.0f},{x2:+.0f})   {net:+6.1f}  {y:+6.0f}  {ti:+5.0f}  {ok:>5s}")

    print("-" * 72)
    if sucesso:
        print(f"RESULTADO: a regra de Hebb ENCONTROU os pesos corretos "
              f"({acertos}/4 padroes) -- funcao linearmente separavel.")
    else:
        print(f"RESULTADO: a regra de Hebb NAO conseguiu aprender esta funcao "
              f"({acertos}/4 padroes corretos) -- funcao NAO linearmente separavel.")
    print("=" * 72)

    return w1, w2, b, t, sucesso


# ---------------------------------------------------------------------------
# 4. Geracao do grafico para a funcao escolhida
# ---------------------------------------------------------------------------
def gera_grafico(nome, w1, w2, b, t, sucesso):
    os.makedirs(PASTA_SAIDA, exist_ok=True)  # evita o FileNotFoundError

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    cores = ["#d62728" if v == -1 else "#1f77b4" for v in t]
    ax.scatter(X[:, 0], X[:, 1], c=cores, s=220, edgecolor="k", zorder=3)

    for (x1, x2), v in zip(X, t):
        ax.annotate(f"({x1:+.0f},{x2:+.0f})  t={int(v):+d}", (x1, x2),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9)

    xx = np.linspace(-1.8, 1.8, 50)
    if sucesso:
        if abs(w2) > 1e-9:
            yy = -(w1 * xx + b) / w2
            ax.plot(xx, yy, "k-", linewidth=2,
                    label=f"$w_1$={w1:+.0f}, $w_2$={w2:+.0f}, $b$={b:+.0f}")
        elif abs(w1) > 1e-9:
            xv = -b / w1
            ax.axvline(xv, color="k", linewidth=2,
                       label=f"$w_1$={w1:+.0f}, $w_2$={w2:+.0f}, $b$={b:+.0f}")
        ax.legend(loc="lower right", fontsize=8)
        titulo = f"{nome}: linearmente separavel\n(Hebb converge)"
        cor_titulo = "black"
    else:
        ax.text(0, 0, "NAO SEPARAVEL\n(Hebb falha)", ha="center", va="center",
                fontsize=11, color="#d62728", fontweight="bold")
        titulo = f"{nome}: NAO linearmente separavel"
        cor_titulo = "#d62728"

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(titulo, fontsize=11, color=cor_titulo)

    fig.tight_layout()

    nome_arquivo = nome.split(" ")[0].replace("(", "").replace(")", "").lower()
    caminho = os.path.join(PASTA_SAIDA, f"hebb_{nome_arquivo}.png")
    fig.savefig(caminho, dpi=170)
    plt.show()
    print(f"\nGrafico salvo em: {caminho}")


# ---------------------------------------------------------------------------
# 5. Loop principal
# ---------------------------------------------------------------------------
def main():
    while True:
        mostra_menu()
        escolha = input("Escolha uma funcao (numero): ").strip()

        if escolha == "0":
            print("Encerrando.")
            break

        if escolha not in FUNCOES:
            print("Opcao invalida, tente novamente.")
            continue

        nome, alvo = FUNCOES[escolha]
        w1, w2, b, t, sucesso = mostra_resultado(nome, alvo)
        gera_grafico(nome, w1, w2, b, t, sucesso)

        input("\nPressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()
