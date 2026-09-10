"""
Perceptron para reconhecimento de 10 digitos em matriz 10x10.
Trabalho 04 - Disciplina de Redes Neurais

Este arquivo contem TUDO o que a rede precisa: os padroes, o algoritmo de
treinamento e o gerador de ruido. Sao pouco mais de 200 linhas, pensadas
para serem lidas de cima a baixo durante a apresentacao.

Convencao bipolar:  pixel aceso = +1  |  pixel apagado = -1
Arquitetura:        100 entradas (10x10)  ->  10 neuronios de saida
"""

import numpy as np

LADO = 10                 # a matriz e 10x10
N_ENTRADAS = LADO * LADO  # 100 pixels
N_SAIDAS = 10             # um neuronio por digito


# ===========================================================================
# 1. OS DEZ DIGITOS
# ===========================================================================
# '#' = pixel aceso (+1)   '.' = pixel apagado (-1)

DIGITOS = {
    0: ["..######..",
        ".##....##.",
        "##......##",
        "##......##",
        "##......##",
        "##......##",
        "##......##",
        "##......##",
        ".##....##.",
        "..######.."],

    1: ["....##....",
        "...###....",
        "..####....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "....##....",
        "..######..",
        "..######.."],

    2: ["..######..",
        ".##....##.",
        "##......##",
        ".......##.",
        "......##..",
        "....###...",
        "..###.....",
        ".##.......",
        "##........",
        "##########"],

    3: [".########.",
        ".......##.",
        "......##..",
        ".....##...",
        "...#####..",
        ".......##.",
        "........##",
        "##......##",
        ".##....##.",
        "..######.."],

    4: ["......##..",
        ".....###..",
        "....####..",
        "...##.##..",
        "..##..##..",
        ".##...##..",
        "##########",
        "......##..",
        "......##..",
        "......##.."],

    5: ["##########",
        "##........",
        "##........",
        "##........",
        "########..",
        ".......##.",
        "........##",
        "##......##",
        ".##....##.",
        "..######.."],

    6: ["...#####..",
        "..##......",
        ".##.......",
        "##........",
        "##.#####..",
        "###....##.",
        "##......##",
        "##......##",
        ".##....##.",
        "..######.."],

    7: ["##########",
        "##......##",
        ".......##.",
        "......##..",
        ".....##...",
        "....##....",
        "...##.....",
        "...##.....",
        "...##.....",
        "...##....."],

    8: ["..######..",
        ".##....##.",
        "##......##",
        ".##....##.",
        "..######..",
        ".##....##.",
        "##......##",
        "##......##",
        ".##....##.",
        "..######.."],

    9: ["..######..",
        ".##....##.",
        "##......##",
        "##......##",
        ".##....###",
        "..#####.##",
        "........##",
        ".......##.",
        "......##..",
        "..#####..."],
}


def carregar_digitos():
    """
    Monta o conjunto de treinamento.

    Retorna:
      X  (10, 100)  cada linha e um digito achatado em vetor bipolar
      T  (10, 10)   alvos: T[k] tem +1 na posicao k e -1 nas outras
      y  (10,)      o rotulo de cada linha: [0, 1, 2, ..., 9]
    """
    X = np.array([
        [1.0 if c == "#" else -1.0 for linha in DIGITOS[d] for c in linha]
        for d in range(10)
    ])
    T = -np.ones((10, 10))
    T[np.arange(10), np.arange(10)] = 1.0     # diagonal = +1
    return X, T, np.arange(10)


def para_imagem(vetor):
    """Transforma o vetor de 100 componentes de volta na matriz 10x10."""
    return np.asarray(vetor, dtype=float).reshape(LADO, LADO)


# ===========================================================================
# 2. O PERCEPTRON
# ===========================================================================

class Perceptron:
    """
    Perceptron de camada unica, com o algoritmo classico do Fausett.

        Passo 0.  Inicializa pesos e bias com zero.
        Passo 1.  Repete enquanto houver mudanca de peso.
        Passo 2.    Para cada par de treinamento (s, t):
        Passo 3.      x = s
        Passo 4.      y_in_j = b_j + soma_i (x_i * w_ij)
                      y_j    = +1 se y_in_j >= 0, senao -1
        Passo 5.      Se t_j != y_j:
                          b_j  += alfa * t_j
                          w_ij += alfa * t_j * x_i
        Passo 6.  Para quando nenhum peso mudou na epoca.
    """

    def __init__(self, alfa=1.0, limiar=0.0):
        self.alfa = alfa              # taxa de aprendizado
        self.limiar = limiar          # theta da funcao degrau
        self.W = np.zeros((N_ENTRADAS, N_SAIDAS))   # Passo 0
        self.b = np.zeros(N_SAIDAS)
        self.historico = []           # guarda o que aconteceu em cada epoca

    # -- Passo 4 -----------------------------------------------------------
    def entrada_liquida(self, X):
        """y_in = b + X.W  (aceita um digito ou varios de uma vez)."""
        return np.atleast_2d(X) @ self.W + self.b

    def ativacao(self, y_in):
        """Funcao degrau bipolar: +1 se y_in >= limiar, senao -1."""
        return np.where(y_in >= self.limiar, 1.0, -1.0)

    def classificar(self, X):
        """
        Qual digito a rede acha que e?

        A saida bipolar pode ser ambigua (nenhum ou varios neuronios em +1),
        entao decidimos pelo neuronio de maior y_in -- o "mais confiante".
        """
        return np.argmax(self.entrada_liquida(X), axis=1)

    # -- Passos 1 a 6 ------------------------------------------------------
    def treinar_uma_epoca(self, X, T):
        """
        Executa UMA epoca (uma passada por todos os padroes).

        Retorna quantas atualizacoes de peso foram feitas. Zero significa
        que a rede convergiu.
        """
        n_atualizacoes = 0

        for k in range(len(X)):          # Passo 2
            x = X[k]                     # Passo 3
            t = T[k]

            y_in = self.b + x @ self.W   # Passo 4
            y = self.ativacao(y_in)

            errou = y != t               # Passo 5: so corrige onde errou
            if errou.any():
                correcao = self.alfa * t * errou
                self.W += np.outer(x, correcao)
                self.b += correcao
                n_atualizacoes += int(errou.sum())

        return n_atualizacoes

    def treinar(self, X, T, max_epocas=100, mostrar=True):
        """Repete epocas ate convergir (Passo 6) ou atingir max_epocas."""
        rotulos = np.argmax(T, axis=1)

        if mostrar:
            print(f"{'epoca':>6} | {'atualizacoes':>13} | {'acertos':>8}")
            print("-" * 34)

        for epoca in range(1, max_epocas + 1):
            n = self.treinar_uma_epoca(X, T)
            acertos = int((self.classificar(X) == rotulos).sum())
            self.historico.append((epoca, n, acertos))

            if mostrar:
                print(f"{epoca:>6} | {n:>13} | {acertos:>6}/10")

            if n == 0:                                     # Passo 6
                if mostrar:
                    print("-" * 34)
                    print(f"Convergiu na epoca {epoca}: nenhum peso mudou.")
                return epoca

        if mostrar:
            print(f"Parou no limite de {max_epocas} epocas.")
        return max_epocas

    # -- utilidades --------------------------------------------------------
    def acuracia(self, X, rotulos):
        """Fracao de acertos (0 a 1)."""
        return float((self.classificar(X) == rotulos).mean())

    def mapa_de_pesos(self, j):
        """Os pesos do neuronio j vistos como imagem 10x10."""
        return self.W[:, j].reshape(LADO, LADO)


# ===========================================================================
# 3. RUIDO
# ===========================================================================

def adicionar_ruido(X, p, rng=None):
    """
    Inverte cada pixel com probabilidade p (ruido sal-e-pimenta).

    p = 0.0  -> nada muda
    p = 0.2  -> em media 20 dos 100 pixels sao invertidos
    p = 0.5  -> a imagem vira ruido puro, sem relacao com o original
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.array(X, dtype=float)          # copia, nao altera o original
    inverter = rng.random(X.shape) < p
    X[inverter] *= -1
    return X


def testar_com_ruido(rede, X, rotulos, p, repeticoes=200, rng=None):
    """
    Mede a acuracia da rede em um nivel de ruido p.

    Como o ruido e aleatorio, uma unica medida nao diz muito: repetimos
    varias vezes e tiramos a media.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    acertos = 0
    for _ in range(repeticoes):
        acertos += int((rede.classificar(adicionar_ruido(X, p, rng))
                        == rotulos).sum())
    return acertos / (repeticoes * len(X))


# ===========================================================================
# 4. EXECUCAO PRINCIPAL
# ===========================================================================

def main():
    print("=" * 60)
    print("PERCEPTRON - RECONHECIMENTO DE 10 DIGITOS (matriz 10x10)")
    print("=" * 60)

    X, T, rotulos = carregar_digitos()
    print(f"\nConjunto de treinamento: {X.shape[0]} digitos, "
          f"{X.shape[1]} pixels cada.\n")

    # ---- treinamento --------------------------------------------------
    print("TREINAMENTO")
    rede = Perceptron(alfa=1.0, limiar=0.0)
    epocas = rede.treinar(X, T)
    print(f"\nAcuracia nos digitos originais: "
          f"{rede.acuracia(X, rotulos):.0%}\n")

    # ---- o que a rede responde para cada digito -----------------------
    print("SAIDA DA REDE PARA CADA DIGITO")
    print(f"{'digito':>7} | {'predito':>8} | {'y_in do vencedor':>17}")
    print("-" * 40)
    y_in = rede.entrada_liquida(X)
    for k in rotulos:
        print(f"{k:>7} | {rede.classificar(X[k])[0]:>8} | {y_in[k].max():>17.0f}")

    # ---- analise de ruido ---------------------------------------------
    print("\nANALISE DE RUIDO")
    print("Quantos pixels podem ser invertidos antes da rede errar?\n")
    print(f"{'p':>6} | {'pixels trocados':>16} | {'acuracia':>9}")
    print("-" * 38)
    for p in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        acc = testar_com_ruido(rede, X, rotulos, p)
        print(f"{p:>6.2f} | {int(p * 100):>14} % | {acc:>8.1%}")

    print("\nObs.: em p = 0.50 a imagem perde toda relacao com o original,")
    print("      entao acertar 10% e o mesmo que chutar entre 10 digitos.")


if __name__ == "__main__":
    main()
