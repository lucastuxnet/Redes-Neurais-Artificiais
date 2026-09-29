# Trabalho 07 — Reconhecimento de 10 Dígitos com o Perceptron

Disciplina de Redes Neurais · Pós-Graduação em Engenharia Elétrica, UFU

Um perceptron de camada única (100 entradas → 10 neurônios) que reconhece os
dez dígitos desenhados em matrizes 10×10, com análise de robustez a ruído e
três demonstrações interativas para a apresentação.

Artigo de base: N. Sehta, K. Saraf e A. Vinchi, *"Evaluation of Classifiers
Based on Neural Network for Handwritten Digit Recognition"*, ACROSET/IEEE,
2024 (DOI: 10.1109/ACROSET62108.2024.10743433).

---

## Começar rápido

```bash
pip install numpy matplotlib
python3 perceptron.py
```

Isso treina a rede e imprime tudo no terminal: convergência época a época,
a saída para cada dígito e a tabela de ruído. Leva menos de 2 segundos.

---

## Os quatro arquivos

| arquivo | o que é |
|---|---|
| `perceptron.py` | **Tudo o que importa.** Os dígitos, o algoritmo e o ruído, em ~200 linhas comentadas. É o arquivo para ler e explicar. |
| `demo.py` | As três demonstrações interativas da apresentação. |
| `figuras.py` | Gera as 5 figuras do relatório. |
| `apresentacao.ipynb` | O notebook da apresentação. |

---

## As três demos

```bash
python3 demo.py       # abre o menu
python3 demo.py 1     # treino passo a passo
python3 demo.py 2     # ruido com controle deslizante
python3 demo.py 3     # desenhar o digito com o mouse
```

**1 — Treino passo a passo.** Um botão avança uma época por vez. Os dez mapas
de peso começam todos em zero e vão se formando na tela até a convergência.
Bom para mostrar que o aprendizado é literalmente acumular os padrões nos
pesos.

**2 — Ruído com controle deslizante.** Um cursor ajusta `p` de 0 a 0,5 e os
dez dígitos aparecem degradados ao vivo, com o rótulo predito em verde
(acerto) ou vermelho (erro). O botão "Outro sorteio" gera uma nova amostra no
mesmo nível de ruído.

**3 — Desenhar o dígito e ensinar a rede.** Você desenha numa grade 10×10 com
o mouse (esquerdo pinta, direito apaga) e a rede classifica a cada traço.

| tecla | o que faz |
|---|---|
| `0`–`9` | **ensina**: "o que eu desenhei é este dígito". A rede aplica a regra de aprendizado até acertar o seu desenho |
| `r` | reseta a rede (volta a treinar só nos 10 padrões originais) |
| `c` | limpa a tela |

---

## Resultados

**Treinamento** — converge em 4 épocas, 100% de acerto nos dez dígitos.

| Época | Correções | Acertos |
|---|---|---|
| 1 | 39 | 7/10 |
| 2 | 10 | 10/10 |
| 3 | 4 | 10/10 |
| 4 | 0 | 10/10 |

**Ruído** (inversão de pixels, média de 300 sorteios por ponto):

| p | pixels trocados | acurácia |
|---|---|---|
| 0,10 | 10 | 99,4% |
| 0,20 | 20 | 92,8% |
| 0,30 | 30 | 77,1% |
| 0,50 | 50 | 9,8% |

A rede mantém 90% de acerto até **p ≈ 0,22** — cerca de 22 dos 100 pixels
podem estar errados. Em p = 0,5 o desempenho cai para o do chute (10%), como
esperado: aí a imagem já não tem relação nenhuma com o dígito original.
