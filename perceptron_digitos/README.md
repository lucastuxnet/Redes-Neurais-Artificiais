# Trabalho 04 — Reconhecimento de 10 Dígitos com o Perceptron

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

### Como ler o gráfico de barras

A **linha do zero** é o que importa. Um neurônio só reconhece de fato quando o
`y_in` dele fica **acima** dela. Três situações aparecem:

- **uma barra acima de zero** → resposta limpa, "a rede diz: X" em verde;
- **nenhuma barra acima de zero** → a rede *não reconheceu nada*. A demo diz
  isso explicitamente, em laranja, em vez de inventar uma resposta;
- **várias barras acima de zero** → resposta ambígua, também em laranja.

Isso resolve a maior fonte de confusão: um rabisco qualquer produz dez valores
negativos, e o "maior" deles é só o **menos ruim** — não é reconhecimento.

### O que fazer na aula

1. Rabisque qualquer coisa. A rede vai dizer *"nenhum neurônio reconheceu"*.
2. Aperte a tecla do dígito que você quis desenhar. A rede aprende ali na
   hora e mostra quantas correções de peso foram necessárias.
3. Desenhe um dígito de verdade e **desloque-o um pixel para o lado**. A
   resposta muda bastante — o perceptron **não tem invariância a translação**,
   que é o problema que a convolução resolve.
4. Ensine um padrão conflitante de propósito (desenhe o `1` e diga que é `7`).
   A rede aprende, mas o rodapé avisa: *"agora só acerta 9/10 dos dígitos
   originais"*. Aprender um padrão novo mexeu nos pesos dos outros. Aperte
   `r` para restaurar.

---

## Relatório

```bash
python3 figuras.py          # gera as figuras primeiro
cd latex
pdflatex relatorio.tex
pdflatex relatorio.tex      # 2a passada, para as referências
```

**21 páginas**, coluna única, 6 referências, no padrão ABNT NBR 14724:

| páginas | conteúdo |
|---|---|
| 1 | capa (logos UFU + PPGEELT) |
| 2 | folha de rosto |
| 3–8 | texto do trabalho |
| 10–14 | Apêndice A — código de `perceptron.py` |
| 15–21 | Apêndice B — código de `demo.py` |

Usa `thebibliography` embutido, então duas passadas de `pdflatex` bastam —
não precisa de BibTeX.

### Arquivos do LaTeX

| arquivo | o que é |
|---|---|
| `relatorio.tex` | o documento |
| `abnt_capa.tex` | capa e folha de rosto ABNT. **É aqui que se editam autor, título, matrícula e professor** |
| `logos/` | `logo-ufu.png` e `logo-ppgeel.png` |

Os apêndices usam `\lstinputlisting` apontando para `../perceptron.py` e
`../demo.py`. Ou seja, **o código impresso no PDF nunca fica desatualizado**:
mudou o `.py`, é só recompilar.

> A distribuição TeX Live desta máquina não tem o `babel-portuguese`, então
> os nomes (`Figura`, `Tabela`, `Referências`) estão redefinidos à mão no
> preâmbulo. Com o pacote instalado, dá para trocar por
> `\usepackage[brazil]{babel}`.

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

---

## Três pontos para a apresentação

1. **Por que bipolar e não binário?** Se o pixel apagado valesse 0, ele nunca
   entraria na correção dos pesos — a ausência de tinta deixaria de ser
   informação. Com −1, os dois estados pesam igual.

2. **Por que o treino continua depois de já acertar tudo?** Na época 2 a rede
   já acerta 10/10, mas só para na 4. Acerto é o neurônio vencedor; o
   algoritmo exige que *cada* neurônio dê o valor exato. As épocas extras não
   aumentam o acerto — aumentam a folga, e é a folga que vira resistência a
   ruído.

3. **Por que aguenta tanto ruído?** Os 100 pixels entram numa soma, então as
   inversões aleatórias se cancelam. A rede não depende de nenhum pixel
   isolado.

4. **"A rede diz X" nem sempre significa que ela reconheceu.** A decisão é o
   maior `y_in` entre os dez. Se todos forem negativos, nenhum neurônio
   disparou e o "vencedor" é só o menos ruim. A demo 3 mostra essa diferença
   explicitamente — é o que separa uma resposta de um chute.
