# Proposta — Trabalho 08: Reconhecimento de Flores Iris com Perceptron Multicamadas

**Aluno:** Lucas Albino Martins (12622EEL006)
**Disciplina:** EL056 – Redes Neurais Artificiais — PPGEELT/UFU
**Professor:** Prof. Dr. Keiji Yamanaka
**Local/Ano:** Uberlândia, 2026

---

## 1. Introdução e motivação

O conjunto de dados *Iris* foi publicado por R. A. Fisher em 1936, no artigo *The use of multiple measurements in taxonomic problems*, para ilustrar a análise discriminante linear. As medidas foram coletadas pelo botânico Edgar Anderson e descrevem 150 flores de três espécies do gênero *Iris* — *I. setosa*, *I. versicolor* e *I. virginica* —, com 50 exemplares cada, a partir de quatro atributos morfológicos em centímetros: comprimento e largura da sépala e da pétala. A versão usada aqui foi doada ao UCI Machine Learning Repository por Michael Marshall em 1988.

O Iris é um dos *benchmarks* mais citados em reconhecimento de padrões (Duda & Hart, 1973; Dasarathy, 1980; Gates, 1972) por reunir, em escala pequena, os dois cenários típicos de classificação: uma classe (*Setosa*) linearmente separável das demais e um par (*Versicolor* × *Virginica*) que **não** é linearmente separável. Com isso, o conjunto permite verificar se um classificador não linear contribui de fato onde o modelo linear falha. Além disso, como é pequeno, viabiliza a implementação e a verificação manual de todos os componentes de uma rede neural, como gradientes, estabilidade numérica e protocolo experimental, sem depender de frameworks.

Neste trabalho, uma MLP com saída **Softmax** e custo de **entropia cruzada** é implementada do zero em NumPy e avaliada com um protocolo experimental reprodutível, compatível com o nível de pós-graduação.

## 2. Objetivos

**Objetivo geral.** Projetar, implementar do zero e avaliar uma rede Perceptron Multicamadas para classificar as três espécies do conjunto Iris a partir dos quatro atributos morfológicos.

**Objetivos específicos** (alinhados aos requisitos do enunciado):

1. **(Req. 1)** Implementar e treinar uma MLP (forward, retropropagação e gradiente descendente em mini-batch com momentum) em NumPy, sem frameworks de *deep learning*.
2. **(Req. 2)** Tratar o problema como classificação em três classes balanceadas (50 instâncias cada), com divisão estratificada e validação cruzada k-fold estratificada.
3. **(Req. 3)** Implementar a função Softmax na camada de saída na forma numericamente estável (subtração do máximo).
4. **(Req. 4)** Implementar a entropia cruzada categorial, derivar analiticamente o gradiente conjunto Softmax + CE ($\partial L/\partial\mathbf z=\hat{\mathbf y}-\mathbf y$) e validá-lo por *gradient checking*.
5. **(Req. 5)** Usar os quatro atributos em cm como entrada, com análise exploratória da relevância de cada um.
6. Comparar arquiteturas, taxas de aprendizado e funções de ativação, e contrastar a MLP com baselines clássicos (regressão logística multinomial e k-NN).

## 3. Descrição e análise exploratória dos dados

**Atributos** (todos contínuos, em cm): *sepal length*, *sepal width*, *petal length*, *petal width*. **Rótulo:** `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` (33,3% cada). Não há valores ausentes.

**Formato do arquivo fornecido.** `iris_data.xlsx` tem 150 × 5 células e não tem cabeçalho. Os atributos estão armazenados como **texto** (serão convertidos explicitamente para `float64`) e as linhas estão ordenadas por classe em blocos de 50, o que torna obrigatório embaralhar os dados antes de dividi-los.

**Estatísticas documentadas** (`iris_names.txt`):

| Atributo | Mín. | Máx. | Média | DP | Correlação c/ classe |
|---|---|---|---|---|---|
| sepal length | 4,3 | 7,9 | 5,84 | 0,83 | 0,7826 |
| sepal width | 2,0 | 4,4 | 3,05 | 0,43 | −0,4194 |
| petal length | 1,0 | 6,9 | 3,76 | 1,76 | **0,9490** |
| petal width | 0,1 | 2,5 | 1,20 | 0,76 | **0,9565** |

**Separabilidade e atributos discriminantes.** As medidas de pétala têm correlação com a classe ≈ 0,95, o que as torna os atributos mais discriminantes; *sepal width* é o menos informativo e tem correlação negativa. No plano das pétalas, a *Setosa* forma um aglomerado isolado. Já *Versicolor* e *Virginica* se sobrepõem em uma faixa estreita (pétalas com cerca de 4,5–5,1 cm de comprimento e 1,5–1,8 cm de largura), região onde se espera que se concentrem os erros. A análise exploratória vai incluir matriz de dispersão, boxplots por classe, mapa de correlação e a conferência das estatísticas calculadas com a tabela acima.

**Amostras 35 e 38 — decisão: corrigir.** Segundo a documentação, a versão UCI difere do artigo de Fisher em duas linhas da *Setosa*. Na planilha, as duas aparecem como `4.9, 3.1, 1.5, 0.1`; os valores corretos são `4.9, 3.1, 1.5, 0.2` (amostra 35) e `4.9, 3.6, 1.4, 0.1` (amostra 38). A correção se justifica por três motivos:

- os valores verdadeiros são conhecidos e documentados pela própria fonte, então não se trata de imputação;
- na planilha as duas linhas são idênticas, uma duplicata espúria que daria peso dobrado a um único ponto;
- a correção restaura a fidelidade ao dado original.

O efeito esperado sobre o desempenho é desprezível, pois ambas são *Setosa*, classe separável. Ainda assim, a decisão fica parametrizada (`CORRIGIR_UCI` / `--corrigir-uci`) para permitir reproduzir a versão sem correção.

## 4. Fundamentação teórica

**4.1 Propagação direta.** Para uma entrada $\mathbf x\in\mathbb R^4$ e $L$ camadas:
$$\mathbf z^{(l)}=\mathbf W^{(l)\top}\mathbf a^{(l-1)}+\mathbf b^{(l)},\qquad \mathbf a^{(l)}=f\!\left(\mathbf z^{(l)}\right),\quad l=1,\dots,L-1,\qquad \mathbf a^{(0)}=\mathbf x,$$
$$\hat{\mathbf y}=\mathrm{softmax}\!\left(\mathbf z^{(L)}\right).$$

**4.2 Softmax estável.**
$$\hat y_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}=\frac{e^{z_k-m}}{\sum_{j=1}^{K}e^{z_j-m}},\qquad m=\max_j z_j .$$
A invariância a translação permite subtrair $m$ sem alterar o resultado. Com isso, $e^{z_k-m}\in(0,1]$ e o cálculo fica livre de *overflow*.

**4.3 Entropia cruzada categorial (one-hot).**
$$L=-\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K}y_{nk}\ln\hat y_{nk},$$
com $\hat y$ limitado a $[\varepsilon,1]$ ($\varepsilon=10^{-12}$). Minimizar $L$ equivale a maximizar a verossimilhança de um modelo multinomial (Goodfellow et al., 2016).

**4.4 Gradiente Softmax + CE.** O jacobiano da Softmax é $\partial\hat y_k/\partial z_j=\hat y_k(\delta_{kj}-\hat y_j)$. Assim,
$$\frac{\partial L_n}{\partial z_j}=-\sum_k\frac{y_k}{\hat y_k}\,\hat y_k(\delta_{kj}-\hat y_j)=-y_j+\hat y_j\underbrace{\textstyle\sum_k y_k}_{=1}=\hat y_j-y_j\;\Rightarrow\;\boldsymbol\delta^{(L)}=\hat{\mathbf y}-\mathbf y .$$

**4.5 Retropropagação e atualização.** Para um mini-batch $\mathcal B$ de tamanho $N_b$:
$$\boldsymbol\delta^{(l)}=\left(\mathbf W^{(l+1)}\boldsymbol\delta^{(l+1)}\right)\odot f'\!\left(\mathbf z^{(l)}\right),\qquad
\nabla_{\mathbf W^{(l)}}L=\frac{1}{N_b}\sum_{n\in\mathcal B}\mathbf a^{(l-1)}_n\boldsymbol\delta^{(l)\top}_n,\qquad
\nabla_{\mathbf b^{(l)}}L=\frac{1}{N_b}\sum_{n\in\mathcal B}\boldsymbol\delta^{(l)}_n .$$
Atualização por gradiente descendente com momentum (Haykin, 2001):
$$\mathbf v\leftarrow\mu\mathbf v-\eta\nabla_\theta L,\qquad\theta\leftarrow\theta+\mathbf v .$$
Com $\mu=0$, recupera-se o gradiente descendente em mini-batch puro.

## 5. Metodologia

- **Pré-processamento.** Conversão texto → float, correção das amostras 35/38 e codificação one-hot. Padronização z-score $x'=(x-\mu)/\sigma$ com $\mu,\sigma$ estimados **somente no treino** e aplicados à validação e ao teste, para evitar vazamento de informação. A padronização é necessária porque os atributos têm escalas diferentes e porque mantém `tanh`/`sigmoid` fora da saturação no início do treino.
- **Divisão.** Holdout estratificado 70/15/15 com semente fixa (`SEED = 42`): 35/8/7 amostras por classe, ou 105/24/21 no total. A validação é usada apenas para o early stopping. Como o teste tem poucas amostras, a estimativa de desempenho principal será a validação cruzada **k-fold estratificada (k = 5)**, com média ± desvio-padrão. Em cada fold, 15% do treino é separado para o early stopping.
- **Arquitetura.** 4 → 8 (`tanh`) → 3 (Softmax), com 67 parâmetros.
  - Uma camada oculta basta para fronteiras não lineares, pelo teorema da aproximação universal.
  - Oito neurônios dão folga sobre o mínimo necessário (a fronteira Versicolor/Virginica é quase linear) sem super-parametrizar as 105 amostras de treino; a hipótese será testada variando-se o número de neurônios.
  - A `tanh` é centrada em zero, o que produz gradientes mais bem condicionados que os da sigmoide e combina com a inicialização Xavier.
- **Inicialização.** Xavier/Glorot $\mathcal N(0,2/(n_{in}+n_{out}))$ para `tanh`/`sigmoid` e para a camada Softmax; He $\mathcal N(0,2/n_{in})$ para ReLU. Vieses começam em zero.
- **Hiperparâmetros.** η = 0,05; μ = 0,9; mini-batch de 16; até 2000 épocas; early stopping com paciência de 100 épocas sobre a perda de validação, restaurando os pesos da melhor época.
- **Verificação.** *Gradient checking* por diferenças centrais ($\epsilon=10^{-5}$), com critério de erro relativo $<10^{-6}$.
- **Reprodutibilidade.** Cada etapa estocástica usa um gerador próprio derivado da semente. O notebook e o script compartilham o mesmo código e devem produzir resultados idênticos.

## 6. Experimentos

Varia-se um fator por vez em torno da configuração base, com avaliação por k-fold (k = 5):

| Fator | Valores |
|---|---|
| Neurônios ocultos | 2, 4, 8, 16, 32 e duas camadas (10-6) |
| Taxa de aprendizado η | 0,005; 0,01; 0,05; 0,1; 0,5 |
| Ativação oculta | sigmoid, tanh, ReLU |

**Baselines** (scikit-learn, só para comparação, nos mesmos splits e folds): regressão logística multinomial, equivalente a uma "MLP sem camada oculta" e, portanto, uma medida direta do ganho trazido pela não linearidade; e k-NN com k = 5, referência clássica do Iris (Dasarathy, 1980; Gates, 1972). Também será treinada uma MLP auxiliar só com os atributos de pétala, para visualizar a fronteira de decisão em 2-D.

## 7. Métricas e análises

- Curvas de perda (entropia cruzada) e de acurácia de treino e validação por época, com a melhor época marcada.
- Matriz de confusão no teste e matriz agregada das predições fora-do-fold do k-fold, cobrindo as 150 amostras.
- Precisão, recall e F1 por classe e em média macro; acurácia global.
- Análise específica da confusão *Versicolor* × *Virginica*: onde os erros se localizam no espaço das pétalas e se a MLP reduz esses erros em relação ao modelo linear.

## 8. Cronograma

| Etapa | Atividades | S1 | S2 | S3 | S4 |
|---|---|:-:|:-:|:-:|:-:|
| 1 | Estudo teórico, análise exploratória, decisão sobre amostras 35/38 | ● | | | |
| 2 | Implementação NumPy (Softmax, CE, backprop) e *gradient checking* | ● | ● | | |
| 3 | Treinamento, early stopping e avaliação holdout | | ● | | |
| 4 | Experimentos (k-fold, hiperparâmetros, baselines, fronteira) | | ● | ● | |
| 5 | Relatório LaTeX, revisão e entrega | | | ● | ● |

(S = semana, contada a partir da aprovação da proposta.)

## Referências

DASARATHY, B. V. Nosing around the neighborhood: a new system structure and classification rule for recognition in partially exposed environments. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, v. PAMI-2, n. 1, p. 67-71, 1980.

DUDA, R. O.; HART, P. E. **Pattern classification and scene analysis**. New York: John Wiley & Sons, 1973.

FISHER, R. A. The use of multiple measurements in taxonomic problems. **Annals of Eugenics**, v. 7, n. 2, p. 179-188, 1936.

FISHER, R. A. **Iris**. UCI Machine Learning Repository, 1988. Disponível em: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/. Acesso em: 7 out. 2026.

GATES, G. W. The reduced nearest neighbor rule. **IEEE Transactions on Information Theory**, v. 18, n. 3, p. 431-433, 1972.

GLOROT, X.; BENGIO, Y. Understanding the difficulty of training deep feedforward neural networks. In: INTERNATIONAL CONFERENCE ON ARTIFICIAL INTELLIGENCE AND STATISTICS, 13., 2010, Sardinia. **Proceedings** [...]. PMLR, 2010. p. 249-256.

GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep learning**. Cambridge, MA: MIT Press, 2016.

HAYKIN, S. **Redes neurais**: princípios e prática. 2. ed. Porto Alegre: Bookman, 2001.

HE, K. et al. Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2015, Santiago. **Proceedings** [...]. IEEE, 2015. p. 1026-1034.
