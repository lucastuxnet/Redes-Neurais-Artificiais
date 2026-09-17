# Trabalho 05 — Classificação de Padrões com a Rede Neural Adaline

**EL056 – Redes Neurais Artificiais · PPGEELT/UFU**

---

## Conteúdo

```
trabalho05_adaline/
├── adaline_b2.py                  programa de treinamento (gera figuras e resultados)
├── requirements.txt               dependências Python
├── Trabalho05_Adaline_B2.ipynb    notebook de apresentação (já executado)
├── Basedados_B2.xlsx              base de dados
├── figs/                          8 figuras geradas pelo programa
├── resultados/
│   ├── resultados.json            todos os números do experimento
│   └── saidas_treinamento.csv     u, y e acerto para cada um dos 20 padrões
└── relatorio/
    ├── trabalho05_adaline.tex     relatório (6 páginas + capa, folha de rosto e apêndice)
    ├── adaline_b2.py              cópia do programa, incluída como apêndice via \lstinputlisting
    ├── abnt_capa.tex              bloco ABNT de capa/folha de rosto
    ├── figs/                      cópia das figuras usadas no relatório
    └── trabalho05_adaline.pdf     relatório compilado
```

## Como executar

**Dependências:** Python 3.9+ e as bibliotecas listadas em `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# execução completa (treinamento + figuras + os três protocolos de teste)
python adaline_b2.py

# variando hiperparâmetros
python adaline_b2.py --eta 0.01 --eps 1e-7 --seed 7
```

O programa imprime um relatório no terminal, salva as figuras em `figs/` e os números
em `resultados/`.

**Notebook:**

```bash
jupyter lab Trabalho05_Adaline_B2.ipynb
```

O notebook importa `adaline_b2.py`, então mantenha os dois arquivos na mesma pasta
(junto com `Basedados_B2.xlsx`). Ele já vem com todas as saídas e gráficos salvos —
basta abrir para apresentar, sem precisar reexecutar.

**Relatório:**

```bash
cd relatorio
pdflatex trabalho05_adaline.tex
pdflatex trabalho05_adaline.tex      # segunda passada resolve as referências cruzadas
```

> **Logos da capa:** o template procura `relatorio/figs/logo-ppgeel.png` e
> `relatorio/figs/logo-ufu.png`. Eles não estão incluídos aqui — o arquivo compila
> normalmente sem eles (há um espaçador de fallback), mas basta copiar os logos do
> Trabalho 03 para essa pasta para a capa ficar completa.

O preâmbulo detecta automaticamente se o `babel` em português está instalado; se não
estiver, traduz os rótulos essenciais (Figura, Tabela, Referências) e compila do mesmo
jeito. Funciona tanto localmente quanto no Overleaf.

## Resumo dos resultados

| Métrica | Valor |
|---|---|
| Épocas até a convergência (η = 0,0025, ε = 10⁻⁶) | 704 |
| Pesos finais | θ = −2,0418 · w₁ = −0,7492 · w₂ = −0,6202 |
| EQT inicial → final | 28,2577 → 3,7098 |
| Acurácia sobre a base de treinamento | 100 % |
| *Holdout* estratificado 70/30 (10 rodadas) | 98,3 % ± 5,3 % |
| Validação cruzada *leave-one-out* (20 partições) | 100 % |
| Margem mínima: Adaline × Perceptron | 0,1703 × 0,0018 |

**Fronteira de decisão obtida:**  −0,7492·s₁ − 0,6202·s₂ + 2,0418 = 0

### Pontos de discussão para a apresentação

- O EQT estabiliza em ≈ 3,71 e **não** cai a zero: esse é o resíduo estrutural entre
  alvos bipolares e a saída linear `u`, e não erro de classificação. Nesse ponto a rede
  já acerta os 20 padrões.
- Com η ≥ 0,05 o critério `|ΔEQM| ≤ ε` dispara por *oscilação*, não por convergência —
  o treinamento para cedo e a acurácia cai para 95 %.
- 20 inicializações aleatórias distintas convergem para os mesmos pesos (desvio ~10⁻⁵),
  confirmando empiricamente que a superfície de erro é convexa e o mínimo é único.
- O Perceptron converge em 23 épocas (contra 704), mas para na *primeira* fronteira
  válida: ela passa a 0,0018 do padrão mais próximo, contra 0,1703 da Adaline.
