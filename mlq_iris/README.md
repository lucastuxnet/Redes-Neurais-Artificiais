# Trabalho 08 — Reconhecimento de Flores Iris (EL056)

Fluxo completo (na raiz `trabalho08/`):

```bash
pip install -r requirements.txt
python iris_mlp.py                  # treino + experimentos -> figs/ e resultados.json (~1 min)
jupyter nbconvert --to notebook --execute iris_mlp.ipynb --inplace   # mesmo resultado (confere com resultados.json)
python gerar_macros_latex.py        # resultados.json -> latex/resultados_macros.tex
cd latex && pdflatex relatorio_trabalho08 && pdflatex relatorio_trabalho08
```

- Sem `latex/resultados_macros.tex`, o relatório compila com os marcadores **XX,X%**.
- Coloque os logos em `figs/logo-ppgeel.png` e `figs/logo-ufu.png` (sem eles, a capa reserva o espaço em branco).
- `proposta_trabalho08.md`: proposta do trabalho (entregável A).
