"""
Demonstracoes interativas para a apresentacao.
Trabalho 04 - Disciplina de Redes Neurais

Rode:  python3 demo.py            (mostra o menu)
       python3 demo.py 1          (treino passo a passo)
       python3 demo.py 2          (ruido com controle deslizante)
       python3 demo.py 3          (desenhar o digito com o mouse)

As tres demos abrem uma janela do matplotlib. Nao precisa de Jupyter.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from perceptron import (LADO, Perceptron, adicionar_ruido, carregar_digitos,
                        para_imagem)

# cores usadas nas tres demos
VERDE = "#1e8449"
VERMELHO = "#c0392b"
AZUL = "#1f4e79"
CINZA = "#7f8c8d"
LARANJA = "#b9770e"


def _desenhar_grade(ax, matriz, cmap="gray_r", vmin=-1, vmax=1):
    """Desenha uma matriz 10x10 com a gradinha por cima."""
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, LADO, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, LADO, 1), minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ===========================================================================
# DEMO 1 - TREINO PASSO A PASSO
# ===========================================================================
def demo_treino():
    """
    Mostra a rede aprendendo, uma epoca por vez.

    Cada clique em "Proxima epoca" executa uma passada completa pelos 10
    digitos. Os dez mapas de peso vao se formando na tela, e o painel de
    texto mostra quantas correcoes foram feitas.
    """
    X, T, rotulos = carregar_digitos()
    rede = Perceptron()
    estado = {"epoca": 0, "convergiu": False}

    fig = plt.figure(figsize=(13, 6.2))
    fig.canvas.manager.set_window_title("Demo 1 - Treino passo a passo")
    eixos = [fig.add_subplot(2, 5, j + 1) for j in range(10)]
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.16,
                        hspace=0.30, wspace=0.15)

    imagens = []
    for j, ax in enumerate(eixos):
        im = _desenhar_grade(ax, rede.mapa_de_pesos(j), cmap="RdBu_r",
                             vmin=-1, vmax=1)
        ax.set_title(f"neuronio {j}", fontsize=9)
        imagens.append(im)

    titulo = fig.suptitle("", fontsize=12)

    def atualizar_titulo(n_atualizacoes=None):
        if estado["epoca"] == 0:
            txt = ("Epoca 0 -- todos os pesos comecam em ZERO (Passo 0)\n"
                   "Clique em 'Proxima epoca' para a rede aprender")
        elif estado["convergiu"]:
            acc = rede.acuracia(X, rotulos)
            txt = (f"CONVERGIU na epoca {estado['epoca']} -- "
                   f"nenhum peso mudou (Passo 6)\n"
                   f"Acuracia: {acc:.0%}  |  o treinamento acabou")
        else:
            acertos = int((rede.classificar(X) == rotulos).sum())
            txt = (f"Epoca {estado['epoca']}: {n_atualizacoes} correcoes de peso"
                   f"  |  acertos: {acertos}/10\n"
                   f"vermelho = peso positivo (evidencia a favor) ; "
                   f"azul = peso negativo (contra)")
        titulo.set_text(txt)

    atualizar_titulo()

    def proxima_epoca(evento):
        if estado["convergiu"]:
            return
        n = rede.treinar_uma_epoca(X, T)
        estado["epoca"] += 1
        if n == 0:
            estado["convergiu"] = True

        # reescala as cores conforme os pesos crescem. Usamos o percentil 98
        # em vez do maximo: uns poucos pesos extremos achatariam o contraste
        # e esconderiam o desenho que se forma nos mapas.
        vmax = max(1.0, float(np.percentile(np.abs(rede.W), 98)))
        for j, im in enumerate(imagens):
            im.set_data(rede.mapa_de_pesos(j))
            im.set_clim(-vmax, vmax)
        atualizar_titulo(n)
        fig.canvas.draw_idle()

    def reiniciar(evento):
        rede.W[:] = 0.0
        rede.b[:] = 0.0
        rede.historico.clear()
        estado["epoca"] = 0
        estado["convergiu"] = False
        for j, im in enumerate(imagens):
            im.set_data(rede.mapa_de_pesos(j))
            im.set_clim(-1, 1)
        atualizar_titulo()
        fig.canvas.draw_idle()

    b1 = Button(fig.add_axes([0.34, 0.04, 0.18, 0.07]), "Proxima epoca")
    b1.on_clicked(proxima_epoca)
    b2 = Button(fig.add_axes([0.54, 0.04, 0.12, 0.07]), "Reiniciar")
    b2.on_clicked(reiniciar)

    # o matplotlib nao guarda referencia para os widgets: se elas sumirem,
    # os botoes param de responder. Penduramos na figura para mante-las vivas.
    fig._widgets = [b1, b2]

    plt.show()
    return fig


# ===========================================================================
# DEMO 2 - RUIDO COM CONTROLE DESLIZANTE
# ===========================================================================
def demo_ruido():
    """
    Aumenta o ruido ao vivo e mostra a rede acertando ate quebrar.

    O slider controla p, a probabilidade de cada pixel ser invertido.
    Os dez digitos aparecem degradados, com o rotulo predito em verde
    (acerto) ou vermelho (erro).
    """
    X, T, rotulos = carregar_digitos()
    rede = Perceptron()
    rede.treinar(X, T, mostrar=False)
    estado = {"semente": 0}

    fig = plt.figure(figsize=(13, 6.4))
    fig.canvas.manager.set_window_title("Demo 2 - Ruido")
    eixos = [fig.add_subplot(2, 5, k + 1) for k in range(10)]
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.20,
                        hspace=0.35, wspace=0.15)

    imagens = [_desenhar_grade(ax, para_imagem(X[k])) for k, ax in enumerate(eixos)]
    titulo = fig.suptitle("", fontsize=12)

    def atualizar(_=None):
        p = slider.val
        rng = np.random.default_rng(estado["semente"])
        Xr = adicionar_ruido(X, p, rng)
        preditos = rede.classificar(Xr)
        acertos = int((preditos == rotulos).sum())

        for k, (ax, im) in enumerate(zip(eixos, imagens)):
            im.set_data(para_imagem(Xr[k]))
            ok = preditos[k] == rotulos[k]
            ax.set_title(f"{rotulos[k]} -> {preditos[k]}",
                         color=VERDE if ok else VERMELHO, fontsize=11)

        titulo.set_text(
            f"p = {p:.2f}   ->   em media {int(p * 100)} dos 100 pixels "
            f"invertidos   |   acertos: {acertos}/10")
        fig.canvas.draw_idle()

    slider = Slider(fig.add_axes([0.20, 0.08, 0.60, 0.04]),
                    "ruido  p", 0.0, 0.5, valinit=0.0, valstep=0.01,
                    color=AZUL)
    slider.on_changed(atualizar)

    def sortear(evento):
        estado["semente"] += 1
        atualizar()

    botao = Button(fig.add_axes([0.84, 0.06, 0.12, 0.07]), "Outro sorteio")
    botao.on_clicked(sortear)

    # mantem os widgets vivos (ver comentario na demo 1)
    fig._widgets = [slider, botao]

    atualizar()
    plt.show()
    return fig


# ===========================================================================
# DEMO 3 - DESENHAR O DIGITO COM O MOUSE (e ensinar a rede)
# ===========================================================================
def demo_desenhar():
    """
    Desenhe um digito e veja a rede classificar na hora.

    Mouse:
      botao esquerdo  pinta o pixel
      botao direito   apaga o pixel

    Teclado:
      0 a 9   ensina a rede: "o que eu desenhei e este digito".
              A rede aplica a regra de aprendizado repetidas vezes,
              ate acertar o seu desenho.
      r       reseta a rede (volta a treinar so nos 10 padroes originais)
      c       limpa a tela

    O grafico de barras mostra o y_in de cada neuronio. A linha do zero
    importa: um neuronio so "reconhece" quando seu y_in fica ACIMA dela.
    Se todas as barras estiverem abaixo de zero, nenhum neuronio reconheceu
    o desenho -- e a rede nao tem resposta de verdade para dar.
    """
    X, T, rotulos = carregar_digitos()
    rede = Perceptron()
    rede.treinar(X, T, mostrar=False)

    tela = -np.ones((LADO, LADO))
    estado = {"pintando": None, "mensagem": "", "cor_msg": "black"}

    fig = plt.figure(figsize=(12.5, 6.8))
    fig.canvas.manager.set_window_title("Demo 3 - Desenhe e ensine a rede")
    ax_desenho = fig.add_axes([0.05, 0.28, 0.40, 0.56])
    ax_barras = fig.add_axes([0.55, 0.28, 0.42, 0.56])

    im = _desenhar_grade(ax_desenho, tela)
    ax_desenho.set_title("Desenhe aqui\n"
                         "esquerdo pinta  |  direito apaga", fontsize=10)

    barras = ax_barras.bar(range(10), np.zeros(10), color=CINZA)
    ax_barras.set_xticks(range(10))
    ax_barras.set_xlabel("neuronio (digito)")
    ax_barras.set_ylabel("entrada liquida  y_in")
    ax_barras.axhline(0, color="black", lw=1.2)
    ax_barras.grid(axis="y", alpha=0.3, linestyle=":")
    ax_barras.set_title("acima de zero = o neuronio reconheceu", fontsize=9)

    titulo = fig.suptitle("", fontsize=13)
    rodape = fig.text(0.5, 0.15, "", ha="center", fontsize=10, color="#444444")
    ajuda = fig.text(
        0.5, 0.035,
        "teclas:  0-9 ensina a rede o digito que voce desenhou   |   "
        "r reseta a rede   |   c limpa a tela",
        ha="center", fontsize=9.5, color="#666666")

    # -- atualizacao da tela ---------------------------------------------
    def atualizar():
        y_in = rede.entrada_liquida(tela.reshape(-1))[0]
        vencedor = int(np.argmax(y_in))
        acesos = int((tela > 0).sum())
        # quantos neuronios de fato dispararam (+1)
        ativos = int((y_in >= rede.limiar).sum())

        for j, barra in enumerate(barras):
            barra.set_height(y_in[j])
            if j == vencedor and ativos >= 1:
                barra.set_color(VERDE)
            elif y_in[j] >= rede.limiar:
                barra.set_color("#a9cce3")
            else:
                barra.set_color(CINZA)

        ax_barras.relim()
        ax_barras.autoscale_view()

        ordenado = np.sort(y_in)
        margem = ordenado[-1] - ordenado[-2]

        # ---- a parte importante: so afirmar quando ha o que afirmar ----
        if acesos == 0:
            titulo.set_text("Tela vazia -- desenhe alguma coisa")
            titulo.set_color(CINZA)
            rodape.set_text("")
        elif ativos == 0:
            # nenhuma barra passou do zero: a rede nao reconheceu nada.
            titulo.set_text("Nenhum neuronio reconheceu este desenho")
            titulo.set_color(LARANJA)
            rodape.set_text(
                f"todos os y_in estao abaixo de zero. O maior deles e o do "
                f"neuronio {vencedor} ({y_in[vencedor]:.0f}), mas isso e so o "
                f"'menos ruim' -- nao e um reconhecimento.\n"
                f"Aperte a tecla do digito correto para ENSINAR a rede.")
        elif ativos > 1:
            empatados = [j for j in range(10) if y_in[j] >= rede.limiar]
            titulo.set_text(f"Resposta ambigua: {ativos} neuronios "
                            f"reconheceram  {empatados}")
            titulo.set_color(LARANJA)
            rodape.set_text(
                f"o mais forte e o {vencedor} (y_in = {y_in[vencedor]:.0f}), "
                f"com margem de apenas {margem:.0f} para o segundo.\n"
                f"Aperte a tecla do digito correto para ENSINAR a rede.")
        else:
            titulo.set_text(f"A rede diz:  {vencedor}")
            titulo.set_color(VERDE)
            rodape.set_text(
                f"y_in = {y_in[vencedor]:.0f}   |   margem para o segundo = "
                f"{margem:.0f}   |   {acesos} pixels acesos")

        if estado["mensagem"]:
            rodape.set_text(estado["mensagem"])
            rodape.set_color(estado["cor_msg"])
        else:
            rodape.set_color("#444444")

        im.set_data(tela)
        fig.canvas.draw_idle()

    # -- ENSINAR: aplica a regra do perceptron ate acertar ----------------
    def ensinar(digito, max_tentativas=100):
        """
        Treina a rede no desenho atual, repetindo a regra de correcao
        ate que a saida bipolar fique exatamente igual ao alvo.

        E o mesmo Passo 5 do algoritmo, so que aplicado a um unico
        padrao -- o que voce acabou de desenhar.
        """
        if (tela > 0).sum() == 0:
            estado["mensagem"] = "Tela vazia: desenhe algo antes de ensinar."
            estado["cor_msg"] = LARANJA
            atualizar()
            return

        x = tela.reshape(-1)
        alvo = -np.ones(10)
        alvo[digito] = 1.0

        correcoes = 0
        for tentativa in range(1, max_tentativas + 1):
            y_in = rede.b + x @ rede.W
            y = rede.ativacao(y_in)
            errou = y != alvo
            if not errou.any():
                break                      # acertou: nada mais a corrigir
            ajuste = rede.alfa * alvo * errou
            rede.W += np.outer(x, ajuste)
            rede.b += ajuste
            correcoes += int(errou.sum())
        else:
            tentativa = max_tentativas

        # a rede aprendeu o desenho -- mas sera que esqueceu os originais?
        ainda_certos = int((rede.classificar(X) == rotulos).sum())
        predito = int(rede.classificar(x)[0])

        if ainda_certos == 10:
            estado["cor_msg"] = VERDE
            aviso = "e continua acertando os 10 digitos originais."
        else:
            estado["cor_msg"] = LARANJA
            aviso = (f"MAS agora so acerta {ainda_certos}/10 dos digitos "
                     f"originais -- aprender um padrao novo mexeu nos pesos "
                     f"dos outros. Aperte 'r' para resetar a rede.")

        estado["mensagem"] = (
            f"Ensinei que isto e um {digito}: aprendeu em {tentativa} "
            f"passada(s), {correcoes} correcoes de peso. Agora responde "
            f"{predito}.\n{aviso}")
        atualizar()

    def resetar_rede():
        rede.W[:] = 0.0
        rede.b[:] = 0.0
        rede.historico.clear()
        rede.treinar(X, T, mostrar=False)
        estado["mensagem"] = ("Rede resetada: treinada de novo apenas nos 10 "
                              "digitos originais.")
        estado["cor_msg"] = AZUL
        atualizar()

    # -- eventos do mouse -------------------------------------------------
    def pixel_do_evento(evento):
        if evento.inaxes is not ax_desenho:
            return None
        if evento.xdata is None or evento.ydata is None:
            return None
        col = int(round(evento.xdata))
        lin = int(round(evento.ydata))
        if 0 <= lin < LADO and 0 <= col < LADO:
            return lin, col
        return None

    def aplicar(evento):
        pos = pixel_do_evento(evento)
        if pos is None or estado["pintando"] is None:
            return
        if tela[pos] == estado["pintando"]:
            return                       # ja esta assim, evita redesenhar
        tela[pos] = estado["pintando"]
        estado["mensagem"] = ""          # desenhar limpa o aviso anterior
        atualizar()

    def ao_pressionar(evento):
        if evento.button == 1:
            estado["pintando"] = 1.0
        elif evento.button == 3:
            estado["pintando"] = -1.0
        else:
            return
        aplicar(evento)

    def ao_arrastar(evento):
        if estado["pintando"] is not None:
            aplicar(evento)

    def ao_soltar(evento):
        estado["pintando"] = None

    def ao_teclar(evento):
        if evento.key is None:
            return
        if evento.key in "0123456789":
            ensinar(int(evento.key))
        elif evento.key == "r":
            resetar_rede()
        elif evento.key == "c":
            limpar(None)

    fig.canvas.mpl_connect("button_press_event", ao_pressionar)
    fig.canvas.mpl_connect("motion_notify_event", ao_arrastar)
    fig.canvas.mpl_connect("button_release_event", ao_soltar)
    fig.canvas.mpl_connect("key_press_event", ao_teclar)

    # -- botoes -----------------------------------------------------------
    def limpar(evento):
        tela[:] = -1.0
        estado["mensagem"] = ""
        atualizar()

    def carregar_exemplo(evento):
        d = int(np.random.default_rng().integers(0, 10))
        tela[:] = para_imagem(X[d])
        estado["mensagem"] = ""
        atualizar()

    def botao_resetar(evento):
        resetar_rede()

    b1 = Button(fig.add_axes([0.08, 0.06, 0.13, 0.07]), "Limpar (c)")
    b1.on_clicked(limpar)
    b2 = Button(fig.add_axes([0.24, 0.06, 0.20, 0.07]), "Carregar exemplo")
    b2.on_clicked(carregar_exemplo)
    b3 = Button(fig.add_axes([0.47, 0.06, 0.18, 0.07]), "Resetar rede (r)")
    b3.on_clicked(botao_resetar)

    # mantem os widgets vivos (ver comentario na demo 1)
    fig._widgets = [b1, b2, b3]
    fig._ensinar = ensinar          # exposto para os testes

    atualizar()
    plt.show()
    return fig


# ===========================================================================
# MENU
# ===========================================================================
DEMOS = {
    "1": ("Treino passo a passo (ver os pesos se formando)", demo_treino),
    "2": ("Ruido com controle deslizante", demo_ruido),
    "3": ("Desenhar um digito e ENSINAR a rede", demo_desenhar),
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in DEMOS:
        DEMOS[sys.argv[1]][1]()
        return

    print("=" * 56)
    print("DEMONSTRACOES INTERATIVAS - PERCEPTRON")
    print("=" * 56)
    for chave, (descricao, _) in DEMOS.items():
        print(f"  {chave}. {descricao}")
    print("  q. sair")

    while True:
        escolha = input("\nEscolha uma demo: ").strip().lower()
        if escolha in ("q", "sair", ""):
            break
        if escolha in DEMOS:
            print(f"Abrindo: {DEMOS[escolha][0]}")
            print("(feche a janela para voltar ao menu)")
            DEMOS[escolha][1]()
        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    main()
