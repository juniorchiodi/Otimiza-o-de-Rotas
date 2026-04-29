import os
import re
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from utils.formatadores import remover_acentos, extrair_cidade, formatar_minutos
from utils.console import print_colorido, Fore

class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

def gerar_pdf_rota(cidade, ponto_partida, distancia_total, tempo_total_min,
                   estatisticas_cidade, arquivos_qr_gerados,
                   nomes_ordenados, enderecos_ordenados, links,
                   distancias_parciais, duracoes_parciais,
                   enderecos_com_erro, nomes):

    print_colorido("\n📄 Gerando PDF...", Fore.CYAN)
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, "assets", "logo.png")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=170, y=10, w=31.5)

    pdf.set_font("Helvetica", "B", 16)
    data_atual = datetime.now().strftime("%d/%m/%Y")
    titulo = f"Rota de Entregas - {cidade}"

    cidade_segura = re.sub(r'[\\/*?:"<>|]', "", cidade)
    nome_arquivo = remover_acentos(f"{datetime.now().strftime('%Y-%m-%d')} - Rota de Entregas - {cidade_segura}").replace("/", "-")

    pasta_rotas = os.path.join(base_dir, "ROTAS-GERADAS")
    if not os.path.exists(pasta_rotas): os.makedirs(pasta_rotas)
    arquivo_saida_pdf = os.path.join(pasta_rotas, f"{nome_arquivo}.pdf")

    pdf.cell(0, 10, titulo, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 6, f"Gerado em: {data_atual}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Resumo da Rota", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)

    texto_resumo = (
        f"Ponto de Partida: {ponto_partida}\n"
        f"Distância Total: {distancia_total:.1f} km\n"
        f"Tempo Total de Viagem: {formatar_minutos(tempo_total_min)}\n"
        f"Número Total de Entregas: {len(enderecos_ordenados) - 1}\n"
        f"Resumo por Cidade:\n"
    )
    for cid, stats in estatisticas_cidade.items():
        texto_resumo += f"- Entregas em {cid}: {stats['entregas']} Entregas - {stats['distancia']:.1f} km - {formatar_minutos(stats['tempo'])}\n"

    pdf.multi_cell(0, 6, texto_resumo.strip(), border=1, align="L")
    pdf.ln(8)

    if arquivos_qr_gerados:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Navegação GPS Automatizada (QR Codes)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, "Escaneie com a câmera do celular para abrir o trajeto no Maps (Dividido em partes devido ao limite do Google de 10 paradas por vez).")
        x_start, y_start, qr_size = 10, pdf.get_y(), 35
        for idx, qr_file in enumerate(arquivos_qr_gerados):
            if x_start + qr_size > 190:
                x_start = 10; y_start += qr_size + 5
            pdf.image(qr_file, x=x_start, y=y_start, w=qr_size)
            pdf.set_xy(x_start, y_start + qr_size)
            pdf.cell(qr_size, 5, f"Parte {idx+1}", align="C")
            x_start += qr_size + 5
        pdf.set_y(y_start + qr_size + 10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Ordem de Visita", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(255, 0, 0); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 8)

    w_num, w_nome, w_end, w_dist, w_tempo, w_link = 12, 48, 64, 18, 18, 30

    pdf.cell(w_num, 8, "Check", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
    pdf.cell(w_nome, 8, "Nome", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
    pdf.cell(w_end, 8, "Endereço", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
    pdf.cell(w_dist, 8, "Dist (km)", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
    pdf.cell(w_tempo, 8, "Tempo", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
    pdf.cell(w_link, 8, "Link (Maps)", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 8)

    cor_linha, cidade_anterior = 0, None

    for i, (nome, endereco, link) in enumerate(zip(nomes_ordenados[1:], enderecos_ordenados[1:], links[1:]), 1):
        cidade_atual = extrair_cidade(endereco)
        if cidade_anterior and cidade_atual != cidade_anterior:
            pdf.set_fill_color(255, 242, 168); pdf.set_font("Helvetica", "B", 9)
            pdf.cell(190, 6, f"  >>> MUDANÇA DE CIDADE: INDO PARA {cidade_atual.upper()} <<<", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
            pdf.set_font("Helvetica", "", 8)
        cidade_anterior = cidade_atual

        dist_trecho, dur_trecho = distancias_parciais[i-1], duracoes_parciais[i-1]
        pdf.set_fill_color(245, 245, 245) if cor_linha % 2 == 0 else pdf.set_fill_color(255, 255, 255)

        x_inicial, y_inicial, line_height, altura_max = pdf.get_x(), pdf.get_y(), 5, 5

        pdf.multi_cell(w_num, line_height, f"[  ] {i}", border=0, align='C', fill=True)
        h = pdf.get_y() - y_inicial; altura_max = max(h, altura_max)
        pdf.set_xy(x_inicial + w_num, y_inicial)

        pdf.multi_cell(w_nome, line_height, str(nome), border=0, align='L', fill=True)
        h = pdf.get_y() - y_inicial; altura_max = max(h, altura_max)
        pdf.set_xy(x_inicial + w_num + w_nome, y_inicial)

        pdf.multi_cell(w_end, line_height, str(endereco), border=0, align='L', fill=True)
        h = pdf.get_y() - y_inicial; altura_max = max(h, altura_max)

        pdf.set_xy(x_inicial, y_inicial)
        pdf.cell(w_num, altura_max, "", border=1)
        pdf.cell(w_nome, altura_max, "", border=1)
        pdf.cell(w_end, altura_max, "", border=1)

        pdf.set_xy(x_inicial + w_num + w_nome + w_end, y_inicial)
        pdf.cell(w_dist, altura_max, f"{dist_trecho:.1f}", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
        pdf.cell(w_tempo, altura_max, f"{int(round(dur_trecho))} min", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)

        pdf.set_text_color(0, 0, 255); pdf.set_font("Helvetica", "U", 8)
        pdf.cell(w_link, altura_max, "Abrir no Maps", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True, link=link)
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(0, 0, 0)

        cor_linha += 1
        if pdf.get_y() > 260:
            pdf.add_page()
            pdf.set_fill_color(255, 0, 0); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 8)
            pdf.cell(w_num, 8, "Check", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
            pdf.cell(w_nome, 8, "Nome", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
            pdf.cell(w_end, 8, "Endereço", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
            pdf.cell(w_dist, 8, "Dist (km)", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
            pdf.cell(w_tempo, 8, "Tempo", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
            pdf.cell(w_link, 8, "Link (Maps)", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
            pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 8)

    if enderecos_com_erro:
        if pdf.get_y() > 220: pdf.add_page()
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Apêndice A: Endereços com Erro", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "A lista abaixo contém endereços não encontrados. As linhas correspondentes no Excel foram marcadas em vermelho.", align="L")
        pdf.ln(3)
        pdf.set_fill_color(255, 0, 0); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 10)
        pdf.cell(15, 8, "Linha", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
        pdf.cell(45, 8, "Nome", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
        pdf.cell(80, 8, "Endereço", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True)
        pdf.cell(50, 8, "Motivo do Erro", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
        pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 8)
        for linha, endereco, motivo in enderecos_com_erro:
            n_cli = nomes[linha - 1] if 0 <= linha - 1 < len(nomes) else ""
            pdf.cell(15, 8, str(linha), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")
            pdf.cell(45, 8, str(n_cli)[:28], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
            pdf.cell(80, 8, str(endereco)[:50], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
            pdf.cell(50, 8, str(motivo)[:30], border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

    pdf.output(arquivo_saida_pdf)
    print_colorido(f"\n✅ PDF gerado com sucesso: {os.path.abspath(arquivo_saida_pdf)}", Fore.GREEN)
