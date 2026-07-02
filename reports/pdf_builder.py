import os
import re
from datetime import datetime
from utils.formatadores import remover_acentos, extrair_cidade, formatar_minutos
from utils.console import print_colorido, Fore
import jinja2
import weasyprint
import base64

def gerar_pdf_rota(cidade, ponto_partida, distancia_total, tempo_total_min,
                   estatisticas_cidade, arquivos_qr_gerados,
                   nomes_ordenados, enderecos_ordenados, links,
                   distancias_parciais, duracoes_parciais,
                   enderecos_com_erro, nomes):

    print_colorido("\n📄 Gerando PDF...", Fore.CYAN)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, "assets", "logo.png")

    logo_base64 = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logo_base64 = f"data:image/png;base64,{encoded_string}"

    data_atual = datetime.now().strftime("%d/%m/%Y")

    cidade_segura = re.sub(r'[\\/*?:"<>|]', "", cidade)
    nome_arquivo = remover_acentos(f"{datetime.now().strftime('%Y-%m-%d')} - Rota de Entregas - {cidade_segura}").replace("/", "-")

    pasta_rotas = os.path.join(base_dir, "ROTAS-GERADAS")
    if not os.path.exists(pasta_rotas): os.makedirs(pasta_rotas)
    arquivo_saida_pdf = os.path.join(pasta_rotas, f"{nome_arquivo}.pdf")

    # Format stats
    estatisticas_cidade_formatado = {}
    for cid, stats in estatisticas_cidade.items():
        estatisticas_cidade_formatado[cid] = {
            'entregas': stats['entregas'],
            'distancia': stats['distancia'],
            'tempo_formatado': formatar_minutos(stats['tempo'])
        }

    # Prepare list of stops
    paradas = []
    cidade_anterior = None
    for i, (nome, endereco, link) in enumerate(zip(nomes_ordenados[1:], enderecos_ordenados[1:], links[1:]), 1):
        cidade_atual = extrair_cidade(endereco)
        mudanca_cidade = False
        if cidade_anterior and cidade_atual != cidade_anterior:
            mudanca_cidade = True
        cidade_anterior = cidade_atual

        paradas.append({
            'mudanca_cidade': mudanca_cidade,
            'cidade_atual': cidade_atual,
            'nome': str(nome),
            'endereco': str(endereco),
            'dist_trecho': distancias_parciais[i-1],
            'dur_trecho': int(round(duracoes_parciais[i-1])),
            'link': link
        })

    # Prepare errors
    erros_formatados = []
    for linha, endereco, motivo in enderecos_com_erro:
        n_cli = nomes[linha - 1] if 0 <= linha - 1 < len(nomes) else ""
        erros_formatados.append({
            'linha': linha,
            'nome': n_cli,
            'endereco': endereco,
            'motivo': motivo
        })

    template_path = os.path.join(base_dir, "templates", "route_template.html")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_str = f.read()

    env = jinja2.Environment()
    template = env.from_string(template_str)

    html_content = template.render(
        cidade=cidade,
        data_atual=data_atual,
        ponto_partida=ponto_partida,
        distancia_total=f"{distancia_total:.1f} km",
        tempo_total=formatar_minutos(tempo_total_min),
        entregas_total=len(enderecos_ordenados) - 1,
        logo_base64=logo_base64,
        estatisticas_cidade=estatisticas_cidade_formatado,
        arquivos_qr_gerados=[os.path.abspath(qr) for qr in arquivos_qr_gerados] if arquivos_qr_gerados else [],
        paradas=paradas,
        enderecos_com_erro=erros_formatados
    )

    weasyprint.HTML(string=html_content, base_url=base_dir).write_pdf(arquivo_saida_pdf)

    print_colorido(f"\n✅ PDF gerado com sucesso: {os.path.abspath(arquivo_saida_pdf)}", Fore.GREEN)
