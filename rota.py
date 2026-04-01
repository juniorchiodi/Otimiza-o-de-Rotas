import os
import sys
import traceback
from tqdm import tqdm

from utils.console import print_colorido, Fore, Style
from utils.config_loader import carregar_config
from utils.formatadores import is_coordenada, limpar_endereco, extrair_coordenada, extrair_cidade, enriquecer_endereco

from core.data_manager import ler_planilha_excel, marcar_enderecos_erro_excel
from core.geocoding import carregar_cache, salvar_cache, processar_endereco, geocodificar_endereco
from core.routing import (calcular_matriz_distancia_osrm, identificar_outliers,
                          encontrar_melhor_rota_ortools, encontrar_melhor_rota_2opt, ORTOOLS_AVAILABLE)

from reports.qrcode_maker import gerar_qrcodes_rota, apagar_qrcodes
from reports.pdf_builder import gerar_pdf_rota

def main():
    config = carregar_config()
    arquivo_excel = config.get("arquivo_excel", "ENDERECOS-ROTA.xlsx")
    nome_coluna_enderecos = config.get("nome_coluna_enderecos", "Endereco")
    nome_coluna_nomes = config.get("nome_coluna_nomes", "Nome")
    ponto_partida_bruto = config.get("ponto_partida", "Rua Floriano Peixoto, 368, Centro, Itapuí - SP")
    ponto_partida = ponto_partida_bruto if is_coordenada(ponto_partida_bruto) else limpar_endereco(ponto_partida_bruto)

    cidade = sys.argv[1] if len(sys.argv) > 1 else input("Digite a cidade das entregas: ").strip()

    try:
        print_colorido("\n🚀 Iniciando processamento...", Fore.GREEN, Style.BRIGHT)

        # --- 1. Leitura de Dados ---
        nomes, enderecos = ler_planilha_excel(arquivo_excel, nome_coluna_nomes, nome_coluna_enderecos)

        coordenadas, enderecos_validos, nomes_validos, enderecos_com_erro = [], [], [], []
        cache = carregar_cache()

        ponto_partida_enriquecido = enriquecer_endereco(ponto_partida, cidade)
        
        # --- 2. Ponto de Partida ---
        if is_coordenada(ponto_partida):
            coords = extrair_coordenada(ponto_partida)
            if coords:
                coordenadas.append(coords); enderecos_validos.append(ponto_partida); nomes_validos.append("Ponto de Partida")
        elif ponto_partida in cache:
            coordenadas.append(tuple(cache[ponto_partida]['coords'])); enderecos_validos.append(ponto_partida); nomes_validos.append("Ponto de Partida")
        else:
            resultado = geocodificar_endereco(ponto_partida_enriquecido)
            if resultado and 'coords' in resultado:
                coordenadas.append(resultado['coords']); enderecos_validos.append(ponto_partida); nomes_validos.append("Ponto de Partida")
                cache[ponto_partida] = resultado
            else:
                print_colorido("❌ Erro fatal com o Ponto de Partida.", Fore.RED); exit(1)

        # --- 3. Geocodificação ---
        print_colorido("\n🔄 Geocodificando endereços...", Fore.CYAN)
        resultados = []
        for end in tqdm(enderecos, desc="Progresso", unit="endereço"):
            resultados.append(processar_endereco(end, cidade=cidade, cache=cache))
        salvar_cache(cache) 

        for idx_end, (endereco, coords, status, motivo_erro) in enumerate(resultados):
            if coords:
                coordenadas.append(tuple(coords))
                enderecos_validos.append(endereco)
                nomes_validos.append(nomes[idx_end]) 
            else:
                enderecos_com_erro.append((idx_end + 1, endereco, motivo_erro))
                
        if enderecos_com_erro: marcar_enderecos_erro_excel(arquivo_excel, enderecos_com_erro)
        if len(enderecos_validos) <= 1: exit(1)

        # --- 4. Roteamento (Distâncias e Otimização) ---
        dist_matrix, dur_matrix = calcular_matriz_distancia_osrm(coordenadas)
        pontos_principais, outliers_idx = identificar_outliers(dist_matrix, enderecos_validos)
        
        outliers_info = []
        if outliers_idx:
            for idx in outliers_idx:
                distancias_ponto = [dist_matrix[idx][j] for j in range(len(dist_matrix)) if idx != j and dist_matrix[idx][j] != float('inf')]
                media_ponto = (sum(distancias_ponto) / len(distancias_ponto)) if distancias_ponto else 0
                outliers_info.append((idx, nomes_validos[idx], enderecos_validos[idx], media_ponto))
                linha_excel = nomes.index(nomes_validos[idx]) + 1 if nomes_validos[idx] in nomes else 0
                enderecos_com_erro.append((linha_excel, enderecos_validos[idx], "Coordenada em outra cidade (Outlier)"))
            
            if 0 in outliers_idx: exit(1)
            coordenadas = [coordenadas[i] for i in pontos_principais]
            enderecos_validos = [enderecos_validos[i] for i in pontos_principais]
            nomes_validos = [nomes_validos[i] for i in pontos_principais] 
            dist_matrix, dur_matrix = calcular_matriz_distancia_osrm(coordenadas)

        outliers_idx_set = set()

        ordem_rota = None
        if ORTOOLS_AVAILABLE:
            ordem_rota = encontrar_melhor_rota_ortools(dist_matrix)
        
        if not ordem_rota:
            ordem_rota = encontrar_melhor_rota_2opt(dist_matrix)

        # --- 5. Resultados e Estatísticas ---
        distancia_total, tempo_total_min, distancias_parciais, duracoes_parciais = 0, 0, [], []
        estatisticas_cidade = {}
        
        for i in range(len(ordem_rota) - 1):
            origem, destino = ordem_rota[i], ordem_rota[i+1]
            dist, dur = dist_matrix[origem][destino], dur_matrix[origem][destino]
            distancia_total += dist
            tempo_total_min += dur
            distancias_parciais.append(dist)
            duracoes_parciais.append(dur)
            
            cidade_dest = extrair_cidade(enderecos_validos[destino])
            if cidade_dest not in estatisticas_cidade:
                estatisticas_cidade[cidade_dest] = {'entregas': 0, 'distancia': 0, 'tempo': 0}
            
            estatisticas_cidade[cidade_dest]['entregas'] += 1
            estatisticas_cidade[cidade_dest]['distancia'] += dist
            estatisticas_cidade[cidade_dest]['tempo'] += dur

        print_colorido(f"\n📊 Distância total da rota: {distancia_total:.2f} km", Fore.GREEN)
        print_colorido(f"⏱️ Tempo total estimado: {int(tempo_total_min // 60)}h {int(tempo_total_min % 60)}min", Fore.GREEN)

        enderecos_ordenados = [enderecos_validos[i] for i in ordem_rota]
        nomes_ordenados = [nomes_validos[i] for i in ordem_rota] 
        links = [f"https://www.google.com/maps/place/{coordenadas[i][0]},{coordenadas[i][1]}" for i in ordem_rota]

        # --- 6. Relatórios (QR e PDF) ---
        arquivos_qr_gerados = gerar_qrcodes_rota(coordenadas, ordem_rota)

        gerar_pdf_rota(cidade, ponto_partida, distancia_total, tempo_total_min,
                       estatisticas_cidade, arquivos_qr_gerados,
                       nomes_ordenados, enderecos_ordenados, links,
                       distancias_parciais, duracoes_parciais,
                       enderecos_com_erro, nomes)

        apagar_qrcodes(arquivos_qr_gerados)

    except Exception as e:
        print_colorido(f"\n❌ Erro inesperado: {traceback.format_exc()}", Fore.RED)

if __name__ == "__main__":
    main()
