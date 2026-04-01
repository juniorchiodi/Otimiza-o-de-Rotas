import pandas as pd  
from fpdf import FPDF  # Geração de arquivos PDF
from fpdf.enums import XPos, YPos  # Controle de quebra de linha do FPDF atualizado
from geopy.distance import geodesic  # Cálculo de distância geodésica
from geopy.geocoders import Nominatim  # Geocodificador Nominatim (OSM)
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import numpy as np  # Operações numéricas e matrizes
import os  # arquivos
import unicodedata  # Remoção de acentos/normalização de texto
import json  
from datetime import datetime, timedelta  
from tqdm import tqdm  
import colorama
from colorama import Fore, Style  
import requests  # Requisições HTTP
import openpyxl  
from openpyxl.styles import PatternFill  # Estilo de células (preenchimento)
import time  # Pausas entre requisições e backoff
import re  # Expressões regulares

# Importações das novas bibliotecas com proteção (Fallback automático)
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

colorama.init()  # Habilita cores no terminal no Windows

# Configuração do geocodificador Nominatim
geolocator = Nominatim(user_agent="juninho.junirj@gmail.com")

def print_colorido(texto, cor=Fore.WHITE, estilo=Style.NORMAL):  
    print(f"{estilo}{cor}{texto}{Style.RESET_ALL}")  

def is_coordenada(texto):  
    if not isinstance(texto, str):  
        return False
    padrao = r"^\s*(-?\d{1,2}\.\d+),\s*(-?\d{1,3}\.\d+)(?:\s*[,;]\s*.*)?$"  
    return re.match(padrao, texto) is not None  

def extrair_coordenada(texto):  
    padrao = r"^\s*(-?\d{1,2}\.\d+),\s*(-?\d{1,3}\.\d+)"  
    m = re.match(padrao, texto)  
    if m:  
        return (float(m.group(1)), float(m.group(2)))  
    return None  

def remover_acentos(texto):  
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                  if unicodedata.category(c) != 'Mn')  

def limpar_endereco(endereco):
    if not isinstance(endereco, str):
        return ""
    endereco = endereco.strip()
    endereco = re.sub(r'[^\w\s,-]', '', endereco)
    endereco = re.sub(r'\s+', ' ', endereco)
    return endereco

def enriquecer_endereco(endereco, cidade):
    if not isinstance(endereco, str):
        return ""
    endereco_str = endereco.strip()
    endereco_str = re.sub(r'\s*-\s*([A-Z]{2})\b', r', \1', endereco_str, flags=re.IGNORECASE)
    endereco_norm = remover_acentos(endereco_str).lower()
    
    if "brasil" not in endereco_norm:
        endereco_str = f"{endereco_str}, Brasil"
    return endereco_str

def extrair_cidade(endereco):
    end_upper = remover_acentos(endereco).upper()
    if re.search(r'\bJAU\b', end_upper): return "Jaú"
    if re.search(r'\bBARIRI\b', end_upper): return "Bariri"
    if re.search(r'\BITAPUI\b', end_upper): return "Itapuí"
    
    match = re.search(r'([^,]+?)(?:\s*-\s*SP|\s*,\s*SP)', endereco, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    return "Outra"

CACHE_FILE = "geocodificacao_cache.json"  
CACHE_EXPIRATION_DAYS = 30  

def carregar_cache():  
    if os.path.exists(CACHE_FILE):  
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:  
                cache_data = json.load(f)  
                current_time = datetime.now()  
                cache_data = {  
                    k: v for k, v in cache_data.items()
                    if datetime.fromisoformat(v['timestamp']) + timedelta(days=CACHE_EXPIRATION_DAYS) > current_time
                }
                return cache_data  
        except Exception as e:  
            return {}
    return {}

def salvar_cache(cache):  
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:  
            json.dump(cache, f, ensure_ascii=False, indent=2)  
    except Exception as e:  
        pass

def geocodificar_endereco_nominatim(endereco, max_tentativas=3, intervalo=2):
    motivo = "Falha na geocodificação"  
    for tentativa in range(max_tentativas):
        try:
            time.sleep(1.1) 
            location = geolocator.geocode(endereco, timeout=15)
            if location:
                return {'coords': (location.latitude, location.longitude), 'timestamp': datetime.now().isoformat()}
            else:
                return {'error': 'Endereço não encontrado'}
        except GeocoderTimedOut:
            motivo = "Timeout do serviço"
            print_colorido(f"{motivo} ao geocodificar '{endereco}' (tentativa {tentativa + 1}/{max_tentativas})", Fore.YELLOW)
        except GeocoderServiceError as e:
            if "429" in str(e):
                print_colorido("Limite atingido! Dormindo por 10 segundos...", Fore.RED)
                time.sleep(10) 
                return {'error': "429: Limite excedido"}
            return {'error': f"Erro de serviço: {e}"} 
        except Exception as e:
            motivo = f"Erro desconhecido: {e}"
            print_colorido(f"Erro inesperado ao geocodificar '{endereco}': {e}", Fore.RED)

        if tentativa < max_tentativas - 1:
            time.sleep(intervalo)
        else:
            return {'error': motivo}
    return {'error': motivo}

def geocodificar_endereco_photon(endereco, max_tentativas=3, intervalo=5):
    params = {'q': endereco, 'limit': 1}
    headers = {'User-Agent': 'juninho.junirj@gmail.com'} 
    motivo = "Falha na geocodificação (Photon)"
    for tentativa in range(max_tentativas):
        try:
            response = requests.get("https://photon.komoot.io/api/", params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and 'features' in data and data['features']:
                lon, lat = data['features'][0]['geometry']['coordinates']
                return {'coords': (lat, lon), 'timestamp': datetime.now().isoformat()}
            else:
                return {'error': 'Endereço não encontrado (Photon)'}
        except requests.exceptions.Timeout:
            motivo = "Timeout do serviço (Photon)"
        except requests.exceptions.RequestException as e:
            motivo = f"Erro de requisição (Photon): {e}"
            return {'error': motivo}

        if tentativa < max_tentativas - 1:
            time.sleep(intervalo)
        else:
            return {'error': motivo}
    return {'error': motivo}

def geocodificar_endereco(endereco, max_tentativas=3, intervalo=5):
    resultado_nominatim = geocodificar_endereco_nominatim(endereco, max_tentativas=2, intervalo=intervalo)
    if resultado_nominatim and 'coords' in resultado_nominatim:
        resultado_nominatim['provider'] = 'Nominatim'
        return resultado_nominatim

    motivo_falha_nominatim = resultado_nominatim.get('error', 'Falha') if resultado_nominatim else 'Falha'
    print_colorido(f"Falha no Nominatim para '{endereco}' ({motivo_falha_nominatim}). Tentando fallback...", Fore.YELLOW)

    resultado_photon = geocodificar_endereco_photon(endereco, max_tentativas=max_tentativas, intervalo=1)
    if resultado_photon and 'coords' in resultado_photon:
        resultado_photon['provider'] = 'Photon'
        return resultado_photon

    motivo_falha_photon = resultado_photon.get('error', 'Falha') if resultado_photon else 'Falha'
    return {'error': f"Nominatim: {motivo_falha_nominatim} | Photon: {motivo_falha_photon}"}

def calcular_distancia_rua(coords1, coords2):  
    try:
        lat1, lon1 = float(coords1[0]), float(coords1[1])  
        lat2, lon2 = float(coords2[0]), float(coords2[1])
        dist = geodesic((lat1, lon1), (lat2, lon2)).kilometers  
        return dist if dist <= 500 else float('inf')
    except:  
        return float('inf')  

def calcular_matriz_distancia_osrm(coordenadas):
    """ NOVO: Utiliza OSRM para distâncias reais de rua. Se falhar, usa o antigo geodésico. """
    n = len(coordenadas)
    dist_matrix = np.full((n, n), float('inf'))
    dur_matrix = np.full((n, n), float('inf'))

    try:
        print_colorido("Buscando distâncias reais nas rodovias (OSRM)...", Fore.CYAN)
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordenadas])
        url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok':
            distances = data['distances']
            durations = data['durations']
            for i in range(n):
                for j in range(n):
                    if distances[i][j] is not None:
                        dist_matrix[i][j] = distances[i][j] / 1000.0  # km
                        dur_matrix[i][j] = durations[i][j] / 60.0     # minutos
            print_colorido("✅ Matriz OSRM gerada com precisão!", Fore.GREEN)
            return dist_matrix, dur_matrix
    except Exception as e:
        print_colorido(f"⚠️ OSRM indisponível ({e}). Usando fallback (Distância Geodésica)...", Fore.YELLOW)

    # Fallback Geodésico
    with tqdm(total=n*n, desc="Calculando Matriz Geodésica") as pbar:
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0
                    dur_matrix[i][j] = 0
                else:
                    dist = calcular_distancia_rua(coordenadas[i], coordenadas[j]) * 1.4
                    dist_matrix[i][j] = dist
                    if dist != float('inf'):
                        dur_matrix[i][j] = (dist / 40.0) * 60.0
                pbar.update(1)
    return dist_matrix, dur_matrix

def identificar_outliers(dist_matrix, enderecos_validos, limite_desvio=2.5, p80_limite_km=300):  
    n = len(dist_matrix)  
    if n <= 1: return [], []

    distancias = [dist_matrix[i][j] for i in range(n) for j in range(n) if i != j and dist_matrix[i][j] != float('inf')]
    if not distancias: return [], []  

    media = sum(distancias) / len(distancias)  
    desvio = (sum((x - media) ** 2 for x in distancias) / len(distancias)) ** 0.5  
    outliers, pontos_principais = [], []  
    
    for i in range(n):  
        dist_ponto = [dist_matrix[i][j] for j in range(n) if i != j and dist_matrix[i][j] != float('inf')]
        if not dist_ponto: continue
        
        media_ponto = sum(dist_ponto) / len(dist_ponto)  
        p80 = sorted(dist_ponto)[max(0, min(len(dist_ponto) - 1, int(0.8 * (len(dist_ponto) - 1))))]
        
        if (media_ponto > media + (limite_desvio * desvio)) or (p80 > p80_limite_km):
            outliers.append(i)  
            print_colorido(f"Ponto identificado como outlier: {enderecos_validos[i]}", Fore.YELLOW)
        else:
            pontos_principais.append(i)  
    return pontos_principais, outliers  

def encontrar_melhor_rota_ortools(dist_matrix):
    """ NOVO: Motor de Roteamento Profissional Google OR-Tools """
    n = len(dist_matrix)
    if n <= 1: return [0]

    # Criação de um Nó Fantasma (Dummy Node) para permitir que a rota termine em qualquer lugar
    manager = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [n])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == n or to_node == n: return 0  # Custo zero para o nó final fantasma
        return int(dist_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5 # Pensa por 5 segundos para a rota perfeita

    print_colorido("\n🗺️ Otimizando a rota matematicamente com Google OR-Tools...", Fore.CYAN)
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        rota = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < n: rota.append(node)
            index = solution.Value(routing.NextVar(index))
        print_colorido("✅ Rota otimizada com sucesso!", Fore.GREEN)
        return rota
    else:
        return None

def encontrar_melhor_rota_2opt(dist_matrix):
    """ Fallback para o antigo 2-opt caso OR-Tools falhe ou não esteja instalado """
    n = len(dist_matrix)
    if n <= 1: return [0]
    print_colorido("\n🗺️ Otimizando a rota com algoritmo 2-opt (Fallback)...", Fore.YELLOW)
    ponto_atual = 0
    rota = [ponto_atual]
    nao_visitados = set(range(n)); nao_visitados.remove(ponto_atual)
    
    while nao_visitados:
        proximo = min(nao_visitados, key=lambda ponto: dist_matrix[ponto_atual][ponto])
        rota.append(proximo)
        nao_visitados.remove(proximo)
        ponto_atual = proximo

    melhorou = True
    tentativas = 0
    while melhorou and tentativas < 1000: 
        melhorou = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # O ERRO ESTAVA NA LINHA ABAIXO (onde estava rota[c], agora é rota[j])
                if dist_matrix[rota[i-1]][rota[j]] + dist_matrix[rota[i]][rota[j+1]] < dist_matrix[rota[i-1]][rota[i]] + dist_matrix[rota[j]][rota[j+1]]:
                    rota[i:j+1] = reversed(rota[i:j+1])
                    melhorou = True
                    break
            if melhorou: break
        tentativas += 1
    return rota

def carregar_config():
    try:
        with open("config.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print_colorido("❌ Erro: 'config.json' não encontrado.", Fore.RED)
        exit(1)

def main():
    config = carregar_config()
    arquivo_excel = config.get("arquivo_excel", "ENDERECOS-ROTA.xlsx")
    nome_coluna_enderecos = config.get("nome_coluna_enderecos", "Endereco")
    nome_coluna_nomes = config.get("nome_coluna_nomes", "Nome")
    ponto_partida_bruto = config.get("ponto_partida", "Rua Floriano Peixoto, 368, Centro, Itapuí - SP")
    ponto_partida = ponto_partida_bruto if is_coordenada(ponto_partida_bruto) else limpar_endereco(ponto_partida_bruto)

    import sys
    cidade = sys.argv[1] if len(sys.argv) > 1 else input("Digite a cidade das entregas: ").strip()

    try:
        print_colorido("\n🚀 Iniciando processamento...", Fore.GREEN, Style.BRIGHT)
        df = pd.read_excel(arquivo_excel)
        df = df[df[nome_coluna_nomes].notna() & (df[nome_coluna_nomes] != "")]
        enderecos_brutos = df[nome_coluna_enderecos].dropna().tolist()
        enderecos = [end if is_coordenada(end) else limpar_endereco(end) for end in enderecos_brutos]
        nomes = df[nome_coluna_nomes].fillna("").tolist()

        coordenadas, enderecos_validos, nomes_validos, enderecos_com_erro = [], [], [], []
        cache = carregar_cache()

        ponto_partida_enriquecido = enriquecer_endereco(ponto_partida, cidade)
        
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

        def processar_endereco(endereco, cidade, cache): 
            endereco_enriquecido = enriquecer_endereco(endereco, cidade)
            if is_coordenada(endereco):
                coords = extrair_coordenada(endereco)
                return (endereco, coords, 'coordenada', None) if coords else (endereco, None, 'erro', 'Coordenada inválida')
            if endereco in cache:
                return (endereco, tuple(cache[endereco]['coords']), f"cache ({cache[endereco].get('provider', '?')})", None)
            
            resultado = geocodificar_endereco(endereco_enriquecido)
            if resultado and 'coords' in resultado:
                cache[endereco] = resultado 
                salvar_cache(cache)         
                return (endereco, resultado['coords'], f"geocodificado ({resultado.get('provider', '?')})", None)
            return (endereco, None, 'erro', resultado.get('error', 'Erro desconhecido') if resultado else 'Erro')

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
                
        def marcar_enderecos_erro_excel(arquivo_excel, enderecos_com_erro):
            try:
                wb = openpyxl.load_workbook(arquivo_excel)
                ws = wb.active
                fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')
                for linha, _, _ in enderecos_com_erro:
                    for col in range(1, ws.max_column + 1):
                        ws.cell(row=linha + 1, column=col).fill = fill
                wb.save(arquivo_excel)
            except Exception as e: pass

        if enderecos_com_erro: marcar_enderecos_erro_excel(arquivo_excel, enderecos_com_erro)
        if len(enderecos_validos) <= 1: exit(1)

        # Distancias - Chamada atualizada com OSRM
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

        # Otimização OR-Tools ou 2-opt
        ordem_rota = None
        if ORTOOLS_AVAILABLE:
            ordem_rota = encontrar_melhor_rota_ortools(dist_matrix)
        
        if not ordem_rota:
            ordem_rota = encontrar_melhor_rota_2opt(dist_matrix)

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
        
        # NOVO: Links gerados com precisão cirúrgica de Latitude e Longitude!
        links = [f"https://www.google.com/maps/place/{coordenadas[i][0]},{coordenadas[i][1]}" for i in ordem_rota]

        # NOVO: Geração de QR Codes divididos em partes de 10 endereços
        arquivos_qr_gerados = []
        if QRCODE_AVAILABLE:
            print_colorido("📱 Gerando QR Codes para o motorista...", Fore.CYAN)
            chunk_size = 10
            for i in range(0, len(ordem_rota), chunk_size - 1):
                chunk = ordem_rota[i:i+chunk_size]
                if len(chunk) < 2: break
                
                orig_str = f"{coordenadas[chunk[0]][0]},{coordenadas[chunk[0]][1]}"
                dest_str = f"{coordenadas[chunk[-1]][0]},{coordenadas[chunk[-1]][1]}"
                url_maps = f"https://www.google.com/maps/dir/?api=1&origin={orig_str}&destination={dest_str}"
                
                if len(chunk) > 2:
                    wp_str = "|".join([f"{coordenadas[w][0]},{coordenadas[w][1]}" for w in chunk[1:-1]])
                    url_maps += f"&waypoints={wp_str}"
                
                qr = qrcode.make(url_maps)
                nome_qr = f"temp_qr_part_{i}.png"
                qr.save(nome_qr)
                arquivos_qr_gerados.append(nome_qr)

        print_colorido("\n📄 Gerando PDF...", Fore.CYAN)
        class PDF(FPDF):
            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        if os.path.exists("./assets/logo.png"):
            pdf.image("./assets/logo.png", x=170, y=10, w=31.5)

        pdf.set_font("Helvetica", "B", 16)
        data_atual = datetime.now().strftime("%d/%m/%Y")
        titulo = f"Rota de Entregas - {cidade}"
        
        nome_arquivo = remover_acentos(f"{datetime.now().strftime('%Y-%m-%d')} - Rota de Entregas - {cidade}").replace("/", "-")
        pasta_rotas = "ROTAS-GERADAS"
        if not os.path.exists(pasta_rotas): os.makedirs(pasta_rotas)
        arquivo_saida_pdf = os.path.join(pasta_rotas, f"{nome_arquivo}.pdf")

        pdf.cell(0, 10, titulo, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 6, f"Gerado em: {data_atual}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(8)

        def formatar_minutos(minutos):
            try:
                h, m = int(minutos // 60), int(round(minutos % 60))
                if m == 60: h += 1; m = 0
                return f"{h}h {m:02d}min" if h > 0 else f"{m}min"
            except: return "-"

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Resumo da Rota", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        
        # CORREÇÃO FPDF UNICODE: Usando hífen ("-") em vez da bolinha gorda ("•")
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

        # Injeção dos QR Codes na capa
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
        
        # NOVO: Larguras reajustadas para comportar o Checkbox " [ ] #1 "
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

            # NOVO CHECKBOX INJETADO AQUI
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
        print_colorido(f"\n✅ PDF gerado com sucesso: {arquivo_saida_pdf}", Fore.GREEN)

        # Deletar imagens QR temporárias para não poluir sua pasta
        for temp_img in arquivos_qr_gerados:
            try: os.remove(temp_img)
            except: pass

    except Exception as e:
        import traceback
        print_colorido(f"\n❌ Erro inesperado: {traceback.format_exc()}", Fore.RED)

if __name__ == "__main__":
    main()