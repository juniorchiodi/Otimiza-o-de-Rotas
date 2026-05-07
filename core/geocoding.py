import os
import json
import time
import requests
import threading
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from utils.console import print_colorido, Fore
from utils.formatadores import enriquecer_endereco, is_coordenada, extrair_coordenada, remover_acentos

# Configuração do geocodificador Nominatim (Agora 100% Local e Sem Limites!)
geolocator = Nominatim(
    #domain="192.168.88.94:5001",
    #scheme="http",
    user_agent="juninho.junirj@gmail.com"
)

# Lock global para chamadas ao Nominatim e salvamento no cache (thread safety)
NOMINATIM_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()

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
    with CACHE_LOCK:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

def geocodificar_endereco_nominatim(query_params, endereco_legivel, max_tentativas=3, intervalo=2):
    motivo = "Falha na geocodificação"
    for tentativa in range(max_tentativas):
        try:
            with NOMINATIM_LOCK:
                time.sleep(1.2) # Hard rate-limit (Nominatim exige 1 req/sec absoluta max)
                location = geolocator.geocode(query_params, timeout=15)
            if location:
                return {
                    'coords': (location.latitude, location.longitude),
                    'address': location.address,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Endereço não encontrado'}
        except GeocoderTimedOut:
            motivo = "Timeout do serviço"
            print_colorido(f"{motivo} ao geocodificar '{endereco_legivel}' (tentativa {tentativa + 1}/{max_tentativas})", Fore.YELLOW)
        except GeocoderServiceError as e:
            if "429" in str(e):
                print_colorido("Limite atingido! Dormindo por 10 segundos...", Fore.RED)
                time.sleep(10)
                return {'error': "429: Limite excedido"}
            return {'error': f"Erro de serviço: {e}"}
        except Exception as e:
            motivo = f"Erro desconhecido: {e}"
            print_colorido(f"Erro inesperado ao geocodificar '{endereco_legivel}': {e}", Fore.RED)

        if tentativa < max_tentativas - 1:
            time.sleep(intervalo)
        else:
            return {'error': motivo}
    return {'error': motivo}

def geocodificar_endereco_photon(endereco_str, max_tentativas=3, intervalo=5):
    params = {'q': endereco_str, 'limit': 1}
    headers = {'User-Agent': 'juninho.junirj@gmail.com'}
    motivo = "Falha na geocodificação (Photon)"
    for tentativa in range(max_tentativas):
        try:
            response = requests.get("https://photon.komoot.io/api/", params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and 'features' in data and data['features']:
                lon, lat = data['features'][0]['geometry']['coordinates']
                props = data['features'][0]['properties']
                rua = props.get('street', props.get('name', ''))
                num = props.get('housenumber', '')
                cidade = props.get('city', props.get('town', ''))
                # Constrói a string do endereço encontrado
                addr_encontrado = f"{rua}, {num} - {cidade}".strip(", -")
                return {
                    'coords': (lat, lon),
                    'address': addr_encontrado,
                    'timestamp': datetime.now().isoformat()
                }
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

def validar_cidade_reversa(coords, cidade_esperada):
    """
    Geocodificação Reversa via Photon para validar se a coordenada encontrada
    realmente pertence à cidade solicitada (evita rotas de 10.000km de erro).
    """
    if not cidade_esperada:
        return True, "Validação ignorada (cidade vazia)"

    lat, lon = coords
    try:
        response = requests.get(f"https://photon.komoot.io/reverse?lon={lon}&lat={lat}", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and 'features' in data and data['features']:
            props = data['features'][0]['properties']
            cidade_encontrada = props.get('city', props.get('town', props.get('village', props.get('county', ''))))

            cidade_esperada_norm = remover_acentos(str(cidade_esperada).lower()).strip()
            cidade_encontrada_norm = remover_acentos(str(cidade_encontrada).lower()).strip()

            # Checagem leve (substring)
            if cidade_esperada_norm in cidade_encontrada_norm or cidade_encontrada_norm in cidade_esperada_norm:
                return True, ""
            else:
                return False, f"Cidade divergente (Esperada: {cidade_esperada}, Encontrada: {cidade_encontrada})"
    except requests.exceptions.RequestException:
        return True, "Falha na validação (Fallback aceito)"
    return True, "Cidade não identificada (Aceito por padrão)"

def construir_string_endereco(end_dict):
    """
    Constrói a string de endereço a partir do dicionário estruturado.
    """
    if isinstance(end_dict, str):
        return end_dict

    logradouro = end_dict.get('logradouro', '')
    numero = end_dict.get('numero', '')
    bairro = end_dict.get('bairro', '')
    cidade = end_dict.get('cidade', '')
    cep = end_dict.get('cep', '')

    partes = []
    if logradouro: partes.append(logradouro)
    if numero: partes.append(numero)
    if bairro: partes.append(bairro)
    if cidade: partes.append(f"{cidade} - SP")
    if cep: partes.append(cep)

    # Photon Query String
    photon_query = ", ".join(partes) + ", Brasil" if partes else ""
    return photon_query

def geocodificar_endereco_estruturado(end_dict, max_tentativas=3, intervalo=2):
    """
    Implementa o fallback:
    1. Photon (String Completa)
    2. Nominatim (Estruturada COM Número)
    3. Nominatim (Estruturada SEM Número)
    """
    cidade_esperada = end_dict.get('cidade', '')
    endereco_str = construir_string_endereco(end_dict)

    # ==========================================
    # TENTATIVA 1: Photon (Busca em string única)
    # ==========================================
    resultado = geocodificar_endereco_photon(endereco_str, max_tentativas=1, intervalo=intervalo)
    provider = 'Photon'

    if resultado and 'coords' in resultado:
        valido, motivo = validar_cidade_reversa(resultado['coords'], cidade_esperada)
        if valido:
            resultado['provider'] = provider
            return resultado

    # ==========================================
    # TENTATIVA 2: Nominatim (Busca Estruturada COM número)
    # ==========================================
    logradouro = end_dict.get('logradouro', '')
    numero = end_dict.get('numero', '')

    street_with_number = f"{logradouro} {numero}".strip()

    params_nominatim_1 = {
        'street': street_with_number,
        'city': end_dict.get('cidade', ''),
        'state': 'SP',
        'postalcode': end_dict.get('cep', ''),
        'country': 'Brasil'
    }

    # Remove empty values to not confuse Nominatim
    params_nominatim_1 = {k: v for k, v in params_nominatim_1.items() if v}

    resultado = geocodificar_endereco_nominatim(params_nominatim_1, endereco_str, max_tentativas=1, intervalo=intervalo)
    provider = 'Nominatim-Exato'

    if resultado and 'coords' in resultado:
        valido, motivo = validar_cidade_reversa(resultado['coords'], cidade_esperada)
        if valido:
            resultado['provider'] = provider
            return resultado

    # ==========================================
    # TENTATIVA 3: Nominatim (Busca Estruturada SEM número)
    # ==========================================
    params_nominatim_2 = {
        'street': logradouro, # Apenas nome da rua
        'city': end_dict.get('cidade', ''),
        'state': 'SP',
        'postalcode': end_dict.get('cep', ''),
        'country': 'Brasil'
    }
    params_nominatim_2 = {k: v for k, v in params_nominatim_2.items() if v}

    resultado = geocodificar_endereco_nominatim(params_nominatim_2, endereco_str, max_tentativas=1, intervalo=intervalo)
    provider = 'Nominatim-SemNumero'

    if resultado and 'coords' in resultado:
        valido, motivo = validar_cidade_reversa(resultado['coords'], cidade_esperada)
        if valido:
            resultado['provider'] = provider
            return resultado
        else:
            return {'error': motivo}

    # Se todas falharem
    return {'error': "Nao_Encontrado"}

def processar_endereco(end_dict, cidade, cache):
    endereco_str = construir_string_endereco(end_dict)

    if is_coordenada(endereco_str):
        coords = extrair_coordenada(endereco_str)
        return (endereco_str, coords, 'coordenada', None, endereco_str) if coords else (endereco_str, None, 'erro', 'Coordenada inválida', None)

    # Chave para cache agora usa a string completa construída
    cache_key = endereco_str

    with CACHE_LOCK:
        if cache_key in cache:
            return (endereco_str, tuple(cache[cache_key]['coords']), f"cache ({cache[cache_key].get('provider', '?')})", None, cache[cache_key].get('address', 'Endereço do Cache'))

    # Se for apenas uma string (ex: Ponto de partida passado em rota.py), geocodifica só como string via Photon/Nominatim string (legado)
    if isinstance(end_dict, str):
        # Para string simples (ponto de partida não estruturado) vamos usar Nominatim fallback (como era antes)
        resultado = geocodificar_endereco_nominatim(end_dict, endereco_legivel=end_dict, max_tentativas=2)
        provider = 'Nominatim-String'
        if not (resultado and 'coords' in resultado):
            resultado = geocodificar_endereco_photon(end_dict, max_tentativas=1)
            provider = 'Photon-String'

        if resultado and 'coords' in resultado:
             valido, motivo = validar_cidade_reversa(resultado['coords'], cidade)
             if valido:
                 resultado['provider'] = provider
             else:
                 resultado = {'error': motivo}
    else:
        # Caminho Novo: Estruturado
        resultado = geocodificar_endereco_estruturado(end_dict)

    if resultado and 'coords' in resultado:
        with CACHE_LOCK:
            cache[cache_key] = resultado
        salvar_cache(cache)
        return (endereco_str, resultado['coords'], f"geocodificado ({resultado.get('provider', '?')})", None, resultado.get('address', 'Endereço Encontrado'))

    return (endereco_str, None, 'erro', resultado.get('error', 'Erro desconhecido') if resultado else 'Erro', None)
