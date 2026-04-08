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

geolocator = Nominatim(user_agent="juninho.junirj@gmail.com")

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

def geocodificar_endereco_nominatim(endereco, max_tentativas=3, intervalo=2):
    motivo = "Falha na geocodificação"
    for tentativa in range(max_tentativas):
        try:
            with NOMINATIM_LOCK:
                time.sleep(1.2) # Hard rate-limit (Nominatim exige 1 req/sec absoluta max)
                location = geolocator.geocode(endereco, timeout=15)
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

def geocodificar_endereco(endereco, cidade_esperada=None, max_tentativas=3, intervalo=5):
    # 1. Tentar Nominatim
    resultado = geocodificar_endereco_nominatim(endereco, max_tentativas=2, intervalo=intervalo)
    provider = 'Nominatim'

    # 2. Se falhar, Tentar Photon
    if not (resultado and 'coords' in resultado):
        motivo_falha = resultado.get('error', 'Falha') if resultado else 'Falha'
        # print_colorido(f"Falha no Nominatim para '{endereco}' ({motivo_falha}). Tentando fallback Photon...", Fore.YELLOW)
        resultado = geocodificar_endereco_photon(endereco, max_tentativas=max_tentativas, intervalo=1)
        provider = 'Photon'

    # 3. Validar se pegou coords e pertence à cidade
    if resultado and 'coords' in resultado:
        coords = resultado['coords']
        valido, motivo = validar_cidade_reversa(coords, cidade_esperada)
        if valido:
            resultado['provider'] = provider
            return resultado
        else:
            return {'error': motivo}

    # Se ambos falharem ou for inválido
    return {'error': "Não encontrado após fallback"}

def processar_endereco(endereco, cidade, cache):
    endereco_enriquecido = enriquecer_endereco(endereco, cidade)
    if is_coordenada(endereco):
        coords = extrair_coordenada(endereco)
        return (endereco, coords, 'coordenada', None, endereco) if coords else (endereco, None, 'erro', 'Coordenada inválida', None)

    with CACHE_LOCK:
        if endereco in cache:
            return (endereco, tuple(cache[endereco]['coords']), f"cache ({cache[endereco].get('provider', '?')})", None, cache[endereco].get('address', 'Endereço do Cache'))

    resultado = geocodificar_endereco(endereco_enriquecido, cidade_esperada=cidade)
    if resultado and 'coords' in resultado:
        with CACHE_LOCK:
            cache[endereco] = resultado
        salvar_cache(cache)
        return (endereco, resultado['coords'], f"geocodificado ({resultado.get('provider', '?')})", None, resultado.get('address', 'Endereço Encontrado'))

    return (endereco, None, 'erro', resultado.get('error', 'Erro desconhecido') if resultado else 'Erro', None)
