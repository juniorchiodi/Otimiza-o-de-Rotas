import os
import json
import time
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from utils.console import print_colorido, Fore
from utils.formatadores import enriquecer_endereco, is_coordenada, extrair_coordenada

geolocator = Nominatim(user_agent="juninho.junirj@gmail.com")

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
