import re
import unicodedata

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

def formatar_minutos(minutos):
    try:
        h, m = int(minutos // 60), int(round(minutos % 60))
        if m == 60: h += 1; m = 0
        return f"{h}h {m:02d}min" if h > 0 else f"{m}min"
    except: return "-"
