import json
from utils.console import print_colorido, Fore

def carregar_config():
    try:
        with open("config.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print_colorido("❌ Erro: 'config.json' não encontrado.", Fore.RED)
        exit(1)
