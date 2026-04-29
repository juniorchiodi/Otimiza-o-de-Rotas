import json
import os
from utils.console import print_colorido, Fore

def carregar_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print_colorido(f"❌ Erro: '{config_path}' não encontrado.", Fore.RED)
        exit(1)
