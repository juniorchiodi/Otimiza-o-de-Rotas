import colorama
from colorama import Fore, Style

colorama.init()

def print_colorido(texto, cor=Fore.WHITE, estilo=Style.NORMAL):
    print(f"{estilo}{cor}{texto}{Style.RESET_ALL}")
