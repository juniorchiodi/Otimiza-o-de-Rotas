import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
from utils.formatadores import is_coordenada, limpar_endereco

def ler_planilha_excel(arquivo_excel, nome_coluna_nomes, nome_coluna_enderecos):
    df = pd.read_excel(arquivo_excel)
    df = df[df[nome_coluna_nomes].notna() & (df[nome_coluna_nomes] != "")]
    enderecos_brutos = df[nome_coluna_enderecos].dropna().tolist()
    enderecos = [end if is_coordenada(end) else limpar_endereco(end) for end in enderecos_brutos]
    nomes = df[nome_coluna_nomes].fillna("").tolist()
    return nomes, enderecos

def marcar_enderecos_erro_excel(arquivo_excel, enderecos_com_erro):
    try:
        wb = openpyxl.load_workbook(arquivo_excel)
        ws = wb.active
        fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')
        for linha, _, _ in enderecos_com_erro:
            for col in range(1, ws.max_column + 1):
                ws.cell(row=linha + 1, column=col).fill = fill
        wb.save(arquivo_excel)
    except Exception as e:
        pass
