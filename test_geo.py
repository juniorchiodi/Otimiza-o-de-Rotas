import sys
from core.geocoding import geocodificar_endereco
from utils.formatadores import enriquecer_endereco, limpar_endereco

cidade = "Jau"
ponto_partida_bruto = "Rua Floriano Peixoto, 368, Centro, Itapuí - SP"
ponto_partida = limpar_endereco(ponto_partida_bruto)
ponto_partida_enriquecido = enriquecer_endereco(ponto_partida, cidade)

print(f"Bruto: {ponto_partida_bruto}")
print(f"Limpo: {ponto_partida}")
print(f"Enriquecido: {ponto_partida_enriquecido}")

res = geocodificar_endereco(ponto_partida_enriquecido, cidade_esperada=cidade)
print(f"Resultado Geocoding: {res}")
