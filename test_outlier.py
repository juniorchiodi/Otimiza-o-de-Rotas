import numpy as np
from core.routing import identificar_outliers

# Distances:
# Index 0: Ponto de partida (Itapuí) -> ~30km away from Jau points
# Index 1-4: Jau points (very close to each other, ~1-2km)

dist_matrix = np.array([
    [0.0, 30.0, 31.0, 29.0, 32.0],
    [30.0, 0.0, 1.0, 1.5, 2.0],
    [31.0, 1.0, 0.0, 0.8, 1.2],
    [29.0, 1.5, 0.8, 0.0, 1.1],
    [32.0, 2.0, 1.2, 1.1, 0.0]
])

enderecos = [
    "Rua Floriano Peixoto, 368, Centro, Itapuí - SP",
    "Jau Endereco 1",
    "Jau Endereco 2",
    "Jau Endereco 3",
    "Jau Endereco 4",
]

pontos_principais, outliers = identificar_outliers(dist_matrix, enderecos, limite_desvio=2.5, p80_limite_km=300)

print(f"Principais: {pontos_principais}")
print(f"Outliers: {outliers}")
