import numpy as np
import requests
from tqdm import tqdm
from geopy.distance import geodesic
from utils.console import print_colorido, Fore

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

def calcular_distancia_rua(coords1, coords2):
    try:
        lat1, lon1 = float(coords1[0]), float(coords1[1])
        lat2, lon2 = float(coords2[0]), float(coords2[1])
        dist = geodesic((lat1, lon1), (lat2, lon2)).kilometers
        return dist if dist <= 500 else float('inf')
    except:
        return float('inf')

def calcular_matriz_distancia_osrm(coordenadas):
    n = len(coordenadas)
    dist_matrix = np.full((n, n), float('inf'))
    dur_matrix = np.full((n, n), float('inf'))

    try:
        print_colorido("Buscando distâncias reais nas rodovias (OSRM)...", Fore.CYAN)
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coordenadas])
        url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 'Ok':
            distances = data['distances']
            durations = data['durations']
            for i in range(n):
                for j in range(n):
                    if distances[i][j] is not None:
                        dist_matrix[i][j] = distances[i][j] / 1000.0  # km
                        dur_matrix[i][j] = durations[i][j] / 60.0     # minutos
            print_colorido("✅ Matriz OSRM gerada com precisão!", Fore.GREEN)
            return dist_matrix, dur_matrix
    except Exception as e:
        print_colorido(f"⚠️ OSRM indisponível ({e}). Usando fallback (Distância Geodésica)...", Fore.YELLOW)

    with tqdm(total=n*n, desc="Calculando Matriz Geodésica") as pbar:
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0
                    dur_matrix[i][j] = 0
                else:
                    dist = calcular_distancia_rua(coordenadas[i], coordenadas[j]) * 1.4
                    dist_matrix[i][j] = dist
                    if dist != float('inf'):
                        dur_matrix[i][j] = (dist / 40.0) * 60.0
                pbar.update(1)
    return dist_matrix, dur_matrix

def identificar_outliers(dist_matrix, enderecos_validos, limite_desvio=2.5, p80_limite_km=300):
    n = len(dist_matrix)
    if n <= 1: return [], []

    distancias = [dist_matrix[i][j] for i in range(n) for j in range(n) if i != j and dist_matrix[i][j] != float('inf')]
    if not distancias: return [], []

    media = sum(distancias) / len(distancias)
    desvio = (sum((x - media) ** 2 for x in distancias) / len(distancias)) ** 0.5
    outliers, pontos_principais = [], []

    for i in range(n):
        dist_ponto = [dist_matrix[i][j] for j in range(n) if i != j and dist_matrix[i][j] != float('inf')]
        if not dist_ponto: continue

        media_ponto = sum(dist_ponto) / len(dist_ponto)
        p80 = sorted(dist_ponto)[max(0, min(len(dist_ponto) - 1, int(0.8 * (len(dist_ponto) - 1))))]

        if (media_ponto > media + (limite_desvio * desvio)) or (p80 > p80_limite_km):
            outliers.append(i)
            print_colorido(f"Ponto identificado como outlier: {enderecos_validos[i]}", Fore.YELLOW)
        else:
            pontos_principais.append(i)
    return pontos_principais, outliers

def encontrar_melhor_rota_ortools(dist_matrix):
    if not ORTOOLS_AVAILABLE:
        return None
    n = len(dist_matrix)
    if n <= 1: return [0]

    manager = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [n])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == n or to_node == n: return 0
        return int(dist_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5

    print_colorido("\n🗺️ Otimizando a rota matematicamente com Google OR-Tools...", Fore.CYAN)
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        rota = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < n: rota.append(node)
            index = solution.Value(routing.NextVar(index))
        print_colorido("✅ Rota otimizada com sucesso!", Fore.GREEN)
        return rota
    else:
        return None

def encontrar_melhor_rota_2opt(dist_matrix):
    n = len(dist_matrix)
    if n <= 1: return [0]
    print_colorido("\n🗺️ Otimizando a rota com algoritmo 2-opt (Fallback)...", Fore.YELLOW)
    ponto_atual = 0
    rota = [ponto_atual]
    nao_visitados = set(range(n)); nao_visitados.remove(ponto_atual)

    while nao_visitados:
        proximo = min(nao_visitados, key=lambda ponto: dist_matrix[ponto_atual][ponto])
        rota.append(proximo)
        nao_visitados.remove(proximo)
        ponto_atual = proximo

    melhorou = True
    tentativas = 0
    while melhorou and tentativas < 1000:
        melhorou = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if dist_matrix[rota[i-1]][rota[j]] + dist_matrix[rota[i]][rota[j+1]] < dist_matrix[rota[i-1]][rota[i]] + dist_matrix[rota[j]][rota[j+1]]:
                    rota[i:j+1] = reversed(rota[i:j+1])
                    melhorou = True
                    break
            if melhorou: break
        tentativas += 1
    return rota
