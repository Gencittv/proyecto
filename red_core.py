"""Núcleo matemático del proyecto "Optimización de Red" (MTFM01).

La red se modela como una matriz N×N de retardos en milisegundos:
  - M[i][j] = retardo del enlace entre el equipo i y el equipo j
  - M[i][j] = inf  -> no hay conexión directa
  - M[i][i] = 0
"""
import re

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

INF = np.inf

# Ejemplo de la Etapa 2 (4 equipos)
EQUIPOS_EJEMPLO = ["A", "B", "C", "D"]
ENLACES_EJEMPLO = [("A", "B", 10), ("A", "C", 25), ("B", "C", 8), ("B", "D", 30), ("C", "D", 12)]


def matriz_desde_enlaces(equipos, enlaces):
    """Arma la matriz de retardos a partir de una lista (a, b, ms)."""
    n = len(equipos)
    idx = {e: i for i, e in enumerate(equipos)}
    M = np.full((n, n), INF)
    np.fill_diagonal(M, 0)
    for a, b, ms in enlaces:
        M[idx[a], idx[b]] = ms
        M[idx[b], idx[a]] = ms
    return M


def normalizar(M):
    """Deja la matriz lista para calcular: simétrica, sin vacíos (NaN -> inf) y diagonal 0."""
    M = np.array(M, dtype=float)
    M = np.fmin(M, M.T)          # si solo un lado tiene dato, se usa ese
    M = np.where(np.isnan(M), INF, M)
    np.fill_diagonal(M, 0)
    return M


def validar(M):
    """Devuelve una lista de mensajes de error (vacía si la matriz es válida)."""
    errores = []
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        errores.append("La matriz debe ser cuadrada (N×N).")
    elif np.any(M < 0):
        errores.append("Hay retardos negativos; los retardos deben ser ≥ 0 ms.")
    return errores


def producto_minsuma(A, B):
    """Producto matricial 'mín-suma': en vez de sumar de multiplicar, se suman
    retardos y se elige el menor.  C[i][j] = min_k (A[i][k] + B[k][j])."""
    n = A.shape[0]
    C = np.full((n, n), INF)
    for k in range(n):
        C = np.minimum(C, A[:, k:k + 1] + B[k:k + 1, :])
    return C


def potencia_minsuma(M, saltos):
    """Mejor retardo usando HASTA `saltos` conexiones (M^saltos en mín-suma)."""
    R = M.copy()
    for _ in range(saltos - 1):
        R = producto_minsuma(R, M)
    return R


def potencia_adyacencia(M, saltos):
    """Potencia usual de la matriz de adyacencia (0/1): cuenta recorridos de
    longitud exacta `saltos` entre cada par de equipos."""
    A = (np.isfinite(M) & (M > 0)).astype(int)
    return np.linalg.matrix_power(A, saltos)


def floyd_warshall(M):
    """Retardos mínimos entre TODOS los pares + matriz para reconstruir rutas."""
    n = M.shape[0]
    D = M.copy()
    sig = np.where(np.isfinite(M), np.arange(n)[None, :], -1)
    for k in range(n):
        via = D[:, k:k + 1] + D[k:k + 1, :]
        mejora = via < D
        D = np.where(mejora, via, D)
        sig = np.where(mejora, sig[:, k:k + 1], sig)
    return D, sig


def reconstruir_ruta(sig, i, j):
    """Lista de índices del camino más rápido de i a j (None si no hay camino)."""
    if sig[i, j] == -1:
        return None
    ruta = [i]
    while i != j:
        i = int(sig[i, j])
        ruta.append(i)
    return ruta


def costo_ruta(M, ruta):
    """Retardo total (ms) de una ruta dada como lista de índices."""
    return float(sum(M[a, b] for a, b in zip(ruta, ruta[1:])))


def reduccion_porcentual(actual, nueva):
    """(actual - nueva) / actual × 100"""
    return (actual - nueva) / actual * 100


def parsear_ruta(texto, equipos):
    """'A-B-D' -> [0, 1, 3].  Lanza ValueError si hay un equipo desconocido."""
    nombres = [t for t in re.split(r"[-,>→\s]+", texto.strip()) if t]
    if len(nombres) < 2:
        raise ValueError("Escribe al menos dos equipos, por ejemplo A-B-D.")
    idx = {e: i for i, e in enumerate(equipos)}
    for n in nombres:
        if n not in idx:
            raise ValueError(f"El equipo '{n}' no existe en la red.")
    return [idx[n] for n in nombres]


def cargar_csv(origen):
    """Lee una matriz de retardos desde CSV (ruta o archivo). Celdas vacías = sin conexión."""
    df = pd.read_csv(origen, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    equipos = [str(c) for c in df.columns]
    return equipos, normalizar(df.to_numpy())


def formatear(M):
    """Matriz de texto para mostrar (∞ cuando no hay conexión)."""
    return np.vectorize(lambda x: "∞" if np.isinf(x) else f"{x:g}")(M)


def a_dataframe(M, equipos):
    return pd.DataFrame(formatear(M), index=equipos, columns=equipos)


def dibujar_red(M, equipos, ruta=None):
    """Dibuja la red como grafo y resalta la ruta óptima. Devuelve una Figure."""
    n = len(equipos)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(M[i, j]):
                G.add_edge(i, j, w=M[i, j])
    pos = nx.spring_layout(G, seed=4)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=1.5)
    if ruta:
        aristas = list(zip(ruta, ruta[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=aristas, ax=ax, edge_color="#d93025", width=4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1a73e8", node_size=900)
    nx.draw_networkx_labels(G, pos, labels={i: equipos[i] for i in range(n)},
                            font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(a, b): f"{d['w']:g} ms" for a, b, d in G.edges(data=True)},
        ax=ax, font_size=9)
    ax.set_axis_off()
    fig.tight_layout()
    return fig
