"""Optimización de Red — versión de consola (no requiere Streamlit).

Ejemplos:
  python main.py                              # usa el ejemplo de 4 equipos
  python main.py --origen A --destino D --actual A-B-D
  python main.py --csv ejemplo_retardos.csv --origen A --destino D
"""
import argparse

import numpy as np

from red_core import (
    ENLACES_EJEMPLO, EQUIPOS_EJEMPLO, a_dataframe, cargar_csv, costo_ruta, dibujar_red,
    floyd_warshall, matriz_desde_enlaces, parsear_ruta, potencia_minsuma, reconstruir_ruta,
    reduccion_porcentual, validar,
)


def main():
    p = argparse.ArgumentParser(description="Ruta más rápida en una red usando matrices de retardos.")
    p.add_argument("--csv", help="CSV con la matriz de retardos (vacío = sin conexión)")
    p.add_argument("--origen", default=None)
    p.add_argument("--destino", default=None)
    p.add_argument("--actual", help="ruta que se usa hoy, ej: A-B-D")
    p.add_argument("--salida", default="grafo.png", help="imagen del grafo (por defecto grafo.png)")
    a = p.parse_args()

    if a.csv:
        equipos, M = cargar_csv(a.csv)
    else:
        equipos = EQUIPOS_EJEMPLO
        M = matriz_desde_enlaces(equipos, ENLACES_EJEMPLO)
    errores = validar(M)
    if errores:
        raise SystemExit("\n".join(errores))

    origen = a.origen or equipos[0]
    destino = a.destino or equipos[-1]
    for nombre in (origen, destino):
        if nombre not in equipos:
            raise SystemExit(f"El equipo '{nombre}' no existe. Equipos: {', '.join(equipos)}")
    i, j = equipos.index(origen), equipos.index(destino)

    print("Matriz de retardos M (ms, ∞ = sin conexión):")
    print(a_dataframe(M, equipos).to_string(), "\n")
    for k in range(2, min(4, len(equipos)) + 1):
        print(f"Mejor retardo con hasta {k} conexiones (M^{k}, mín-suma):")
        print(a_dataframe(potencia_minsuma(M, k), equipos).to_string(), "\n")

    D, sig = floyd_warshall(M)
    print("Retardos mínimos entre todos los pares (Floyd-Warshall):")
    print(a_dataframe(D, equipos).to_string(), "\n")

    ruta = reconstruir_ruta(sig, i, j)
    if ruta is None:
        print(f"No existe camino entre {origen} y {destino}.")
    else:
        print(f"Ruta más rápida {origen} -> {destino}: {' -> '.join(equipos[k] for k in ruta)} ({D[i, j]:g} ms)")
        if a.actual:
            actual = parsear_ruta(a.actual, equipos)
            ms = costo_ruta(M, actual)
            if np.isinf(ms) or ms == 0:
                print("La ruta actual no es válida para comparar.")
            else:
                r = reduccion_porcentual(ms, D[i, j])
                print(f"Ruta actual {a.actual}: {ms:g} ms -> reducción estimada: "
                      f"({ms:g} - {D[i, j]:g}) / {ms:g} x 100 = {r:.1f} %  (estimación del modelo)")

    dibujar_red(M, equipos, ruta).savefig(a.salida, dpi=130)
    print(f"\nGrafo guardado en {a.salida}")


if __name__ == "__main__":
    main()
