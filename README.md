# Optimización de Red (MTFM01)

Prototipo que modela la red de una pyme como una matriz de retardos (ms) y calcula la ruta más rápida entre dos equipos.

## Instalación (una vez)

    pip install -r requirements.txt

## Ejecutar

App web (la del proyecto):

    streamlit run app.py

Versión de consola (no necesita Streamlit):

    python main.py --origen A --destino D --actual A-B-D

## Archivos

- `red_core.py`: matemática (matriz de retardos, producto mín-suma, Floyd-Warshall, reducción %).
- `app.py`: interfaz Streamlit.
- `main.py`: versión de consola; guarda `grafo.png`.
- `ejemplo_retardos.csv`: el ejemplo de 4 equipos. Formato: primera fila y columna con los nombres; celda vacía = sin conexión.

## Ejemplo (Etapa 2)

Enlaces: A-B 10, A-C 25, B-C 8, B-D 30, C-D 12 ms.
De A a D: hasta 2 conexiones 37 ms (A-C-D); hasta 3 conexiones 30 ms (A-B-C-D).
Ruta actual A-B-D = 40 ms -> reducción (40 − 30) ÷ 40 × 100 = 25 % (estimación del modelo).
