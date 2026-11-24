"""Optimización de Red — app web (Streamlit).

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from red_core import (
    ENLACES_EJEMPLO, EQUIPOS_EJEMPLO, INF, a_dataframe, cargar_csv, costo_ruta,
    dibujar_red, floyd_warshall, matriz_desde_enlaces, normalizar, parsear_ruta,
    potencia_adyacencia, potencia_minsuma, reconstruir_ruta, reduccion_porcentual, validar,
)

st.set_page_config(page_title="Optimización de Red", layout="wide")
st.title("Optimización de Red")
st.caption("La red de la pyme como matriz de retardos (ms): se calcula la ruta más rápida entre dos equipos.")

# ---------------------------------------------------------------- entrada
with st.sidebar:
    st.header("Datos de la red")
    modo = st.radio("Origen de los datos", ["Ejemplo (4 equipos)", "Armar manualmente", "Cargar CSV"])

equipos, M0, clave = EQUIPOS_EJEMPLO, matriz_desde_enlaces(EQUIPOS_EJEMPLO, ENLACES_EJEMPLO), "ejemplo"

if modo == "Armar manualmente":
    n = st.sidebar.number_input("Cantidad de equipos", min_value=2, max_value=10, value=5, step=1)
    equipos = [chr(65 + i) for i in range(int(n))]
    M0 = np.full((int(n), int(n)), INF)
    np.fill_diagonal(M0, 0)
    clave = f"manual{n}"
elif modo == "Cargar CSV":
    archivo = st.sidebar.file_uploader("Matriz de retardos (.csv)", type="csv")
    if archivo is None:
        st.info("Sube un CSV con la matriz de retardos (primera fila y columna = nombres de los equipos; "
                "celda vacía = sin conexión). Mientras tanto se muestra el ejemplo.")
    else:
        try:
            equipos, M0 = cargar_csv(archivo)
            clave = f"csv{archivo.name}{len(equipos)}"
        except Exception as e:  # CSV mal formado
            st.error(f"No se pudo leer el CSV: {e}")

st.subheader("1. Matriz de retardos (ms)")
st.caption("Edita las celdas. Celda vacía = sin conexión. La red es simétrica: basta llenar un lado.")
df0 = pd.DataFrame(np.where(np.isinf(M0), np.nan, M0), index=equipos, columns=equipos)
df_editado = st.data_editor(df0, key=f"editor_{clave}")
M = normalizar(df_editado.to_numpy(dtype=float))

errores = validar(M)
if errores:
    for e in errores:
        st.error(e)
    st.stop()

# ---------------------------------------------------------------- cálculo
D, sig = floyd_warshall(M)

c1, c2 = st.columns(2)
origen = c1.selectbox("Origen", equipos, index=0)
destino = c2.selectbox("Destino", equipos, index=len(equipos) - 1)
i, j = equipos.index(origen), equipos.index(destino)
ruta = reconstruir_ruta(sig, i, j)

tab1, tab2, tab3 = st.tabs(["Ruta óptima", "Matrices", "Comparación con la ruta actual"])

with tab1:
    if ruta is None:
        st.warning(f"No existe camino entre {origen} y {destino}.")
    else:
        st.success(f"Ruta más rápida: {' → '.join(equipos[k] for k in ruta)}  |  retardo total: {D[i, j]:g} ms")
    st.pyplot(dibujar_red(M, equipos, ruta))

with tab2:
    st.markdown("**Matriz de retardos M** (∞ = sin conexión directa)")
    st.dataframe(a_dataframe(M, equipos))
    saltos = st.slider("Máximo de conexiones (saltos)", 1, max(len(equipos) - 1, 1), min(2, len(equipos) - 1))
    st.markdown(f"**Mejor retardo con hasta {saltos} conexiones** (producto mín-suma, M^{saltos})")
    st.dataframe(a_dataframe(potencia_minsuma(M, saltos), equipos))
    st.markdown(f"**Cantidad de recorridos de longitud {saltos}** (potencia de la matriz de adyacencia)")
    st.dataframe(pd.DataFrame(potencia_adyacencia(M, saltos), index=equipos, columns=equipos))
    st.markdown("**Retardos mínimos entre todos los pares** (Floyd-Warshall)")
    st.dataframe(a_dataframe(D, equipos))

with tab3:
    st.markdown("Ingresa la ruta que usa hoy la red y compárala con la ruta calculada.")
    defecto = "A-B-D" if clave == "ejemplo" else ""
    texto = st.text_input("Ruta actual (ejemplo: A-B-D)", value=defecto)
    if texto and ruta is not None:
        try:
            actual = parsear_ruta(texto, equipos)
            ms_actual = costo_ruta(M, actual)
            ms_nueva = D[i, j]
            if np.isinf(ms_actual):
                st.error("La ruta actual usa un enlace que no existe en la matriz.")
            elif ms_actual == 0:
                st.warning("La ruta actual tiene retardo 0 ms; no se puede calcular el porcentaje.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Ruta actual", f"{ms_actual:g} ms")
                m2.metric("Ruta propuesta", f"{ms_nueva:g} ms")
                m3.metric("Reducción estimada", f"{reduccion_porcentual(ms_actual, ms_nueva):.1f} %")
                st.latex(rf"\frac{{{ms_actual:g} - {ms_nueva:g}}}{{{ms_actual:g}}} \times 100 = "
                         rf"{reduccion_porcentual(ms_actual, ms_nueva):.1f}\%")
                st.caption("Es una estimación del modelo. La mejora real se comprueba midiendo en la red de la pyme.")
        except ValueError as e:
            st.error(str(e))
    with st.expander("¿Cómo obtener los retardos reales de la pyme?"):
        st.markdown(
            "1. Desde un equipo, medir cada enlace con `ping <ip-del-otro-equipo>` (Windows: `ping -n 20 <ip>`).\n"
            "2. Anotar el **promedio** en ms que entrega ping.\n"
            "3. Repetir en distintos horarios (con la red cargada y sin carga).\n"
            "4. Llenar la matriz (o subir un CSV) y comparar el retardo de la ruta actual con el de la propuesta.\n"
            "5. Tras aplicar la nueva ruta, volver a medir: ahí se comprueba la mejora real."
        )
