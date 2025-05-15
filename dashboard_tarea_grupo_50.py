import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import locale
import plotly.graph_objects as go

# -------------------------- Configuración de Streamlit -----------------

# Configurar el idioma local a español
# locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')  # Para sistemas basados en Unix (Linux/Mac)
locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')  # Para Windows

# -------------------------------------------------------------------------------------

# Cargar los datos
@st.cache_data
def load_data():
    ventasHeader = [
        'ID de Factura', 'Sucursal', 'Ciudad', 'Tipo de Cliente', 'Género', 'Línea de Producto', 'Precio Unitario', 'Cantidad', 'Impuesto 5%', 'Total',
        'Fecha', 'Hora', 'Método de Pago', 'Costo de Bienes Vendidos', 'Porcentaje de Margen Bruto', 'Ingreso Bruto', 'Calificación']
    df_ventas = pd.read_csv(r'C:\Users\Ignacio\Desktop\Streamlit\data.csv', sep=',', header=0, names=ventasHeader)

    return df_ventas

df_ventas = load_data()

# Título
st.title("Dashboard de Ventas")

# -------------------------- Analisis Exploratorio ----------------------
st.subheader("Análisis Exploratorio")

if st.checkbox("Mostrar exploración inicial del DataFrame", key="analisis_exploratorio"):
    st.subheader("Información del DataFrame")
    
    # Usar StringIO como buffer
    buffer = io.StringIO()
    df_ventas.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader("Estadísticas descriptivas")
    st.write(df_ventas.describe())

    st.subheader("Nombres de las columnas")
    st.write(df_ventas.columns.tolist())

# -------------------------- Variables Relevantes ----------------------
st.subheader("Variables Relevantes")

if st.checkbox("Mostrar variables relevantes y su justificación", key="variables_relevantes"):
    # Definir los datos como una lista de tuplas
    datos = [
        ("ID de Factura", "Único identificador de cada transacción, útil para evitar duplicados y rastrear operaciones individuales"),
        ("Sucursal", "Permite analizar el desempeño por ubicación física de la empresa"),
        ("Ciudad", "Ayuda a entender patrones geográficos de ventas y preferencias regionales"),
        ("Tipo de Cliente", "Permite segmentar a los clientes y evaluar diferencias en comportamiento de compra"),
        ("Género", "Útil para análisis demográfico y personalización de estrategias de marketing"),
        ("Línea de Producto", "Es clave para medir el rendimiento de categorías de productos y optimizar inventario"),
        ("Precio Unitario", "Fundamental para calcular ingresos y márgenes por producto"),
        ("Cantidad", "Muestra el volumen vendido, lo que ayuda a identificar productos populares o estacionales"),
        ("Total", "Indicador directo del valor de venta final, esencial para medir ingresos totales"),
        ("Fecha", "Permite realizar análisis temporales: ventas por día, mes, hora pico, etc"),
        ("Hora", "Permite realizar análisis de patrones horarios de compra"),
        ("Método de Pago", "Informa sobre preferencias de pago de los clientes, útil para decisiones operativas y financieras"),
        ("Ingreso Bruto", "Mide la rentabilidad antes de costos adicionales; útil para comparar líneas de producto o sucursales")
    ]

    # Crear el DataFrame
    df_variables_relevantes = pd.DataFrame(datos, columns=["Variable", "Justificación"])

    # Mostrar el DataFrame en Streamlit
    st.write(df_variables_relevantes)

# -------------------------- Preguntas de Negocio ----------------------
st.subheader("Preguntas de Negocio")

if st.checkbox("Mostrar preguntas de negocio", key="preguntas_negocio"):

    # Lista de tuplas: (Variable, Pregunta)
    datos = [
        # 1. ID de Factura
        ("ID de Factura", "¿Existen transacciones duplicadas?"),
        ("ID de Factura", "¿Cómo se distribuyen las ventas por número de factura?"),

        # 2. Sucursal / Ciudad
        ("Sucursal / Ciudad", "¿Cuál sucursal tiene mejor desempeño?"),
        ("Sucursal / Ciudad", "¿Hay diferencias significativas en las preferencias de producto entre ciudades?"),

        # 3. Tipo de Cliente
        ("Tipo de Cliente", "¿Qué tipo de cliente genera más ingresos?"),
        ("Tipo de Cliente", "¿Existen patrones de compra distintos según el tipo de cliente?"),

        # 4. Género
        ("Género", "¿Existe predominio de un género en ciertos tipos de productos?"),
        ("Género", "¿Cómo diseñar estrategias de marketing basadas en género?"),

        # 5. Línea de Producto
        ("Línea de Producto", "¿Qué categorías generan mayores ingresos o margen bruto?"),
        ("Línea de Producto", "¿Qué productos tienen mayor rotación?"),

        # 6. Precio Unitario / Cantidad / Total
        ("Precio Unitario / Cantidad / Total", "¿Cuáles son los productos más vendidos?"),
        ("Precio Unitario / Cantidad / Total", "¿Cómo afecta el precio al volumen de ventas?"),
        ("Precio Unitario / Cantidad / Total", "¿Cuál es el promedio de gasto por cliente?"),

        # 7. Fecha / Hora
        ("Fecha / Hora", "¿En qué días/horas hay mayor volumen de ventas?"),
        ("Fecha / Hora", "¿Hay estacionalidad en ciertas líneas de producto?"),

        # 8. Método de Pago
        ("Método de Pago", "¿Cuál es el método de pago más usado?"),
        ("Método de Pago", "¿Hay relación entre método de pago y monto total de la factura?"),

        # 9. Ingreso Bruto
        ("Ingreso Bruto", "¿Cuál línea de producto genera más ganancias?"),
        ("Ingreso Bruto", "¿Cómo varía la rentabilidad por sucursal?"),

        # 10. Calificación
        ("Calificación", "¿Existe una correlación entre calificación y monto gastado?"),
        ("Calificación", "¿Las sucursales con mejor servicio también tienen mayores ventas?")
    ]

    # Creamos el DataFrame
    df_preguntas_por_variable = pd.DataFrame(datos, columns=["Variable", "Pregunta"])

    # Aplicar estilo: alinear a la izquierda y centrar encabezado
    styled_df = df_preguntas_por_variable.style.set_properties(**{'text-align': 'left'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
    ])

    # Mostrar el DataFrame con estilo
    st.write(styled_df)


# ----------------------------------- DASHABOARD ---------------------------------------
# ---------------- 1. Evolución de Ventas Totales a lo largo del tiempo -----------------

st.subheader("Evolución de ventas totales a lo largo del tiempo")

# Convertir la columna 'Fecha' a tipo datetime
df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'])

# Agrupar por mes
ventas_por_mes = df_ventas.resample('ME', on='Fecha')['Total'].sum().reset_index()

# Crear el gráfico con Plotly
fig = go.Figure()

# Agregar la línea de ventas totales
fig.add_trace(go.Scatter(
    x=ventas_por_mes['Fecha'],
    y=ventas_por_mes['Total'],
    mode='lines+markers',
    marker=dict(color='steelblue'),
    line=dict(color='steelblue'),
    name='Ventas Totales'
))

# Formatear manualmente los valores del eje y
formatted_y_values = [f"{int(y):,}".replace(",", ".") for y in ventas_por_mes['Total']]

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text='Ventas Totales por Mes',
        x=0.5, 
        xanchor='center',
        font=dict(size=20)
    ),
    xaxis=dict(
        title='Mes',
        tickmode='array',
        tickvals=ventas_por_mes['Fecha'],
        ticktext=[fecha.strftime('%b').capitalize() + ' - ' + fecha.strftime('%Y') for fecha in ventas_por_mes['Fecha']], 
        tickangle=45 
    ),
    yaxis=dict(
        title='Total de Ventas',
        ticktext=formatted_y_values,
    ),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 2. Ingresos por Línea de Producto ------------------------------------------
st.subheader("Ingresos por Línea de Producto")

# Crear un filtro interactivo para seleccionar la sucursal
sucursales_disponibles = ["Todas las sucursales"] + sorted(df_ventas['Sucursal'].unique())
sucursal_seleccionada = st.selectbox("Selecciona una sucursal:", options=sucursales_disponibles, index=0)

# Filtrar los datos según la sucursal seleccionada
if sucursal_seleccionada == "Todas las sucursales":
    df_filtrado = df_ventas
else:
    df_filtrado = df_ventas[df_ventas['Sucursal'] == sucursal_seleccionada]

# Agrupar los ingresos por línea de producto
product_totals = df_filtrado.groupby('Línea de Producto')['Total'].sum().reset_index()

# Crear el gráfico de barras con colores diferentes por línea de producto
fig = px.bar(
    product_totals,
    x='Línea de Producto',
    y='Total',
    color='Línea de Producto',
    title=f"Ingresos por Línea de Producto - {sucursal_seleccionada}"
)

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text=f"Ingresos por Línea de Producto - {sucursal_seleccionada}",
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Línea de Producto"
    ),
    yaxis=dict(
        title="Total de Ingresos"
    ),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 3. Distribución de la Calificación de Clientes --------------------------------
st.subheader("Distribución de Calificaciones (Rating)")

# Crear un filtro interactivo para seleccionar la sucursal
sucursales_disponibles = ["Todas las sucursales"] + sorted(df_ventas['Sucursal'].unique())
sucursal_seleccionada = st.selectbox("Selecciona una sucursal para el análisis de calificaciones:", options=sucursales_disponibles, index=0)

# Filtrar los datos según la sucursal seleccionada
if sucursal_seleccionada == "Todas las sucursales":
    df_filtrado = df_ventas
else:
    df_filtrado = df_ventas[df_ventas['Sucursal'] == sucursal_seleccionada]

# Crear el histograma con Plotly
fig = px.histogram(
    df_filtrado,
    x='Calificación',
    nbins=10,
    title=f"Distribución de Calificaciones - {sucursal_seleccionada}",
    labels={'Calificación': 'Calificación', 'count': 'Cantidad'},
    opacity=0.6,
)

# Crear un rango de valores para la curva KDE
x_kde = np.linspace(df_filtrado['Calificación'].min(), df_filtrado['Calificación'].max(), 200)
kde = sns.kdeplot(df_filtrado['Calificación'], bw_adjust=1, fill=False).get_lines()[0].get_data()

# Escalar la curva KDE para que coincida con la escala del histograma
hist_data = np.histogram(df_filtrado['Calificación'], bins=10)
kde_scaled = kde[1] * max(hist_data[0]) / max(kde[1])

# Agregar la curva KDE al gráfico
fig.add_trace(
    go.Scatter(
        x=kde[0],
        y=kde_scaled,
        mode='lines',
        line=dict(color='red', width=2),
        name='Curva KDE'
    )
)

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text=f"Distribución de Calificaciones - {sucursal_seleccionada}",
        x=0.5,
        xanchor='center',
        font=dict(size=20) 
    ),
    xaxis=dict(
        title="Calificación",
        range=[3, 12],
        dtick=1 
    ),
    yaxis=dict(
        title="Cantidad"
    ),
    bargap=0.2,
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40) 
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 4. Comparación del Gasto por Tipo de Cliente --------------------------------
st.subheader("Gasto por Tipo de Cliente")

# Crear el gráfico de caja con Plotly
fig = px.box(
    df_ventas,
    x='Tipo de Cliente',
    y='Total',
    title="Distribución del Gasto Total por Tipo de Cliente",
    labels={'Total': 'Monto Total (Gasto)', 'Tipo de Cliente': 'Tipo de Cliente'}
)

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text="Distribución del Gasto Total por Tipo de Cliente",
        x=0.5, 
        xanchor='center',
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Tipo de Cliente"
    ),
    yaxis=dict(
        title="Monto Total (Gasto)"
    ),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 5. Relación entre Costo y Ganancia Bruta --------------------------------
st.subheader("Relación entre Costo y Ganancia Bruta")

# Crear el gráfico de dispersión con línea de tendencia
fig = px.scatter(
    df_ventas,
    x='Costo de Bienes Vendidos',
    y='Ingreso Bruto',
    trendline='ols',  # Agregar línea de tendencia
    title="Relación entre Costo de Bienes Vendidos e Ingreso Bruto",
    labels={'Costo de Bienes Vendidos': 'Costo de Bienes Vendidos', 'Ingreso Bruto': 'Ingreso Bruto'}
)

# Personalizar los colores de los puntos y la línea de tendencia
fig.update_traces(
    marker=dict(color='blue', size=8),  # Cambiar el color y tamaño de los puntos
    selector=dict(mode='markers')  # Aplicar solo a los puntos
)

# Personalizar la línea de tendencia
for trace in fig.data:
    if trace.mode == 'lines':
        trace.line.color = 'red'
        trace.line.width = 3 

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text="Relación entre Costo de Bienes Vendidos e Ingreso Bruto",
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Costo de Bienes Vendidos"
    ),
    yaxis=dict(
        title="Ingreso Bruto"
    ),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 6. Métodos de Pago Preferidos ---------------------------------------
st.subheader("Métodos de Pago Preferidos")

# Crear un filtro interactivo para seleccionar la sucursal
sucursales_disponibles = ["Todas las sucursales"] + sorted(df_ventas['Sucursal'].unique())
sucursal_seleccionada = st.selectbox(
    "Selecciona una sucursal:", 
    options=sucursales_disponibles, 
    index=0, 
    key="metodos_pago_sucursal"  # Agregar un key único
)

# Filtrar los datos según la sucursal seleccionada
if sucursal_seleccionada == "Todas las sucursales":
    df_filtrado = df_ventas
else:
    df_filtrado = df_ventas[df_ventas['Sucursal'] == sucursal_seleccionada]

# Contar los métodos de pago en los datos filtrados
payment_counts = df_filtrado['Método de Pago'].value_counts().reset_index()
payment_counts.columns = ['Método de Pago', 'Cantidad']

# Traducir los valores de la columna 'Método de Pago'
payment_counts['Método de Pago'] = payment_counts['Método de Pago'].replace({
    'Cash': 'Efectivo',
    'Credit card': 'Tarjeta de Crédito',
    'Ewallet': 'Billetera Electrónica'
})

# Crear el gráfico de pastel con los datos filtrados
fig = px.pie(
    payment_counts,
    values='Cantidad',
    names='Método de Pago',
    title=f"Métodos de Pago Preferidos - {sucursal_seleccionada}"
)

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text=f"Métodos de Pago Preferidos - {sucursal_seleccionada}",
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    )
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)


# ---------------- 7. Correlación Numérica --------------------------------------------
st.subheader("Mapa de Correlación Numérica")

# Definir las columnas numéricas disponibles
numeric_cols = ['Precio Unitario', 'Cantidad', 'Impuesto 5%', 'Total', 'Costo de Bienes Vendidos', 'Ingreso Bruto', 'Calificación']

# Crear un filtro interactivo para seleccionar columnas
selected_cols = st.multiselect(
    "Selecciona las columnas numéricas para el análisis de correlación:",
    options=numeric_cols,
    default=numeric_cols  # Seleccionar todas las columnas por defecto
)

# Verificar si se seleccionaron columnas
if selected_cols:
    # Calcular la matriz de correlación solo con las columnas seleccionadas
    correlation_matrix = df_ventas[selected_cols].corr()

    # Crear el gráfico de calor con Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        labels=dict(color="Correlación")
    )

    # Ajustar el tamaño de la fuente de los valores en las celdas
    fig.update_traces(textfont_size=11)

    # Configurar el diseño del gráfico
    fig.update_layout(
        title=dict(
            text="Mapa de Correlación Numérica",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=12),
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            automargin=True
        ),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        uniformtext_minsize=10,
        uniformtext_mode='show',
        xaxis_scaleanchor="y"
    )

    # Mostrar el gráfico en Streamlit con el ancho del contenedor
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Por favor, selecciona al menos una columna para calcular la correlación.")


# ---------------- 8. Composición del Ingreso Bruto por Sucursal y Línea de Producto ------------------------------
st.subheader("Ingreso Bruto por Sucursal y Línea de Producto")

# Traducir los valores de las columnas
pivot = df_ventas.groupby(['Sucursal', 'Línea de Producto'])['Ingreso Bruto'].sum().reset_index()

# Crear el gráfico de barras con Plotly
fig = px.bar(
    pivot,
    x='Sucursal',
    y='Ingreso Bruto',
    color='Línea de Producto',
    title="Ingreso Bruto por Sucursal y Línea de Producto"
)

# Configurar el diseño del gráfico
fig.update_layout(
    title=dict(
        text="Ingreso Bruto por Sucursal y Línea de Producto",
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Sucursal",
        title_standoff=30 
    ),
    yaxis=dict(
        title="Ingreso Bruto"
    ),
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40),
    legend_title=dict(text="Línea de Producto"),
    legend=dict(
        title_font=dict(size=14),
        font=dict(size=12),
        orientation="h",
        yanchor="bottom",
        y=-0.6,
        xanchor="center",
        x=0.5
    )
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)
