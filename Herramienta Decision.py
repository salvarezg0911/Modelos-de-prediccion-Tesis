import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import tensorflow as tf
import json
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State

# Cargar modelos corregidos
modelo_irrigacion = tf.keras.models.load_model("Con Irrigacion.keras")
modelo_sin_irrigacion = tf.keras.models.load_model("Sin Irrigacion.keras")

#Cargar datos para estadísticas
#Cargar Archivo
df1 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cordoba1")
df2 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira2")
df3 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira3")
df4 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Guajira4")
df5 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Antioquia5")
df6 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Antioquia6")
df7 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Atlantico7")
df8 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Atlantico8")
df9 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Magdalena9")
df10 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Magdalena10")
df11 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cesar11")
df12 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Cesar12")
df13 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Bolivar13")
df14 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Bolivar14")
df15 = pd.read_excel("Resultados/Datos sin irrigacion.xlsx", sheet_name = "Resumen_Choco15")
# Lista Df
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
# Unirlos en uno solo
df_estadisticas = pd.concat(dfs, ignore_index=True)
# Eliminar columnas que no deben estar en los predictores
df_estadisticas = df_estadisticas.drop(columns=['Hoja', 'Produccion'])


#Función para mostrar estadísticas
def obtener_estadisticas_departamento(depto):
    codigo = departamento_codigos.get(depto)
    df_depto = df_estadisticas[df_estadisticas['Departamento'] == codigo]

    if df_depto.empty:
        return html.P("No hay datos disponibles para este departamento.")

    # Variables con sus unidades
    columnas_con_unidades = {
        'Temp Promedio': '°C',
        'Precipitacion': 'mm',
        'Irradiacion': 'MJ/m²',
        'Wind Speed': 'm/s',
        'Min Temp': '°C',
        'Max Temp': '°C'
    }

    columnas_interes = list(columnas_con_unidades.keys())
    resumen = df_depto[columnas_interes].agg(['mean', 'min', 'max']).round(2).T.reset_index()
    resumen.columns = ['Variable', 'Media', 'Mínimo', 'Máximo']

    # Añadir unidades a la columna 'Variable'
    resumen['Variable'] = resumen['Variable'].apply(
        lambda var: f"{var} ({columnas_con_unidades.get(var, '')})"
    )

    return html.Div([
        html.H4(f"Estadísticas Históricas de {depto}"),
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in resumen.columns])),
            html.Tbody([
                html.Tr([html.Td(resumen.iloc[i][col]) for col in resumen.columns])
                for i in range(len(resumen))
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'})
    ])


# Mapa de departamentos codificados
departamentos_map = {
    'Córdoba': 'Córdoba',
    'Guajira': 'Guajira',
    'Antioquia': 'Antioquia',
    'Atlántico': 'Atlántico',
    'Magdalena': 'Magdalena',
    'Cesar': 'Cesar',
    'Bolívar': 'Bolívar',
    'Chocó': 'Chocó'
}
departamento_codigos = {
    'Córdoba': 1,
    'Guajira': 2,
    'Antioquia': 3,
    'Atlántico': 4,
    'Magdalena': 5,
    'Cesar': 6,
    'Bolívar': 7,
    'Chocó': 8
}

# Crear app
app = dash.Dash(__name__)
server = app.server

with open("co.json", encoding="utf-8") as f:
    geojson_colombia = json.load(f)
    print("Ejemplo de propiedades de un departamento:")
    print(geojson_colombia["features"][0]["properties"])

def generar_mapa(departamento_seleccionado):
    departamentos = [f["properties"]["name"] for f in geojson_colombia["features"]]

    df = pd.DataFrame({
        "Departamento": departamentos,
        "Seleccionado": [1 if d == departamento_seleccionado else 0 for d in departamentos]
    })

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_colombia,
        locations="Departamento",
        featureidkey="properties.name",
        color="Seleccionado",
        color_continuous_scale=[[0, "lightgray"], [1, "green"]],
        range_color=[0,1],
        mapbox_style="carto-positron",
        zoom=4,  
        center={"lat": 4.5709, "lon": -74.2973},
        opacity=0.6,
        hover_data={"Seleccionado": False, "Departamento": False}
    )

    fig.update_layout(
        coloraxis_showscale=False,  # ❌ Oculta la barra de color
        showlegend=False,           # ❌ Oculta la leyenda
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    return fig


# Layout de la app
app.layout = html.Div([
    html.H1("Predicción de Productividad de Yuca", style={'textAlign': 'center'}),

    # FILA SUPERIOR: Ingreso de datos + Estadísticas
    html.Div([
        # Panel de entrada (izquierda)
        html.Div([
            html.H3("Ingreso de datos", style={"marginBottom": "20px", "color": "#004d40"}),

            html.Label("Departamento", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='depto',
                options=[{'label': k, 'value': v} for k, v in departamentos_map.items()],
                value='Córdoba'
            ),

            html.Label("¿Hay irrigación?", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Dropdown(
                id='irrigacion',
                options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
                value=1
            ),

            *[html.Div([
                html.Label(label, style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.Input(id=id_, type='number', value=val, step=step, style={'width': '100%'})
            ]) for label, id_, val, step in [
                ("Año", 'anio', 2024, 1),
                ("Día del año", 'dia', 150, 1),
                ("Irradiación (MJ/m²)", 'irradiacion', 5500, 1),
                ("Temperatura mínima (°C)", 'min_temp', 22.0, 0.1),
                ("Temperatura máxima (°C)", 'max_temp', 32.0, 0.1),
                ("Temperatura promedio (°C)", 'temp_prom', 27.0, 0.1),
                ("Presión de vapor (hPa)", 'vapor', 35.0, 0.1),
                ("Velocidad del viento (m/s)", 'wind', 0.5, 0.01),
                ("Precipitación (mm)", 'precip', 5.2, 0.01)
            ]]
        ],
        style={
            'padding': '25px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '12px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
            'width': '45%',
            'minWidth': '300px',
        }),

        # Panel de estadísticas + predicción (derecha)
        html.Div([
            html.H3("Estadísticas del departamento", style={"color": "#004d40", "marginBottom": "20px"}),

            html.Div(id='resumen_estadisticas'),

            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),

            html.Div(id='salida_prediccion', style={
                'textAlign': 'center',
                'fontSize': '20px',
                'fontWeight': 'bold',
                'color': '#1b5e20',
                'marginTop': '25px'
            })
        ],
        style={
            'padding': '25px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '12px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
            'width': '45%',
            'minWidth': '300px'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'flexWrap': 'wrap',
        'marginBottom': '30px'
    }),

    # Mapa debajo
    html.Div([
        dcc.Graph(id='mapa_departamento')
    ], style={
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'borderRadius': '12px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
        'width': '90%',
        'maxWidth': '1000px',
        'margin': '40px auto 20px auto'
    })
])






# Callback de predicción
@app.callback(
    [Output('salida_prediccion', 'children'),
     Output('mapa_departamento', 'figure'),
     Output('resumen_estadisticas', 'children')],
    [Input('depto', 'value'),
     Input('irrigacion', 'value'),
     Input('anio', 'value'),
     Input('dia', 'value'),
     Input('irradiacion', 'value'),
     Input('min_temp', 'value'),
     Input('max_temp', 'value'),
     Input('temp_prom', 'value'),
     Input('vapor', 'value'),
     Input('wind', 'value'),
     Input('precip', 'value')]
)
def predecir(depto, irrigacion, anio, dia, irradiacion,
             min_temp, max_temp, temp_prom, vapor, wind, precip):
    try:
        depto_cod = list(departamentos_map.values()).index(depto)

        X_input = np.array([[10, anio, dia, irradiacion,
                             min_temp, max_temp, temp_prom,
                             vapor, wind, precip, depto_cod, irrigacion]])

        modelo = modelo_irrigacion if irrigacion == 1 else modelo_sin_irrigacion
        pred = modelo.predict(X_input)[0][0]

        resultado = html.H4(f"Predicción: {pred:.2f} toneladas por hectárea")
        mapa = generar_mapa(depto)
        resumen = obtener_estadisticas_departamento(depto)

        return resultado, mapa, resumen

    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}"), generar_mapa(depto), obtener_estadisticas_departamento(depto)



# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)


