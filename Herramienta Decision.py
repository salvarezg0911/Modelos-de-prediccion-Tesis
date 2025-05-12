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

# Mapa de departamentos codificados
departamentos_map = {
    'Córdoba': 'CORDOBA',
    'Guajira': 'GUAJIRA',
    'Antioquia': 'ANTIOQUIA',
    'Atlántico': 'ATLANTICO',
    'Magdalena': 'MAGDALENA',
    'Cesar': 'CESAR',
    'Bolívar': 'BOLIVAR',
    'Chocó': 'CHOCO'
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
    html.H1("🌽 Predicción de Productividad de Yuca", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("Departamento"),
            dcc.Dropdown(
                id='depto',
                options=[{'label': k, 'value': v} for k, v in departamentos_map.items()],
                value='CORDOBA'
            ),

            html.Label("¿Hay irrigación?"),
            dcc.Dropdown(
                id='irrigacion',
                options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
                value=1
            ),

            html.Label("Año"),
            dcc.Input(id='anio', type='number', value=2024),

            html.Label("Día del año"),
            dcc.Input(id='dia', type='number', value=150),

            html.Label("Irradiación (MJ/m²)"),
            dcc.Input(id='irradiacion', type='number', value=5500, step=1),

            html.Label("Temperatura mínima (°C)"),
            dcc.Input(id='min_temp', type='number', value=22.0, step=0.1),

            html.Label("Temperatura máxima (°C)"),
            dcc.Input(id='max_temp', type='number', value=32.0, step=0.1),

            html.Label("Temperatura promedio (°C)"),
            dcc.Input(id='temp_prom', type='number', value=27.0, step=0.1),

            html.Label("Presión de vapor (hPa)"),
            dcc.Input(id='vapor', type='number', value=35.0, step=0.1),

            html.Label("Velocidad del viento (m/s)"),
            dcc.Input(id='wind', type='number', value=0.5, step=0.01),

            html.Label("Precipitación (mm)"),
            dcc.Input(id='precip', type='number', value=5.2, step=0.01),
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

        html.Div([
            html.Div(id='salida_prediccion'),
            dcc.Graph(id='mapa_departamento')
        ], style={'width': '58%', 'display': 'inline-block', 'padding': '20px'}),
    ])
])


# Callback de predicción
@app.callback(
    [Output('salida_prediccion', 'children'),
     Output('mapa_departamento', 'figure')],
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

        resultado = html.H4(f"🌾 Predicción: {pred:.2f} toneladas por hectárea")
        mapa = generar_mapa(depto)

        return resultado, mapa

    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}"), generar_mapa(depto)


# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)


