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
    'Cordoba': 0,
    'Guajira': 1,
    'Antioquia': 2,
    'Atlantico': 3,
    'Magdalena': 4,
    'Cesar': 5,
    'Bolivar': 6,
    'Choco': 7
}

# Crear app
app = dash.Dash(__name__)
server = app.server

with open("co.json", encoding="utf-8") as f:
    geojson_colombia = json.load(f)

def generar_mapa(departamento_seleccionado):
    df = pd.DataFrame({
        "Departamento": [feature["properties"]["NOMBRE_DPT"] for feature in geojson_colombia["features"]],
        "Valor": [1 if feature["properties"]["NOMBRE_DPT"] == departamento_seleccionado else 0 for feature in geojson_colombia["features"]]
    })

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_colombia,
        locations="Departamento",
        featureidkey="properties.NOMBRE_DPT",
        color="Valor",
        color_continuous_scale=["lightgray", "green"],
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 4.5709, "lon": -74.2973},
        opacity=0.6
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig


# Layout de la app
app.layout = html.Div([
    html.H1("🌽 Predicción de Productividad de Yuca", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("Departamento"),
            dcc.Dropdown(
                id='depto',
                options=[{'label': d, 'value': d} for d in departamentos_map],
                value='Cordoba'
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

            html.Br(),
            html.Button("Predecir", id="btn_pred", n_clicks=0),
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
    [Input('btn_pred', 'n_clicks')],
    [State('depto', 'value'),
     State('irrigacion', 'value'),
     State('anio', 'value'),
     State('dia', 'value'),
     State('irradiacion', 'value'),
     State('min_temp', 'value'),
     State('max_temp', 'value'),
     State('temp_prom', 'value'),
     State('vapor', 'value'),
     State('wind', 'value'),
     State('precip', 'value')]
)
def predecir(n_clicks, depto, irrigacion, anio, dia, irradiacion,
             min_temp, max_temp, temp_prom, vapor, wind, precip):
    if n_clicks == 0:
        return "", generar_mapa(depto)

    try:
        depto_cod = departamentos_map[depto]

        X_input = np.array([[10, anio, dia, irradiacion,
                             min_temp, max_temp, temp_prom,
                             vapor, wind, precip, depto_cod, irrigacion]])

        modelo = modelo_irrigacion if irrigacion == 1 else modelo_sin_irrigacion
        pred = modelo.predict(X_input)[0][0]

        resultado = html.H4(f"🌾 Predicción: {pred:.2f} toneladas por hectárea")
        mapa = generar_mapa(depto)

        return resultado, mapa

    except Exception as e:
        return html.Div(f"❌ Error en la predicción: {str(e)}"), generar_mapa(depto)


# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)


