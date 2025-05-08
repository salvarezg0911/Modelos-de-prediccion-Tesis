import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import tensorflow as tf

# Cargar modelos
modelo_irrigacion = tf.keras.models.load_model("Con Irrigacion.keras")
modelo_sin_irrigacion = tf.keras.models.load_model("Sin Irrigacion.keras")

# Mapa de departamentos (codificación simple)
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

# Layout
app.layout = html.Div([
    html.H2("Predicción de productividad de yuca 🌱"),

    html.Label("Escenario"),
    dcc.Dropdown(
        id='escenario',
        options=[
            {'label': 'Con irrigación', 'value': 'irrigacion'},
            {'label': 'Sin irrigación', 'value': 'no_irrigacion'}
        ],
        value='irrigacion'
    ),

    html.Br(),
    html.Label("Departamento"),
    dcc.Dropdown(
        id='depto',
        options=[{'label': d, 'value': d} for d in departamentos_map.keys()],
        value='Cordoba'
    ),

    html.Br(),
    html.Label("Temperatura promedio (°C)"),
    dcc.Input(id='temp', type='number', step=0.1, value=26),

    html.Label("Precipitación acumulada (mm)"),
    dcc.Input(id='precip', type='number', step=10, value=800),

    html.Label("Radiación solar (MJ/m²/día)"),
    dcc.Input(id='rad', type='number', step=0.1, value=18),

    html.Br(), html.Br(),
    html.Button("Predecir", id="btn_pred", n_clicks=0),

    html.Br(), html.Br(),
    html.Div(id='salida_prediccion')
])

# Callback
@app.callback(
    Output('salida_prediccion', 'children'),
    Input('btn_pred', 'n_clicks'),
    Input('escenario', 'value'),
    Input('depto', 'value'),
    Input('temp', 'value'),
    Input('precip', 'value'),
    Input('rad', 'value')
)
def predecir(n_clicks, escenario, depto, temp, precip, rad):
    if n_clicks == 0:
        return ""

    if None in [temp, precip, rad, depto]:
        return html.Div("⚠️ Ingresa todos los valores antes de predecir.")

    try:
        depto_code = departamentos_map[depto]
        X_input = np.array([[temp, precip, rad, depto_code]])

        if escenario == 'irrigacion':
            pred = modelo_irrigacion.predict(X_input)[0][0]
        else:
            pred = modelo_sin_irrigacion.predict(X_input)[0][0]

        return html.H4(f"🌾 Predicción: {pred:.2f} toneladas por hectárea")

    except Exception as e:
        return html.Div(f"❌ Error en la predicción: {str(e)}")

# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)
