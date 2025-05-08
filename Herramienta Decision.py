import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import tensorflow as tf

# Cargar modelos
modelo_irrigacion = tf.keras.models.load_model("Con Irrigacion.keras")
modelo_sin_irrigacion = tf.keras.models.load_model("Sin Irrigacion.keras")

# Mapa de departamentos codificados como enteros
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

    html.Label("Departamento"),
    dcc.Dropdown(
        id='depto',
        options=[{'label': d, 'value': d} for d in departamentos_map.keys()],
        value='Cordoba'
    ),

    html.Label("¿Hay irrigación?"),
    dcc.Dropdown(
        id='irrigacion',
        options=[
            {'label': 'Sí', 'value': 1},
            {'label': 'No', 'value': 0}
        ],
        value=1
    ),

    html.Label("Año"),
    dcc.Input(id='anio', type='number', value=2024),

    html.Label("Día del año"),
    dcc.Input(id='dia', type='number', value=150),

    html.Label("Irradiación (MJ/m²)"),
    dcc.Input(id='irradiacion', type='number', value=18.5, step=0.1),

    html.Label("Temperatura mínima (°C)"),
    dcc.Input(id='min_temp', type='number', value=22, step=0.1),

    html.Label("Temperatura máxima (°C)"),
    dcc.Input(id='max_temp', type='number', value=33, step=0.1),

    html.Label("Temperatura promedio (°C)"),
    dcc.Input(id='temp_prom', type='number', value=27.5, step=0.1),

    html.Label("Presión de vapor (hPa)"),
    dcc.Input(id='vapor', type='number', value=2.3, step=0.1),

    html.Label("Velocidad del viento (m/s)"),
    dcc.Input(id='wind', type='number', value=3.5, step=0.1),

    html.Label("Precipitación (mm)"),
    dcc.Input(id='precip', type='number', value=800, step=10),

    html.Br(), html.Button("Predecir", id="btn_pred", n_clicks=0),

    html.Br(), html.Div(id='salida_prediccion')
])

# Callback
@app.callback(
    Output('salida_prediccion', 'children'),
    Input('btn_pred', 'n_clicks'),
    Input('depto', 'value'),
    Input('irrigacion', 'value'),
    Input('anio', 'value'),
    Input('dia', 'value'),
    Input('irradiacion', 'value'),
    Input('min_temp', 'value'),
    Input('max_temp', 'value'),
    Input('temp_prom', 'value'),
    Input('vapor', 'value'),
    Input('wind', 'value'),
    Input('precip', 'value')
)
def predecir(n_clicks, depto, irrigacion, anio, dia, irradiacion,
             min_temp, max_temp, temp_prom, vapor, wind, precip):

    if n_clicks == 0:
        return ""

    try:
        # Hoja y Station Number fijos (invisibles)
        hoja = 1
        station = 100

        depto_cod = departamentos_map[depto]

        # Construir input en el orden original
        X_input = np.array([[hoja, station, anio, dia, irradiacion,
                             min_temp, max_temp, temp_prom, vapor,
                             wind, precip, depto_cod, irrigacion]])

        # Elegir modelo según irrigación
        if irrigacion == 1:
            pred = modelo_irrigacion.predict(X_input)[0][0]
        else:
            pred = modelo_sin_irrigacion.predict(X_input)[0][0]

        return html.H4(f"🌾 Predicción: {pred:.2f} toneladas por hectárea")

    except Exception as e:
        return html.Div(f"❌ Error en la predicción: {str(e)}")

# Ejecutar app
if __name__ == '__main__':
    app.run_server(debug=True)
