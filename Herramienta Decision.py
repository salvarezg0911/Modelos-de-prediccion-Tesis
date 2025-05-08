import dash
from dash import dcc, html
import pickle
import numpy as np
from dash.dependencies import Input, Output

# Inicializar la app
app = dash.Dash(__name__)
server = app.server  # Para desplegar en AWS

# Cargar modelos previamente entrenados
modelo_con_irrigacion = pickle.load(open('models/modelo_riego.pkl', 'rb'))
modelo_sin_irrigacion = pickle.load(open('models/modelo_sin_riego.pkl', 'rb'))

# Layout
app.layout = html.Div([
    html.H1("Herramienta de Decisión - Productividad de Yuca"),

    html.Label("Escenario"),
    dcc.Dropdown(
        id='escenario',
        options=[
            {'label': 'Con Irrigación', 'value': 'riego'},
            {'label': 'Sin Irrigación', 'value': 'no_riego'}
        ],
        value='riego'
    ),

    html.Br(),
    html.Label("Temperatura promedio (°C)"),
    dcc.Input(id='temp', type='number', value=26, step=0.1),

    html.Label("Precipitación acumulada (mm)"),
    dcc.Input(id='precip', type='number', value=800, step=10),

    html.Label("Radiación solar promedio (MJ/m²/día)"),
    dcc.Input(id='rad', type='number', value=18, step=0.1),

    html.Br(),
    html.Button('Predecir Producción', id='boton_pred', n_clicks=0),

    html.Div(id='resultado')
])

# Callback
@app.callback(
    Output('resultado', 'children'),
    Input('boton_pred', 'n_clicks'),
    Input('escenario', 'value'),
    Input('temp', 'value'),
    Input('precip', 'value'),
    Input('rad', 'value')
)
def predecir(n_clicks, escenario, temp, precip, rad):
    if n_clicks == 0:
        return ""

    X = np.array([[temp, precip, rad]])

    if escenario == 'riego':
        pred = modelo_con_irrigacion.predict(X)[0]
    else:
        pred = modelo_sin_irrigacion.predict(X)[0]

    return html.H3(f"Producción estimada: {pred:.2f} ton/ha")

# Ejecutar
if __name__ == '__main__':
    app.run_server(debug=True)
