import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

#Cargar Archivo
df1 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Cordoba1")
df2 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Guajira2")
df3 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Guajira3")
df4 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Guajira4")
df5 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Antioquia5")
df6 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Antioquia6")
df7 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Atlantico7")
df8 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Atlantico8")
df9 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Magdalena9")
df10 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Magdalena10")
df11 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Cesar11")
df12 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Cesar12")
df13 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Bolivar13")
df14 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Bolivar14")
df15 = pd.read_excel("Resultados/Datos SOLO con irrigacion.xlsx", sheet_name = "Resumen_Choco15")
# Lista Df
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
# Unirlos en uno solo
df = pd.concat(dfs, ignore_index=True)

# Dividir en X e y
X = df.drop("Toneladas por hectaria", axis=1)
y = df["Toneladas por hectaria"]
# Train-test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalizacion
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(train_X.to_numpy()) 

# Crear modelo
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(512, activation='elu', input_shape=(train_X.shape[1],)),
    tf.keras.layers.Dense(1)  # Capa de salida para regresión
])

# Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

# Entrenar modelo
model.fit(train_X, train_y, validation_split=0.2, epochs=50, verbose=0)

# Predecir
y_pred = model.predict(test_X).flatten()

# Evaluar
mse = mean_squared_error(test_y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)
# Calcular R² ajustado
n = test_X.shape[0]
k = test_X.shape[1]
r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

# Imprimir resultados
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.6f}")
print(f"R² ajustado: {r2_adj:.6f}")