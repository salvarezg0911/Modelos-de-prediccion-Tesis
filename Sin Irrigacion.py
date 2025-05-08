import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Hiperparámetros fijos (mejores encontrados)
activation = 'elu'
optimizer_name = 'sgd'
learning_rate = 0.005
loss = 'mean_squared_error'
num_layers = 1
units = 512

# Inicializar resultados
results = []

# Definir modelo
model = tf.keras.Sequential()
model.add(norm_layer)

for _ in range(num_layers):
    model.add(tf.keras.layers.Dense(units=units, activation=activation))

model.add(tf.keras.layers.Dense(1))

# Configurar optimizador
if optimizer_name == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif optimizer_name == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
elif optimizer_name == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

# Compilar modelo
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# Entrenar modelo
history = model.fit(train_X, train_y, validation_split=0.2, epochs=50, verbose=0)

# Predecir
y_pred = model.predict(test_X).flatten()

# Controlar NaNs en predicciones
if np.isnan(y_pred).any():
    print("⚠️  Predicción inválida: contiene NaNs.")
else:
    # Calcular métricas
    mse = mean_squared_error(test_y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    # Guardar resultados
    results.append({
        'activation': activation,
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'loss': loss,
        'num_layers': num_layers,
        'units': units,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

    print("✅ Modelo entrenado exitosamente.")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.6f}")