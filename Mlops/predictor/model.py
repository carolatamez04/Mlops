import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Datos simulados con más variabilidad
data = {
    'mes': list(range(1, 25)), 
    'ventas': [100, 120, 130, 150, 160, 180, 200, 210, 230, 250, 270, 300,
               320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540]
}
df = pd.DataFrame(data)

# Separar datos en entrenamiento y prueba
X = df[['mes']]
y = df['ventas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar modelo
predictions = model.predict(X_test)
print("Error Medio Absoluto:", mean_absolute_error(y_test, predictions))

# Guardar modelo entrenado
with open("predictor_model.pkl", "wb") as file:
    pickle.dump(model, file)

