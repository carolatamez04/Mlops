from django.shortcuts import render
from django.http import JsonResponse
import pickle
import numpy as np
import pandas as pd

# Cargar modelo
with open("predictor_model.pkl", "rb") as file:
    model = pickle.load(file)

def index(request):
    return render(request, "index.html")

# API para la predicción de ventas
def predict_sales(request):
    mes = int(request.GET.get('mes', 1))

    mes_df = pd.DataFrame([[mes]], columns=['mes'])

    # Hacer predicción
    prediccion = model.predict(mes_df)[0]

    return JsonResponse({'mes': mes, 'prediccion_ventas': prediccion})
