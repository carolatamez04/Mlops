from django.urls import path
from .views import predict_sales, index 

urlpatterns = [
    path('', index, name='index'),  
    path('predict/', predict_sales, name='predict_sales'),
]
