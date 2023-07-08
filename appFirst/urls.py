from django.urls import path
from . import views

urlpatterns = [
    # Otras URLs de tu aplicación...
    path('convertir_imagen/', views.convertir_imagen_endpoint, name="convertir_imagen"),
    path('', views.convertir_imagen_view, name='html-view'),
]