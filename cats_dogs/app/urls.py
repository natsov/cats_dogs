from django.urls import path
from .views import ClassifyView, main

urlpatterns = [
    path('', main, name='main'),
    path('classifier/classify/', ClassifyView.as_view(), name='classify'),
]