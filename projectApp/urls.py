from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),
    path('analytics/', views.analytics, name='analytics'),
    path('books/', views.books, name='books'),
    path('about/', views.about, name='about'),
    path('get_stock_analysis/', views.get_stock_analytics, name='get_stock_analysis'),
]
