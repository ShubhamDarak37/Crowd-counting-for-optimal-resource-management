from django.contrib import admin
from django.urls import path, include
from mscnn import views as mscnn_views

urlpatterns = [
	path('',mscnn_views.home, name='home'),


]