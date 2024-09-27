from django.urls import path 
from .views import post,HomeView

urlpatterns =[
      path('', HomeView.as_view(), name='home'),
      path('predict', post),
]