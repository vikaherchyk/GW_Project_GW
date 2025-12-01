from django.urls import path, include

from main import views

app_name = "main"

urlpatterns = [
    path("", views.index, name="index"),
    path("catalog/", include("goods.urls")),
]
