from django.urls import path
from goods import views

app_name = "goods"

urlpatterns = [
    path("search/", views.catalog, name="search"),
    path(
        "analysis/", views.sentiment_analysis_view, name="sentiment_analysis"
    ),  # Аналіз рекомендацій
    path("<slug:category_slug>/", views.catalog, name="index"),
    path("product/<slug:product_slug>/", views.product, name="product"),
    path(
        "product/<slug:product_slug>/add_review/", views.add_review, name="add_review"
    ),
    path(
        "product/<slug:product_slug>/analyze_reviews/",
        views.analyze_reviews,
        name="analyze_reviews",
    ),  # Аналіз відгуків конкретної квартири
]
