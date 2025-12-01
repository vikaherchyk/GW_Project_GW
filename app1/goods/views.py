from django.contrib import messages
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Q

from goods.models import Products, Review
from goods.utils import q_search

import os, joblib, numpy as np, re, io, base64
from decimal import Decimal
from keras.models import load_model
from keras.utils import pad_sequences
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Глобальні налаштування ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MAX_SEQUENCE_LENGTH = 100

# === Завантаження моделі ===
model = None
tokenizer = None

try:
    model_path = os.path.join(MODEL_DIR, "sentiment_analysis_model.h5")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.joblib")

    tokenizer = joblib.load(tokenizer_path)
    model = load_model(model_path)
    print("✅ Успішно завантажено модель та токенізатор.")

except Exception as e:
    print(f"❌ Помилка завантаження моделі: {e}")


# === Каталог оголошень ===
def catalog(request, category_slug=None):
    goods = (
        Products.objects.all()
        if category_slug == "all"
        else Products.objects.filter(category__slug=category_slug)
    )
    query = request.GET.get("q")
    if query:
        goods = q_search(query)

    # Фільтри
    on_sale = request.GET.get("on_sale")
    rooms = request.GET.get("rooms")
    max_area = request.GET.get("max_area")
    min_price = request.GET.get("min_price")
    max_price = request.GET.get("max_price")
    order_by = request.GET.get("order_by")

    if on_sale:
        goods = goods.filter(booked_booking=False)
    if rooms:
        goods = goods.filter(rooms=rooms)
    if max_area:
        goods = goods.filter(discount__lte=max_area)
    if min_price:
        goods = goods.filter(price__gte=min_price)
    if max_price:
        goods = goods.filter(price__lte=max_price)
    if order_by and order_by != "default":
        goods = goods.order_by(order_by)

    return render(
        request,
        "goods/catalog.html",
        {"title": "Каталог нерухомості", "goods": goods, "slug_url": category_slug},
    )


# === Сторінка продукту ===
def product(request, product_slug):
    product = get_object_or_404(Products, slug=product_slug)
    return render(
        request, "goods/product.html", {"title": "Оголошення", "product": product}
    )


# === Додавання відгуку ===
@login_required
def add_review(request, product_slug):
    product = get_object_or_404(Products, slug=product_slug)
    if request.method == "POST":
        review_text = request.POST.get("review_text")
        if review_text:
            Review.objects.create(
                product=product,
                user=request.user,
                text=review_text,
                created_at=timezone.now(),
            )
            messages.success(request, "✅ Ваш відгук збережено!")
        else:
            messages.warning(request, "⚠️ Поле відгуку не може бути порожнім.")
    return redirect("goods:product", product_slug=product.slug)


# === Прогнозування тональності ===
def _predict_probabilities(padded_sequences):
    raw = model.predict(padded_sequences)
    raw = np.array(raw)
    if raw.ndim == 2 and raw.shape[1] == 2:
        classes = np.argmax(raw, axis=1)
        probs = np.max(raw, axis=1)
    else:
        probs = raw.flatten()
        classes = (probs >= 0.5).astype(int)
    return probs, classes


# === Аналіз відгуків для рекомендацій ===
# === Аналіз відгуків для рекомендацій ===
def sentiment_analysis_view(request):
    recommendations, detailed_predictions = [], []
    user_query, message = "", ""
    chart_base64 = None
    positive_count, negative_count, total_reviews = 0, 0, 0

    if request.method == "POST":
        user_query = request.POST.get("preferences", "").lower().strip()
        if user_query:
            products = Products.objects.all()

            # --- Фільтри кімнат ---
            rooms_map = {
                "1к": 1,
                "2к": 2,
                "3к": 3,
                "4к": 4,
                "однокімнатна": 1,
                "двокімнатна": 2,
                "трикімнатна": 3,
                "чотирикімнатна": 4,
            }
            match_rooms = re.search(r"(\d+)\s*кім", user_query)
            if match_rooms:
                products = products.filter(rooms=int(match_rooms.group(1)))
            else:
                for word, num in rooms_map.items():
                    if word in user_query:
                        products = products.filter(rooms=num)
                        break

            # --- Фільтри ціни ---
            match_price_range = re.search(r"від\s*(\d+)\s*(до|по)\s*(\d+)", user_query)
            if match_price_range:
                products = products.filter(
                    price__gte=Decimal(match_price_range.group(1)),
                    price__lte=Decimal(match_price_range.group(3)),
                )

            # --- Фільтри площі ---
            match_area_min = re.search(r"від\s*(\d+)\s*(м²|кв\.м)", user_query)
            if match_area_min:
                products = products.filter(discount__gte=int(match_area_min.group(1)))
            match_area_max = re.search(r"до\s*(\d+)\s*(м²|кв\.м)", user_query)
            if match_area_max:
                products = products.filter(discount__lte=int(match_area_max.group(1)))

            # --- Пошук по ключових словах ---
            keywords = re.findall(r"\w+", user_query)
            if keywords:
                keyword_filters = Q()
                for kw in keywords:
                    keyword_filters |= (
                        Q(name__icontains=kw)
                        | Q(description__icontains=kw)
                        | Q(quantity__icontains=kw)
                    )
                products = products.filter(keyword_filters)

            # --- Рекомендації ---
            recommendations = products.distinct()[:6]

            # --- Аналіз відгуків всіх рекомендацій ---
            reviews_qs = Review.objects.filter(product__in=recommendations).order_by(
                "created_at"
            )
            texts_seen, unique_texts = [], []
            for r in reviews_qs:
                txt = (r.text or "").strip()
                if txt:
                    norm = re.sub(r"\s+", " ", txt.lower())
                    if norm not in texts_seen:
                        texts_seen.append(norm)
                        unique_texts.append(txt.strip())

            total_reviews = len(unique_texts)

            if total_reviews > 0 and model and tokenizer:
                sequences = tokenizer.texts_to_sequences(unique_texts)
                padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
                probs, classes = _predict_probabilities(padded)
                positive_count = int(np.sum(classes == 1))
                negative_count = int(np.sum(classes == 0))

                detailed_predictions = [
                    (txt, "ПОЗИТИВ" if int(lab) == 1 else "НЕГАТИВ", float(p))
                    for txt, lab, p in zip(unique_texts, classes, probs)
                ]

                # --- Побудова графіка ---
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = [positive_count, negative_count]
                labels = ["Позитивні", "Негативні"]
                colors = ["#28a745", "#dc3545"]  # зелений і червоний
                ax.bar(labels, counts, color=colors)
                for i, v in enumerate(counts):
                    ax.text(i, v + 0.05, str(v), ha="center", fontweight="bold")
                ax.set_title("Аналіз тональності відгуків", fontsize=14)
                ax.set_ylabel("Кількість відгуків")
                ax.set_ylim(0, max(counts + [1]) + 1)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close(fig)
            else:
                message = "ℹ️ Відгуків ще немає для цих рекомендацій."

            if not message:
                message = (
                    "✅ Знайдені рекомендації."
                    if recommendations
                    else "❗ За вашим описом поки немає відповідних варіантів."
                )
        else:
            message = "⚠️ Введіть опис вашого запиту."

    return render(
        request,
        "goods/sentiment_analysis.html",
        {
            "recommendations": recommendations,
            "user_query": user_query,
            "message": message,
            "chart_base64": chart_base64,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_reviews": total_reviews,
            "detailed_predictions": detailed_predictions,
        },
    )


# === Аналіз відгуків для конкретної квартири ===
def analyze_reviews(request, product_slug):
    product = get_object_or_404(Products, slug=product_slug)
    reviews_qs = Review.objects.filter(product=product).order_by("created_at")

    detailed_predictions, chart_base64 = [], None
    positive_count, negative_count, total_reviews = 0, 0, 0
    message = ""

    texts = [r.text.strip() for r in reviews_qs if r.text and r.text.strip()]
    total_reviews = len(texts)

    if total_reviews > 0 and model and tokenizer:
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        probs, classes = _predict_probabilities(padded)
        positive_count = int(np.sum(classes == 1))
        negative_count = int(np.sum(classes == 0))

        detailed_predictions = [
            (txt, "ПОЗИТИВ" if int(lab) == 1 else "НЕГАТИВ", float(p))
            for txt, lab, p in zip(texts, classes, probs)
        ]

        # --- Побудова графіка ---
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = [positive_count, negative_count]
        labels = ["Позитивні", "Негативні"]
        colors = ["#28a745", "#dc3545"]
        ax.bar(labels, counts, color=colors)
        for i, v in enumerate(counts):
            ax.text(i, v + 0.05, str(v), ha="center", fontweight="bold")
        ax.set_title(f"Аналіз відгуків: {product.name}", fontsize=14)
        ax.set_ylabel("Кількість")
        ax.set_ylim(0, max(counts + [1]) + 1)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
    else:
        message = "ℹ️ Для цієї квартири відгуків ще немає."

    return render(
        request,
        "goods/sentiment_analysis.html",
        {
            "product": product,
            "detailed_predictions": detailed_predictions,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_reviews": total_reviews,
            "chart_base64": chart_base64,
            "message": message,
        },
    )
