from django.db import models

from django.conf import settings


class Categories(models.Model):
    name = models.CharField(max_length=150, unique=True, verbose_name="Назва")
    slug = models.CharField(
        max_length=200, unique=True, blank=True, null=True, verbose_name="URL"
    )

    class Meta:
        db_table = "category"
        verbose_name = "Категорію"
        verbose_name_plural = "Категорії"

    def __str__(self):
        return self.name


class Products(models.Model):
    name = models.CharField(max_length=150, unique=True, verbose_name="Назва")
    slug = models.CharField(
        max_length=200, unique=True, blank=True, null=True, verbose_name="URL"
    )
    description = models.TextField(blank=True, null=True, verbose_name="Описання")
    image = models.ImageField(
        upload_to="goods_images", blank=True, null=True, verbose_name="Фото"
    )
    price = models.DecimalField(
        default=0.00, max_digits=10, decimal_places=2, verbose_name="Ціна"
    )
    discount = models.DecimalField(
        default=0, max_digits=10, decimal_places=0, verbose_name="Площа"
    )
    quantity = models.CharField(max_length=255, blank=True, verbose_name="Адреса")
    rooms = models.DecimalField(
        default=0, max_digits=10, decimal_places=0, verbose_name="Кімнати"
    )
    contact_person = models.CharField(
        max_length=255, blank=True, verbose_name="Контактні дані"
    )  # Name of the contact person
    email = models.EmailField(
        max_length=254, blank=True, verbose_name="Email"
    )  # Email address
    phone_number = models.CharField(
        max_length=20, blank=True, verbose_name="Номер телефону"
    )  # Phone number
    booked_booking = models.BooleanField(
        max_length=150, default=False, verbose_name="Бронювання"
    )
    # checkin = models.DateField(default='2024-01-01', verbose_name="Дата заїзду")  # Use verbose_name for Ukrainian translation
    # checkout = models.DateField(default='2024-01-01', verbose_name="Дата виїзду")  # Use verbose_name for Ukrainian translation
    category = models.ForeignKey(
        to=Categories, on_delete=models.CASCADE, verbose_name="Категорія"
    )

    class Meta:
        db_table = "product"
        verbose_name = "Оголошення"
        verbose_name_plural = "Оголошення"
        ordering = ("id",)

    def __str__(self):
        return f"{self.name} Кількість - {self.quantity}"

    def display_id(self):
        return f"{self.id:05}"


class Review(models.Model):
    product = models.ForeignKey(
        "Products", on_delete=models.CASCADE, related_name="reviews"
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    text = models.TextField("Текст відгуку")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Відгук від {self.user.username} до {self.product.name}"
