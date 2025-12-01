from django import forms

from goods.models import Products


class ProductForm(forms.ModelForm):
    class Meta:
        model = Products
        fields = '__all__'
    
    # name = forms.CharField()
    # slug = forms.CharField()