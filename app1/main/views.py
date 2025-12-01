from django.http import HttpResponse
from django.shortcuts import render

from goods.models import Categories

def index(request):
    
    
    context = {
        'title': 'Home - Головна',
    }
    
    return render(request, 'main/index.html', context)
