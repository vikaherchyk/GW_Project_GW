from django.contrib.auth.decorators import login_required
from audioop import reverse
from django.contrib import auth, messages
from django.shortcuts import redirect, render

from goods.forms import ProductForm
from users.forms import CabinetForm, UserLoginForm, UserRegistrationForm


def login(request):
    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password']
            user = auth.authenticate(username=username, password=password)

            if user:
                auth.login(request, user)
                messages.success(request, f"{username}, Ви увійшли до облікового запису")
                
                if request.POST.get('next', None):
                        return redirect(request.POST.get('next'))
                    
                return redirect('main:index')
    else:
        form = UserLoginForm()

    context = {
        'title': 'Home - Авторизація',
        'form': form
    }
    return render(request, 'users/login.html', context)


def registration(request):
    if request.method == 'POST':
        form = UserRegistrationForm(data=request.POST)
        if form.is_valid():
            form.save()


            user = form.instance
            auth.login(request, user)
            messages.success(request, f"{user.username}, Ви успішно зареєстровані та увійшли до облікового запису")
            return redirect('main:index')
    else:
        form = UserRegistrationForm()
    
    context = {
        'title': 'Home - Реєстрація',
        'form': form
    }
    return render(request, 'users/registration.html', context)

@login_required
def profile(request):
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Оголошення успішно додано")
            return redirect('main:index')
    else:
        form = ProductForm()

    context = {
        'title': 'Home - Додати оголошення',
        'form': form
        }
    return render(request, 'users/profile.html', context)

@login_required
def cabinet(request):
    if request.method == 'POST':
        form = CabinetForm(data=request.POST, instance=request.user, files=request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Профайл успішно оновлено")
            return redirect('user:cabinet')
    else:
        form = CabinetForm(instance=request.user)


    context = {
        'title': 'Home - Профіль',
        'form': form,
    }
    return render(request, 'users/cabinet.html', context)

@login_required
def logout(request):
    messages.success(request, f"{request.user.username}, Ви вийшли з облікового запису")
    auth.logout(request)
    return redirect('main:index')