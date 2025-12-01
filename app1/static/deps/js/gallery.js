let slideIndex = 1;
showSlides(slideIndex);

// Next/previous controls
function plusSlides(n) {
  showSlides(slideIndex += n);
}

// Thumbnail image controls
function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  let i;
  let slides = document.getElementsByClassName("mySlides");
  let dots = document.getElementsByClassName("demo");
  let captionText = document.getElementById("caption");
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active";
  captionText.innerHTML = dots[slideIndex-1].alt;
}

///////////////////////////////////////

const form = document.getElementById('form');
const username = document.getElementById('username');
const email = document.getElementById('email');
const phone = document.getElementById('phone'); // Додано посилання на поле телефону

form.addEventListener('submit', e => {
    e.preventDefault();
    
    checkInputs();
});

function checkInputs() {
    // trim to remove the whitespaces
    const usernameValue = username.value.trim();
    const emailValue = email.value.trim();
    const phoneValue = phone.value.trim(); // Додано отримання значення телефону
    
    if(usernameValue === '') {
        setErrorFor(username, 'Ім’я користувача не може бути пустим');
    } else {
        setSuccessFor(username);
    }
    
    if(emailValue === '') {
        setErrorFor(email, 'Поле електронної пошти не може бути пустим');
    } else if (!isEmail(emailValue)) {
        setErrorFor(email, 'Недійсна електронна адреса');
    } else {
        setSuccessFor(email);
    }
    
    if(phoneValue === '') { // Перевірка, чи введено значення для телефону
        setErrorFor(phone, 'Телефон не може бути пустим');
    } else if (!isPhone(phoneValue)) { // Перевірка за допомогою функції isPhone
        setErrorFor(phone, 'Недійсний номер телефону');
    } else {
        setSuccessFor(phone);
    }
}

function setErrorFor(input, message) {
    const formControl = input.parentElement;
    const small = formControl.querySelector('small');
    formControl.className = 'form-control error';
    small.innerText = message;
}

function setSuccessFor(input) {
    const formControl = input.parentElement;
    formControl.className = 'form-control success';
}

function isEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function isPhone(phone) { // Функція для перевірки номера телефону
    return /^\+?3?8?(0\d{9})$/.test(phone);
}

