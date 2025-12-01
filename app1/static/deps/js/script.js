// var dropdown = document.getElementById('myDropdown');
// var btn = document.getElementById('myBtn');

// btn.onclick = function() {
//   dropdown.style.display = 'block';
// }

// window.onclick = function(event) {
//   if (event.target !== btn && event.target !== dropdown) {
//     dropdown.style.display = 'none';
//   }
// }



// document.addEventListener("DOMContentLoaded", function() {
//     const form = document.getElementById('bookingForm');
//     const bookedText = document.getElementById('id_booked');

//     form.addEventListener('submit', function(event) {
//         event.preventDefault(); // Зупинити стандартну відправку форми

//         // Перевірка валідності форми
//         if (form.checkValidity() === true) {
//             bookedText.style.display = 'block';
//         } else {
//             form.reportValidity(); // Відобразити повідомлення про необхідність заповнення полів
//         }
//     });
// });


// $(document).ready(function() {
//     $('#booking-form').submit(function(event) {
//         event.preventDefault(); // Зупиняємо стандартну поведінку форми

//         // Відправляємо AJAX-запит
//         $.ajax({
//             type: 'POST',
//             url: $(this).attr('action'),
//             data: $(this).serialize(),
//             success: function(response) {
//                 // Якщо запит успішний, оновлюємо сторінку
//                 location.reload();
//             },
//             error: function(xhr, errmsg, err) {
//                 // Обробляємо помилку (якщо необхідно)
//                 console.log(xhr.status + ': ' + xhr.responseText); // Виводимо помилку в консоль
//             }
//         });
//     });
// });

// $(document).ready(function() {
//     $('.booked-button').click(function(event) {
//       var bookingId = $(this).data('id');
//       $.ajax({
//         url: '/booking/update/' + bookingId + '/',
//         data: {
//           booked_booking: true
//         },
//         method: 'POST',
//         success: function(response) {
//           // Оновіть інтерфейс користувача, щоб відобразити зміни
//         },
//         error: function(error) {
//           console.error('Помилка оновлення бронювання:', error);
//         }
//       });
//     });
//   });

// $(document).ready(function() {
//     $('#bookingForm').submit(function(e) {
//       e.preventDefault(); // Запобігти надсиланню форми стандартним способом
  
//       $.ajax({
//         type: 'POST',
//         url: '{% url "users:profile" product_id %}', // URL для оновлення booked_booking
//         data: {
//           csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
//           // ... (інші дані форми)
//         },
//         success: function(response) {
//           // Оновіть значення booked_booking на front-endі
//           $('#booked_booking').val(true);
  
//           // Додайте код для обробки успішного відповіді 
//           // (наприклад, оновлення UI)
//         },
//         error: function(xhr, errmsg, err) {
//           // Додайте код для обробки помилки
//         }
//       });
//     });
//   });
  