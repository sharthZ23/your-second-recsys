В данной части репозитория представлена простейшая реализация микросервиса, который на запрос отдает рекомендации


### Запуск
    - Устанавливаем докер (https://www.docker.com/get-started)
    - В консоли переходим на тот же уровень, на котором лежит Dockerfile
    - Выполняем docker build --tag <your_tag_here>
    - Запускаем с помощью docker run -p 80:80 (можно замапить иначе, если 80 порт уже чем-то занят)

### Работа с приложением
В main.py реализованы 2 эндопоинта: /api/v1/recommend_for_user, /api/v1/recommend_bruteforce. Первый отдает рекомендации с помощью Annoy, второй вычислением в лоб. Также есть swagger с документацией по адресу /docs, в нем же можно делать простые запросы.

Если вы все делали так, как указано в **запуск**, то достучаться до приложения можно по адресам `http://localhost:80/docs`, `http://localhost:80/api/v1/recommend_for_user`, `http://localhost:80/api/v1/recommend_bruteforce`. Где 80 может быть другим портом в зависимости от того, с каким аргументом -p была выполнена команда docker run.

Из python можно делать запросы, например, так:
```
import requests
response = requests.post("http://localhost:80/api/v1/recommend_for_user", json={"user_id": 0, "item_whitelist": []})
print(response.json())
```
Оба метода для выдачи рекомендаций принимают на вход json с обязательными полями user_id, item_whitelist, ознакомиться можноо здесь: `http://localhost:80/docs`.
Отметим, что для простоты методы реализованы так, что если приходит пустой item_whitelist, то считается, что доступны все айтемы. В реальном мире эту логику следует заменить на какую-то другую, ну а в учебных целях оставляем за собой право оставить как есть.

### Модификация
Маппинги (python dict) и векторы (numpy ndarray) лежат по путям paths из конфига config/config.yaml. Вместо существующих маппингов и векторов можно подложить свое, не забыва указать в config.yaml размерность в ключе dim в конфиге, поскольку Annoy требует передачи размерности в явном виде.
