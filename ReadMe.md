# Проект на Python с использованием ClickHouse и веб-интерфейса

## Описание проекта

Данный проект написан на языке программирования Python (версия 3.12 или выше). Он работает с базой данных в облаке ClickHouse и предоставляет визуальный интерфейс в виде веб-страницы.

## Установка

1. Убедитесь, что у вас установлена версия Python не ниже 3.12.
2. Установите необходимые библиотеки, выполнив команду:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск проекта

Для запуска проекта выполните следующий шаги:

1. Запустите основной файл проекта `_MainHttp.py`:
   ```bash
   py _MainHttp.py
   ```
   После успешного запуска вы увидите сообщение:
   ```
   ======== Running on http://localhost:8090 ========
   ```

2. Откройте файл `index.html` в современном браузере. Для этого просто дважды кликните на файл. Вы увидите окно с предложением выбрать даты временного окна.

3. Выберите необходимые даты и нажмите кнопку "Найти". Программа начнет обработку данных.

## Логи и результаты

- Логи работы программы можно посмотреть в консоли сервера.
- После обработки запроса результаты отображаются на странице.

## Визуальный интерфейс

- В левой части страницы отображаются временные диапазоны вероятностей аномалий и их вероятности.
- В правой части страницы выводятся четыре графика исходных данных.

### Минималистичный стиль

Визуальный стиль проекта выполнен в супер-минимализме (у нас лапки🐈), чтобы не отвлекать пользователя от основной информации.

![Демонстрация](demo.gif)

## Пример команд

Запуск основного файла:
```bash
py _MainHttp.py
```

Установка библиотек:
```bash
pip install -r requirements.txt
```

## Требования

- Python 3.12+
- Интернет-соединение для доступа к базе данных ClickHouse
- Современный браузер для работы с веб-интерфейсом