# базовый образ: linux с python 3.12 (slim - облегчённая версия)
FROM python:3.12-slim

# печатать вывод python сразу, без буфера: иначе print из сервиса может не появиться в docker logs
ENV PYTHONUNBUFFERED=1

# рабочая папка внутри контейнера; все следующие команды выполняются в ней
WORKDIR /app

# сначала только список библиотек: если он не менялся, docker возьмёт установку из кэша и не будет ставить заново
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код сервиса и база законов (тесты, эталон и оценщик сервису не нужны)
COPY main.py extractor.py law_aliases.json ./

# порт, на котором сервис слушает внутри контейнера
EXPOSE 8978

# команда при запуске контейнера: uvicorn, приложение app из main.py, все интерфейсы, порт 8978
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8978"]
