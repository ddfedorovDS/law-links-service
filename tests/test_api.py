"""
Тесты сервиса: запросы к /health и /detect без запуска сервера.
Запуск из корня проекта: python -m pytest
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient   # клиент, который шлёт запросы приложению прямо из кода (нужен httpx)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


@pytest.fixture(scope='module')
def client():
    # with запускает lifespan: при входе строится индекс, при выходе сервис "останавливается"
    with TestClient(main.app) as c:
        yield c          # yield в fixture: отдать клиента тестам, а после тестов выйти из with


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_detect_example_from_task(client):
    response = client.post('/detect', json={'text': 'пп. 1, 2, 3 п. 2 ст. 3 НК РФ'})   # json=: словарь отправится как json-тело
    assert response.status_code == 200
    links = response.json()['links']
    assert len(links) == 3
    assert links[0] == {'law_id': 15, 'article': '3', 'point_article': '2', 'subpoint_article': '1'}


def test_detect_curl_example(client):
    # пример запроса из README задания
    response = client.post('/detect', json={'text': 'Согласно статье 23 Налогового кодекса и ст. 145 Гражданского кодекса'})
    links = response.json()['links']
    assert [(l['law_id'], l['article']) for l in links] == [(15, '23'), (5, '145')]


def test_empty_text(client):
    response = client.post('/detect', json={'text': ''})
    assert response.status_code == 200
    assert response.json() == {'links': []}


def test_wrong_body_gives_422(client):
    # поле названо неправильно -> fastapi сам отвечает 422, до нашего кода дело не доходит
    response = client.post('/detect', json={'txt': 'ст. 5 НК РФ'})
    assert response.status_code == 422
