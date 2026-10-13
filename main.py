import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

import extractor   # наш алгоритм: load_aliases, build_index, extract_links

# путь к базе законов рядом с этим файлом; так он находится, из какой бы папки ни запускали сервис
ALIASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'law_aliases.json')


class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # всё до yield выполняется один раз при запуске сервиса
    aliases = extractor.load_aliases(ALIASES_PATH)       # читает json в кодировке utf-8
    app.state.index = extractor.build_index(aliases)     # индекс названий законов, около секунды
    print('Сервис запускается: индекс законов построен')
    yield
    print('Сервис завершается')


def get_index(request: Request) -> Dict:
    # функция-зависимость: отдаёт индекс, построенный при запуске
    return request.app.state.index


app = FastAPI(
    title='Law Links Service',
    description='Сервис для выделения юридических ссылок из текста',
    version='1.0.0',
    lifespan=lifespan,
)


@app.post('/detect')
def get_law_links(
    data: TextRequest,
    index: Dict = Depends(get_index),
) -> LinksResponse:
    """
    Принимает текст и возвращает список юридических ссылок
    """
    # обычная def, а не async def: алгоритм только считает и ничего не ждёт,
    # такую функцию fastapi выполняет в отдельном потоке и не блокирует остальные запросы
    links = extractor.extract_links(data.text, index)              # список словарей с 4 полями
    return LinksResponse(links=[LawLink(**link) for link in links])  # **link: словарь -> именованные аргументы


@app.get('/health')
async def health_check():
    """
    Проверка состояния сервиса
    """
    return {'status': 'healthy'}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8978)
