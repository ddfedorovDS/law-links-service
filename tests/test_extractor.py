"""
Тесты на ловушки из тестового текста задания и данных.
Каждый тест подаёт алгоритму одну трудную фразу и проверяет ответ.
Запуск из корня проекта: python -m pytest
"""

import os
import sys

import pytest

# путь к корню проекта (папка выше tests); добавляем его в sys.path, чтобы работал import extractor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import extractor  # noqa: E402  (импорт после правки sys.path - так задумано)


@pytest.fixture(scope='module')
def index():
    # fixture: pytest вызовет эту функцию один раз на весь файл (scope='module')
    # и передаст результат в каждый тест, у которого есть параметр index
    aliases = extractor.load_aliases(os.path.join(ROOT, 'law_aliases.json'))
    return extractor.build_index(aliases)


def run(text, index):
    # ответ алгоритма -> список кортежей (law_id, статья, пункт, подпункт), так проще сравнивать
    result = []
    for link in extractor.extract_links(text, index):
        result.append((link['law_id'], link['article'], link['point_article'], link['subpoint_article']))
    return result


def test_full_chain(index):
    assert run('В соответствии с пп. 1 п. 1 ст. 374 НК РФ объектами', index) == [(15, '374', '1', '1')]


def test_double_space_and_no_article(index):
    assert run('Согласно п.  10 АПК для целей учета', index) == [(0, None, '10', None)]


def test_inflected_law_name(index):
    assert run('упомянуть ст. 275 Налогового Кодекса Российской Федерации', index) == [(15, '275', None, None)]


def test_enumeration_from_task(index):
    # пример из README задания: три подпункта -> три ссылки
    assert run('пп. 1, 2, 3 п. 2 ст. 3 НК РФ', index) == [
        (15, '3', '2', '1'), (15, '3', '2', '2'), (15, '3', '2', '3')]


def test_two_level_enumeration(index):
    # 3 подпункта x 2 пункта = 6 ссылок; «в» перед «статье» - предлог, а не пункт
    result = run('в подпунктах а, б и с пункта 3.345, 23 в статье 66 НК РФ', index)
    assert len(result) == 6
    assert (15, '66', '23', 'с') in result


def test_typo_in_marker(index):
    text = 'Согласно подпункту 2 пунта б статьи 22 Федерального закона "О государственной гражданской службе Российской Федерации"'
    assert run(text, index) == [(362, '22', 'б', '2')]


def test_letter_point(index):
    assert run('согласно пункта а ст. 20 АПК, запрещены', index) == [(0, '20', 'а', None)]


def test_law_not_in_base(index):
    assert run('Согласно статье 17 Конституции Российской Федерации, каждый', index) == []
    assert run('А в пп. 1 п. 2 ст. 34 рандомного кодекса было всё.', index) == []
    assert run('Согласно ст. 37 ФЗ №20-456, прокурор', index) == []


def test_short_alias_not_inside_word(index):
    # «нк» не должно находиться внутри слов «банках», «банковской»
    assert run('О банках и банковской деятельности; см. ст. 5 НК РФ', index) == [(15, '5', None, None)]


def test_latin_letter_in_base(index):
    # в базе Семейный кодекс записан как 'CК' с латинской C, в тексте - кириллица
    assert run('Согласно ст. 90 СК РФ, алименты', index) == [(8, '90', None, None)]


def test_quotes_and_yo(index):
    assert run('Согласно ст. 19 Федерального закона "О бухгалтерском учёте", учёт', index) == [(224, '19', None, None)]


def test_law_word_between_chain_and_title(index):
    # в базе нет варианта «Закон "О персональных данных"», есть только голое название
    assert run('В соответствии с п. 2 ст. 14 Закона "О персональных данных", оператор', index) == [(330, '14', '2', None)]


def test_part_instead_of_point(index):
    assert run('общие правила часть 3, ст. 30.1 КоАП РФ (постановления', index) == [(17, '30.1', '3', None)]


def test_ambiguous_number(index):
    # «№ 135-ФЗ» в базе у трёх разных законов - ссылку не возвращаем (правило 5)
    assert run('Согласно ст. 11 Федерального закона № 135-ФЗ картельные соглашения запрещены', index) == []


def test_title_without_chain_is_not_link(index):
    assert run('учитываемое в качестве объектов основных средств в порядке', index) == []


def test_duplicates_kept(index):
    text = 'установленном для ст. 105 УК РФ. Однако, в соответствии со ст. 105 УК РФ, убийство'
    assert run(text, index) == [(13, '105', None, None), (13, '105', None, None)]


@pytest.mark.parametrize('word', [
    'статус', 'статистика', 'статуя', 'статичный',
    'частично', 'частный', 'частота', 'пунктир', 'подпунктир',
])
def test_ordinary_words_are_not_citation_markers(index, word):
    assert run(f'{word} 5 НК РФ', index) == []
    assert run(f'{word} НК РФ', index) == []


def test_ordinary_word_does_not_add_a_point_to_valid_article(index):
    assert run('частично 2 ст. 5 НК РФ', index) == [(15, '5', None, None)]


@pytest.mark.parametrize(('markers', 'expected'), [
    ('статья статьи статье статью статьей статьею статей статьям статьями статьях',
     (15, '5', None, None)),
    ('часть части частью частей частям частями частях', (15, None, '5', None)),
    ('пункт пункта пункту пунктом пункте пункты пунктов пунктам пунктами пунктах',
     (15, None, '5', None)),
    ('подпункт подпункта подпункту подпунктом подпункте подпункты подпунктов подпунктам подпунктами подпунктах',
     (15, None, None, '5')),
    ('ст. ст.ст. стат.', (15, '5', None, None)),
    ('п. п.п. ч.', (15, None, '5', None)),
    ('пп. подп.', (15, None, None, '5')),
])
def test_supported_marker_forms(index, markers, expected):
    for marker in markers.split():
        assert run(f'{marker} 5 НК РФ', index) == [expected], marker


def test_compact_abbreviated_chain(index):
    assert run('пп.1 п.2 ст.5 НК РФ', index) == [(15, '5', '2', '1')]
