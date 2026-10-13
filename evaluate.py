"""
Оценка качества извлечения юридических ссылок.

Сравнивает ответы алгоритма с эталоном (data/*.jsonl) и считает precision, recall, F1.
Ссылка считается верной, только если совпали все четыре поля:
law_id, article, point_article, subpoint_article.

Запуск:
  python evaluate.py --check                                  проверить сам оценщик
  python evaluate.py                                          алгоритм из extractor.py на dev
  python evaluate.py --data data/gold_holdout.jsonl           алгоритм на отложенной выборке
  python evaluate.py --url http://localhost:8978/detect       оценить работающий сервис по http
"""

import argparse
import json
import urllib.request
from collections import Counter

FIELDS = ('law_id', 'article', 'point_article', 'subpoint_article')


def load_gold(path):
    # читаем jsonl: на каждой строке один json-объект {'id': ..., 'text': ..., 'links': [...]}
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()                 # strip() убирает пробелы и перевод строки по краям
            if line:                            # пустые строки пропускаем
                records.append(json.loads(line))   # json.loads: строка json -> словарь python
    return records


def to_tuple(link):
    # словарь ссылки -> кортеж из 4 полей; .get(имя) вернёт None, если поля нет
    # кортеж нужен, потому что словарь нельзя положить в Counter, а кортеж можно
    return tuple(link.get(name) for name in FIELDS)


def predict_local(texts):
    # импортируем наш алгоритм только здесь: в режиме --check он не нужен
    import extractor
    aliases = extractor.load_aliases('law_aliases.json')   # словарь law_id -> варианты названия
    index = extractor.build_index(aliases)                  # индекс строится один раз на все тексты
    predictions = []
    for text in texts:
        predictions.append(extractor.extract_links(text, index))   # список словарей с 4 полями
    return predictions


def predict_http(texts, url):
    # отправляем каждый текст на сервис так же, как это сделает проверяющий
    predictions = []
    for text in texts:
        body = json.dumps({'text': text}).encode('utf-8')   # словарь -> строка json -> байты
        request = urllib.request.Request(
            url,
            data=body,
            headers={'Content-Type': 'application/json'},   # сообщаем серверу, что в теле json
            method='POST',
        )
        with urllib.request.urlopen(request) as response:   # urlopen отправляет запрос и возвращает ответ
            answer = json.loads(response.read().decode('utf-8'))   # байты -> строка -> словарь
        predictions.append(answer['links'])
    return predictions


def safe_div(a, b):
    # деление, которое не падает на нуле: если знаменатель 0, метрика не определена (None)
    if b == 0:
        return None
    return a / b


def f1_score(p, r):
    if p is None or r is None or p + r == 0:
        return None
    return 2 * p * r / (p + r)


def score(records, predictions):
    tp = fp = fn = 0                     # счётчики по ссылкам целиком
    law_tp = law_fp = law_fn = 0         # счётчики по законам (только law_id)
    errors = []
    # zip идёт по двум спискам параллельно: эталон текста и ответ алгоритма на этот текст
    for record, predicted in zip(records, predictions):
        gold_counter = Counter(to_tuple(link) for link in record['links'])   # Counter: сколько раз встречается каждая ссылка
        pred_counter = Counter(to_tuple(link) for link in predicted)
        matched = gold_counter & pred_counter    # & у Counter: пересечение мультимножеств (минимум из двух счётчиков)
        extra = pred_counter - gold_counter      # - у Counter: что есть в ответе сверх эталона -> ложные срабатывания
        missed = gold_counter - pred_counter     # что есть в эталоне, но не найдено -> пропуски
        tp += sum(matched.values())
        fp += sum(extra.values())
        fn += sum(missed.values())

        # отдельная метрика: нашли ли нужные законы в тексте, без учёта статей и пунктов
        gold_laws = set(link['law_id'] for link in record['links'])
        pred_laws = set(link.get('law_id') for link in predicted)
        law_tp += len(gold_laws & pred_laws)     # & у set: пересечение множеств
        law_fp += len(pred_laws - gold_laws)
        law_fn += len(gold_laws - pred_laws)

        if extra or missed:
            errors.append((record['id'], record['text'], missed, extra))

    links_p = safe_div(tp, tp + fp)
    links_r = safe_div(tp, tp + fn)
    laws_p = safe_div(law_tp, law_tp + law_fp)
    laws_r = safe_div(law_tp, law_tp + law_fn)
    return {
        'links': (links_p, links_r, f1_score(links_p, links_r), tp, fp, fn),
        'laws': (laws_p, laws_r, f1_score(laws_p, laws_r), law_tp, law_fp, law_fn),
        'errors': errors,
    }


def fmt_metric(x):
    # число -> строка с тремя знаками; None -> 'н/д' (не определено)
    if x is None:
        return '  н/д'
    return f'{x:.3f}'


def fmt_link(t):
    # кортеж (law_id, article, point, subpoint) -> читаемая строка
    law_id, article, point, subpoint = t
    parts = [f'law_id={law_id}']
    if article is not None:
        parts.append(f'ст. {article}')
    if point is not None:
        parts.append(f'п./ч. {point}')
    if subpoint is not None:
        parts.append(f'пп. {subpoint}')
    return ', '.join(parts)


def print_report(title, records, result, max_errors):
    n_links = sum(len(r['links']) for r in records)
    print(f'\n{title}')
    print(f'текстов: {len(records)}, эталонных ссылок: {n_links}')
    print(f'{"":20}{"precision":>10}{"recall":>9}{"f1":>8}{"tp":>6}{"fp":>5}{"fn":>5}')
    for name, key in [('ссылки целиком', 'links'), ('законы в тексте', 'laws')]:
        p, r, f, tp, fp, fn = result[key]
        print(f'{name:20}{fmt_metric(p):>10}{fmt_metric(r):>9}{fmt_metric(f):>8}{tp:>6}{fp:>5}{fn:>5}')

    errors = result['errors']
    if not errors or max_errors == 0:
        return
    print(f'\nтекстов с ошибками: {len(errors)}, показано: {min(len(errors), max_errors)}')
    for record_id, text, missed, extra in errors[:max_errors]:
        short = ' '.join(text.split())          # split() + join схлопывают переносы и двойные пробелы
        if len(short) > 160:
            short = short[:160] + '...'
        print(f'\n[{record_id}] {short}')
        for t, count in missed.items():          # .items() у Counter: пары (ссылка, сколько раз)
            print(f'    пропущено ({count}): {fmt_link(t)}')
        for t, count in extra.items():
            print(f'    лишнее    ({count}): {fmt_link(t)}')


def self_check(paths):
    # проверка самого оценщика на случаях с заранее известным ответом
    all_ok = True
    for path in paths:
        records = load_gold(path)
        perfect = [r['links'] for r in records]  # подаём эталон как ответ алгоритма -> должно быть f1 = 1
        empty = [[] for _ in records]            # пустой ответ на каждый текст -> recall должен быть 0
        r_perfect = score(records, perfect)
        r_empty = score(records, empty)
        ok_perfect = r_perfect['links'][2] == 1.0
        ok_empty = r_empty['links'][1] == 0.0
        print(f'{path}: эталон как ответ -> f1 = {fmt_metric(r_perfect["links"][2])} '
              f'[{"OK" if ok_perfect else "ОШИБКА"}]; '
              f'пустой ответ -> recall = {fmt_metric(r_empty["links"][1])} '
              f'[{"OK" if ok_empty else "ОШИБКА"}]')
        all_ok = all_ok and ok_perfect and ok_empty
    print('\nоценщик работает корректно' if all_ok else '\nв оценщике ошибка')


def main():
    # argparse разбирает параметры командной строки вида --data путь
    parser = argparse.ArgumentParser(description='оценка извлечения юридических ссылок')
    parser.add_argument('--data', default='data/gold_dev.jsonl', help='файл эталона')
    parser.add_argument('--url', default=None, help='адрес сервиса, например http://localhost:8978/detect')
    parser.add_argument('--check', action='store_true', help='проверить сам оценщик')
    parser.add_argument('--max-errors', type=int, default=30, help='сколько текстов с ошибками показать')
    args = parser.parse_args()               # args.data, args.url, args.check, args.max_errors

    if args.check:
        self_check(['data/gold_dev.jsonl', 'data/gold_holdout.jsonl'])
        return

    records = load_gold(args.data)
    texts = [r['text'] for r in records]
    if args.url:
        predictions = predict_http(texts, args.url)
        source = f'сервис {args.url}'
    else:
        predictions = predict_local(texts)
        source = 'extractor.py'
    result = score(records, predictions)
    print_report(f'{args.data} ({source})', records, result, args.max_errors)


if __name__ == '__main__':
    main()
