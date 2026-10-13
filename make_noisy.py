"""
Зашумлённая выборка: те же фразы, что в эталоне, но записанные небрежно.
Смысл ссылок не меняется, поэтому правильные ответы берутся из исходной разметки без изменений.

Виды шума (в каждой фразе 1-2 случайных):
  - опечатка в слове из названия закона: пропуск, удвоение, замена или перестановка букв
  - точка после маркера пропущена: «ст. 5» -> «ст 5»
  - пробел после маркера пропущен: «ст. 5» -> «ст.5»
  - номер слеплен с названием: «ст. 5 НК РФ» -> «ст. 5НК РФ»
  - случайный регистр: «Налогового» -> «НАЛОГОВОГО» или «налогового»

Запуск: python make_noisy.py  -> data/gold_noisy.jsonl (seed фиксирован, результат воспроизводим)
"""

import json
import random
import re

import extractor

SOURCES = ['data/gold_dev.jsonl', 'data/gold_holdout.jsonl', 'data/gold_blind.jsonl']
CYR = 'абвгдежзийклмнопрстуфхцчшщыэюя'


def law_vocabulary():
    # основы слов, которые встречаются в названиях законов: в них и будем делать опечатки
    aliases = extractor.load_aliases('law_aliases.json')
    vocab = set()
    for variants in aliases.values():
        for v in variants:
            for stem_, _, _ in extractor.tokenize(extractor.normalize(v)):
                if stem_.isalpha() and len(stem_) >= 5:
                    vocab.add(stem_)
    return vocab


def typo(word):
    # одна опечатка в середине слова (первую букву не трогаем)
    i = random.randint(1, len(word) - 2)
    kind = random.choice(['drop', 'double', 'replace', 'swap'])
    if kind == 'drop':
        return word[:i] + word[i + 1:]
    if kind == 'double':
        return word[:i] + word[i] + word[i:]
    if kind == 'replace':
        return word[:i] + random.choice(CYR) + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def name_typo(text, vocab):
    # находим слова длиной от 6 букв, чья основа есть в названиях законов, и портим одно из них
    candidates = []
    for m in re.finditer(r'[А-Яа-яЁё]{6,}', text):
        if extractor.stem(m.group(0).lower().replace('ё', 'е')) in vocab:
            candidates.append(m)
    if not candidates:
        return text
    m = random.choice(candidates)
    return text[:m.start()] + typo(m.group(0)) + text[m.end():]


def drop_marker_dot(text):
    return re.sub(r'\b(ст|п|пп|ч)\. (?=\d)', r'\1 ', text, count=1)


def drop_marker_space(text):
    return re.sub(r'\b(ст|п|пп|ч)\. (?=\d)', r'\1.', text)


def glue_number(text):
    return re.sub(r'(\d) ((?:НК|ГК|УК|ТК|АПК|ГПК|КоАП|ЖК|ЗК|БК|СК|УПК) РФ)', r'\1\2', text, count=1)


def random_case(text, vocab):
    candidates = [m for m in re.finditer(r'[А-Яа-яЁё]{6,}', text)
                  if extractor.stem(m.group(0).lower().replace('ё', 'е')) in vocab]
    if not candidates:
        return text
    m = random.choice(candidates)
    word = m.group(0).upper() if random.random() < 0.5 else m.group(0).lower()
    return text[:m.start()] + word + text[m.end():]


def main(seed=0, out_path='data/gold_noisy.jsonl'):
    random.seed(seed)                            # фиксируем случайность: при одном seed файл всегда одинаковый
    vocab = law_vocabulary()
    rows = []
    for path in SOURCES:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if not rec['links']:
                    continue                      # фразы без ссылок для этой проверки неинтересны
                text = rec['text']
                text = name_typo(text, vocab)     # опечатка в названии есть всегда - это главное, что проверяем
                other = random.choice(['dot', 'space', 'glue', 'case', 'none'])
                if other == 'dot':
                    text = drop_marker_dot(text)
                elif other == 'space':
                    text = drop_marker_space(text)
                elif other == 'glue':
                    text = glue_number(text)
                elif other == 'case':
                    text = random_case(text, vocab)
                rows.append({'id': 'noisy_' + rec['id'], 'text': text, 'links': rec['links']})
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(out_path + ':', len(rows), 'текстов,', sum(len(r['links']) for r in rows), 'ссылок')


if __name__ == '__main__':
    main()
