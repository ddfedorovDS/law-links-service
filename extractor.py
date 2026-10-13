"""
Извлечение юридических ссылок из текста.

Схема:
  1. normalize      - приводим текст к единому виду (регистр, ё, кавычки, тире, пробелы, №, латинские двойники)
  2. tokenize       - режем на слова, каждое слово стеммим: «Налогового Кодекса» -> ('налогов', 'кодекс')
  3. find_laws      - ищем названия актов из law_aliases.json (самое длинное совпадение)
  4. parse_chains   - слева от каждого найденного акта разбираем цепочку «подпункт / пункт или часть / статья»
  5. extract_links  - раскрываем перечисления и собираем ответ

Использование:
  aliases = load_aliases('law_aliases.json')
  index = build_index(aliases)                  # один раз, около секунды
  links = extract_links('пп. 1 п. 1 ст. 374 НК РФ', index)
  # [{'law_id': 15, 'article': '374', 'point_article': '1', 'subpoint_article': '1'}]
"""

import itertools
import json
import re

from nltk.stem.snowball import SnowballStemmer   # стеммер: отрезает окончания по правилам, словари не нужны

STEMMER = SnowballStemmer('russian')
STEM_CACHE = {}                                  # словарь слово -> основа, чтобы не стеммить одно слово дважды


# ---------- 1. нормализация ----------

QUOTES = '«»“”„‟"'                               # все виды кавычек приводим к прямой "
DASHES = '–—−‒'                                  # все виды тире приводим к дефису -
LATIN_TO_CYR = str.maketrans('acekmhopxytb', 'асекмнорхутв')   # латинские буквы-двойники -> кириллица


def fix_mixed_word(match):
    # вызывается для каждого слова; match.group(0) - само слово
    word = match.group(0)
    has_cyr = re.search('[а-я]', word) is not None
    has_lat = re.search('[a-z]', word) is not None
    if has_cyr and has_lat:                      # слово смешанное, например 'cк' (латинская c + кириллица)
        return word.translate(LATIN_TO_CYR)      # translate заменяет символы по таблице maketrans
    return word                                  # чисто латинские слова (fifa) не трогаем


def normalize(text):
    s = text
    s = re.sub(r'(?<![A-Za-zА-Яа-яЁё])N\s?(?=\d)', '№', s)   # латинская N перед числом -> №  (N 217н -> №217н)
    s = s.lower()                                            # нижний регистр
    s = s.replace('ё', 'е')
    for q in QUOTES:
        s = s.replace(q, '"')
    for d in DASHES:
        s = s.replace(d, '-')
    s = re.sub(r'\s+', ' ', s)                   # любые пробельные символы (переносы, двойные, неразрывные) -> один пробел
    s = re.sub(r'№\s*', '№', s)                  # '№ 152' -> '№152'
    s = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', s)  # '194 - 199' -> '194-199'; \1 и \2 - то, что захватили скобки
    s = re.sub(r'[a-zа-я]+', fix_mixed_word, s)  # re.sub с функцией: для каждого совпадения вызывается fix_mixed_word
    s = re.sub(r'(\d)(?=[а-я]{2,})', r'\1 ', s)  # '374нк рф' -> '374 нк рф'; (?=...) проверяет, что дальше 2+ буквы, не забирая их
    return s.strip()


# ---------- 2. токенизация со стеммингом ----------

TOKEN_RE = re.compile(r'№?[0-9a-zа-я]+(?:[./-][0-9a-zа-я]+)*|"')   # слово или число (можно с точками, дефисом, /) или кавычка


def stem(word):
    if word not in STEM_CACHE:
        parts = word.split('-')                    # 'уголовно-процессуального' -> ['уголовно', 'процессуального']
        if len(parts) > 1 and all(p.isalpha() for p in parts):
            # составное слово через дефис: стеммим каждую часть отдельно
            STEM_CACHE[word] = '-'.join(stem(p) for p in parts)
        elif len(word) <= 3 or not word.isalpha():   # короткие слова (рф, нк, апк) и числа (№152-фз) не стеммим
            STEM_CACHE[word] = word
        else:
            STEM_CACHE[word] = STEMMER.stem(word)
    return STEM_CACHE[word]


def tokenize(norm_text):
    # возвращает список кортежей (основа или '"', начало, конец) - позиции в нормализованном тексте
    tokens = []
    for m in TOKEN_RE.finditer(norm_text):        # finditer: все совпадения по очереди, с позициями
        word = m.group(0)
        if word == '"':
            tokens.append(('"', m.start(), m.end()))
        else:
            tokens.append((stem(word), m.start(), m.end()))
    return tokens


# ---------- 3. индекс названий и поиск актов ----------

def load_aliases(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)                       # dict: '15' -> ['Налоговый кодекс Российской Федерации', ...]


def build_index(aliases):
    # ключ индекса - кортеж основ слов названия без кавычек: ('налогов', 'кодекс', 'российск', 'федерац')
    keys = {}
    for law_id, variants in aliases.items():
        for variant in variants:
            norm = normalize(variant)
            toks = tokenize(norm)
            key = tuple(t[0] for t in toks if t[0] != '"')
            if not key:
                continue
            bare = norm.startswith('"')           # вариант - только название в кавычках, без типа акта («О полиции»)
            if key not in keys:
                keys[key] = {'ids': set(), 'needs_quote': True}
            keys[key]['ids'].add(int(law_id))
            if not bare:
                keys[key]['needs_quote'] = False  # если хоть один вариант с таким ключом не голый, кавычки не обязательны
    # by_first: первая основа -> список ключей, начинающихся с неё, от длинных к коротким
    by_first = {}
    for key in keys:
        by_first.setdefault(key[0], []).append(key)   # setdefault: взять список по ключу, а если его нет - создать пустой
    for first in by_first:
        by_first[first].sort(key=len, reverse=True)
    return {'keys': keys, 'by_first': by_first, 'first_stems': list(by_first)}


def within_one_edit(a, b):
    # True, если строки отличаются не больше чем на одну правку: пропуск, лишняя буква, замена или перестановка соседних
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:   # идём до первого несовпадения
        i += 1
    if len(a) == len(b):
        if a[i + 1:] == b[i + 1:]:                       # замена одной буквы
            return True
        return a[i:i + 2] == b[i:i + 2][::-1] and a[i + 2:] == b[i + 2:]   # перестановка соседних; [::-1] разворачивает строку
    if len(a) > len(b):
        return a[i + 1:] == b[i:]                        # в a лишняя буква
    return a[i:] == b[i + 1:]                            # в a пропущена буква


def close(a, b):
    # совпадение основ с допуском опечатки; короткие слова (нк, рф, апк, фз) и числа - только точно
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4 or not a.isalpha() or not b.isalpha():
        return False
    if within_one_edit(a, b):
        return True
    # опечатка в окончании мешает стеммеру отрезать его: 'жилищногг' vs 'жилищн', 'гражданскго' vs 'гражданск'
    short, long_ = sorted([a, b], key=len)       # sorted по длине: сначала короткая строка
    return len(short) >= 5 and long_.startswith(short) and len(long_) - len(short) <= 3


def fuzzy_match(stems, i, index):
    # ищем название с опечатками, начинающееся со слова i; допускаем не больше одного неточного слова
    best = None
    for first in index['first_stems']:
        if not close(first, stems[i]):
            continue
        for key in index['by_first'][first]:
            if best is not None and len(key) <= len(best):
                break                                    # ключи от длинных к коротким: короче лучшего не ищем
            if i + len(key) > len(stems):
                continue
            inexact = 0
            ok = True
            for j in range(len(key)):
                if key[j] == stems[i + j]:
                    continue
                if close(key[j], stems[i + j]):
                    inexact += 1
                    if inexact > 1:
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok and inexact == 1 and len(key) >= 2:    # одно неточное слово и название минимум из двух слов
                best = key
    return best


def find_laws(tokens, index, fuzzy_starts=None):
    # идём по словам текста; в каждой позиции пробуем самое длинное название, которое здесь начинается
    words = []                                    # (основа, начало, конец, стоит ли перед словом кавычка)
    prev_quote = False
    for stem_, start, end in tokens:
        if stem_ == '"':
            prev_quote = True
            continue
        words.append((stem_, start, end, prev_quote))
        prev_quote = False

    stems = [w[0] for w in words]
    found = []                                    # (law_id или None, начало, конец) в нормализованном тексте
    i = 0
    while i < len(words):
        match = None
        for key in index['by_first'].get(stems[i], []):          # .get(..., []) - пустой список, если ключа нет
            if tuple(stems[i:i + len(key)]) == key:              # срез stems[i:i+n] - n слов начиная с i
                info = index['keys'][key]
                if info['needs_quote'] and not words[i][3]:
                    continue                                     # голое название без кавычек в тексте не считаем
                match = (key, info)
                break                                            # ключи отсортированы, первый подошедший - самый длинный
        if match is None and fuzzy_starts and words[i][1] in fuzzy_starts:
            key = fuzzy_match(stems, i, index)           # опечатка в названии сразу после «ст. N»
            if key is not None:
                info = index['keys'][key]
                if not (info['needs_quote'] and not words[i][3]):
                    match = (key, info)
        if match is None:
            i += 1
            continue
        key, info = match
        start = words[i][1]
        end = words[i + len(key) - 1][2]
        law_id = None
        if len(info['ids']) == 1:
            law_id = next(iter(info['ids']))      # единственный элемент множества
        found.append((law_id, start, end))        # law_id = None: название относится к нескольким актам (правило 5)
        i += len(key)
    return found


# ---------- 4. разбор цепочки слева от акта ----------

ORDINALS = {'перв': '1', 'втор': '2', 'трет': '3', 'четверт': '4', 'пят': '5',
            'шест': '6', 'седьм': '7', 'восьм': '8', 'девят': '9', 'десят': '10'}

NUM = r'\d+(?:\.\d+)*(?:-\d+(?:\.\d+)*)?'                     # 12, 30.1, 3.345, 43.2-6, 194-199
ORD = r'(?:' + '|'.join(ORDINALS) + r')[а-я]*'                # второй, первой, ...
LETTER = r'"?[а-я]"?\)?(?![а-я0-9])'                         # одна буква, можно в кавычках или со скобкой: а, "а", а)
VALUE = r'(?:' + NUM + '|' + ORD + '|' + LETTER + ')'
SEP = r'(?:\s*,\s*|\s+и\s+)'                                  # разделитель внутри перечисления
LIST = VALUE + r'(?:' + SEP + VALUE + r')*'                   # 4, 5, 6 и 8

B = r'(?<![а-я0-9.])'                                         # маркер начинается не внутри другого слова
# Только словоформы маркеров: «статус» и «частично» не обозначают уровни ссылки.
# Граница после слова запрещает принять его остаток за буквенное значение.
WORD_END = r'(?![a-zа-я_])'
POINT_ENDING = r'(?:а|у|ом|е|ы|ов|ам|ами|ах)?'
# пп(?=\s*\d), п(?=\s*\d), ч(?=\s*\d), ст(?=\s*\d): маркер без точки, если сразу за ним число («ст 5»)
M_SUB = r'(?:подпункт' + POINT_ENDING + WORD_END + r'|подп\.|пп\.|пп(?=\s*\d))'
M_POINT = (r'(?:пунк?т' + POINT_ENDING + WORD_END
           + r'|п\.\s?п\.|п\.|п(?=\s*\d)|част(?:ь|и|ью|ей|ям|ями|ях)' + WORD_END + r'|ч\.|ч(?=\s*\d))')
# пунк?т сохраняет поддержку опечатки «пунта»; стат. — сокращение «статья».
M_ART = (r'(?:ст\.\s?ст\.|стат\.|(?:стать(?:я|и|е|ю|ей|ею|ям|ями|ях)|статей)'
         + WORD_END + r'|ст\.|ст(?=\s*\d))')
LVL = r'\s*,?\s*(?:в\s+)?'                                    # между уровнями: пробел, запятая, предлог «в»

CHAIN = (
    r'(?:' + B + r'(?P<sub_m>' + M_SUB + r')\s*(?P<sub>' + LIST + r')' + LVL + r')?'
    r'(?:' + B + r'(?P<pt_m>' + M_POINT + r')\s*(?P<pt>' + LIST + r')' + LVL + r')?'
    r'(?:' + B + r'(?P<pt2_m>' + M_POINT + r')\s*(?P<pt2>' + LIST + r')' + LVL + r')?'
    r'(?:' + B + r'(?P<art_m>' + M_ART + r')\s*(?P<art>' + LIST + r'))?'
)
# слова, которые могут стоять между цепочкой и названием акта: «Закона», «Федерального закона от 27.07.2006 № 152-ФЗ»
FILL_WORD = (r'(?:федеральн[а-я]*|закон[а-я]*|фз|рф|кодекс[а-я]*|российск[а-я]*|федераци[а-я]*'
             r'|указ[а-я]*|от \d{1,2}\.\d{1,2}\.\d{2,4}|№[0-9][0-9a-zа-я./-]*)')
LAST_CHAIN_RE = re.compile(CHAIN + r'(?:\s+' + FILL_WORD + r')*\s*"?\s*$')   # цепочка, прижатая к концу левого текста
PREV_CHAIN_RE = re.compile(CHAIN + r'\s*$')                                  # предыдущая цепочка («п. 2 ст. 450 и ...»)
JOINER_RE = re.compile(r'(?:\s*,\s*|\s+и\s+|\s+а также\s+)$')                # чем соединены цепочки перед одним актом


CHAIN_FIND_RE = re.compile(CHAIN + r'(?:\s+' + FILL_WORD + r')*\s*"?\s*')    # цепочка в любом месте текста


def chain_ends(norm_text):
    # позиции, где сразу после цепочки «пп./п./ст. ...» должно начинаться название акта
    ends = set()
    for m in CHAIN_FIND_RE.finditer(norm_text):
        if m.group('art') or m.group('pt') or m.group('sub'):   # пустые совпадения (без единого уровня) пропускаем
            ends.add(m.end())
    return ends


def split_values(list_text):
    # '4, 5, 6 и 8' -> ['4', '5', '6', '8']; '"а"' -> ['а']; 'второй' -> ['2']
    values = []
    for raw in re.split(SEP, list_text):          # re.split режет строку по шаблону-разделителю
        v = raw.strip().strip('")')               # strip('")') убирает кавычки и скобку по краям: '"а"' -> 'а', 'б)' -> 'б'
        for prefix, digit in ORDINALS.items():
            if re.fullmatch(prefix + r'[а-я]*', v):   # fullmatch: шаблон должен совпасть со всей строкой
                v = digit
        values.append(v)
    return values


def chain_from_match(m):
    # объект совпадения -> словарь уровней или None, если ни один уровень не найден
    if m is None:
        return None
    subs = split_values(m.group('sub')) if m.group('sub') else [None]
    points = split_values(m.group('pt')) if m.group('pt') else [None]   # если есть и пункт, и часть - берём первый (правило 16)
    arts = split_values(m.group('art')) if m.group('art') else [None]
    if subs == [None] and points == [None] and arts == [None]:
        return None
    return {'subs': subs, 'points': points, 'arts': arts}


def parse_chains(left_text):
    # left_text - текст между предыдущим актом и текущим; ищем цепочки, прижатые к его концу
    m = LAST_CHAIN_RE.search(left_text)           # search ищет первое совпадение в любом месте строки
    chain = chain_from_match(m)
    if chain is None:
        return []
    chains = [chain]
    rest = left_text[:m.start()]                  # всё, что левее найденной цепочки
    while True:
        j = JOINER_RE.search(rest)                # перед цепочкой стоит «и» или запятая?
        if j is None:
            break
        rest = rest[:j.start()]
        m = PREV_CHAIN_RE.search(rest)
        chain = chain_from_match(m)
        if chain is None:
            break
        chains.insert(0, chain)                   # insert(0, ...) - вставить в начало списка, чтобы сохранить порядок текста
        rest = rest[:m.start()]
    return chains


# ---------- 5. сборка ответа ----------

def extract_links(text, index):
    norm = normalize(text)
    laws = find_laws(tokenize(norm), index, fuzzy_starts=chain_ends(norm))
    links = []
    prev_end = 0
    for law_id, start, end in laws:
        left = norm[max(prev_end, start - 300):start]   # левый контекст: не дальше предыдущего акта и не длиннее 300 символов
        prev_end = end
        if law_id is None:
            continue                                     # неоднозначное название - ссылку не возвращаем
        for chain in parse_chains(left):
            # itertools.product - все комбинации: 2 подпункта x 3 пункта x 1 статья = 6 ссылок
            for art, point, sub in itertools.product(chain['arts'], chain['points'], chain['subs']):
                links.append({'law_id': law_id, 'article': art, 'point_article': point, 'subpoint_article': sub})
    return links
