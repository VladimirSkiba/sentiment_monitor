import random
import csv
import os

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR,exist_ok=True)

POSITIVE_CORE = [
    "Отличный товар, всё понравилось",
    "Качество на высоте, рекомендую",
    "Быстрая доставка и хорошая упаковка",
    "Соответствует описанию, очень доволен покупкой",
    "Лучшая покупка за последнее время",
    "Превзошло мои ожидания, буду заказывать ещё",
    "Удобно, практично и приятно выглядит",
    "Цена полностью оправдана качеством",
    "Использую уже месяц, никаких проблем",
    "Прекрасное сочетание цены и качества",
]

NEGATIVE_CORE = [
    "Качество ужасное, не рекомендую",
    "Сломалось через неделю использования",
    "Не соответствует описанию на сайте",
    "Доставка задержалась на две недели",
    "Деньги на ветер, очень разочарован",
    "Материал дешёвый и неприятный",
    "Заказывал одно, получил совсем другое",
    "После первого использования появились дефекты",
    "Хуже, чем я ожидал, верну обратно",
    "Не работает как обещано в описании",
]

FILLERS = [
    "", " Спасибо магазину.", " Буду заказывать снова.",
    " Не ожидал такого.", " Друзьям тоже посоветую.",
    " Жду следующую покупку.", " Менеджер был вежлив.",
    " Упаковка немного помялась.", " В целом всё ок.",
    " Цена кстати приятная."
]

PRODUCTS = [
    "наушники", "кофеварку", "рюкзак", "клавиатуру", "лампу",
    "куртку", "блендер", "телефон", "коврик для йоги", "часы",
    "колонку", "стул", "чайник", "кроссовки", "зарядку",
]

DRIFT_POSITIVE = [
    "ну норм вообще, юзаю норм, ваще не жалею",
    "топчик, реально стоящая вещь имхо",
    "збс, всем советую брать не думая",
    "вроде неплохо, хотя были сомнения сначала но окей",
    "после месяца юзания – вполне ок, без нареканий короче",
]

DRIFT_NEGATIVE = [
    "ну такое... ожидал большего если честно",
    "вообще никак, зря потратился, печально",
    "разрекламировали а по факту фигня полная",
    "не то чтобы плохо но и не айс, скорее мимо",
    "продавец красиво описал а реально хлам пришел",
]

def make_review(label: int) -> str:
    core = random.choice(POSITIVE_CORE if label == 1 else NEGATIVE_CORE)
    product = random.choice(PRODUCTS)
    filler = random.choice(FILLERS)
    return f"{core} {product}.{filler}"


def make_drift_review(label: int)->str:
    core = random.choice(DRIFT_POSITIVE if label == 1 else DRIFT_NEGATIVE)
    product = random.choice(PRODUCTS)
    filler = random.choice(FILLERS)
    return f"{core} {product}.{filler}"

def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
def main():
    train_rows = []
    for _ in range(300):
        label = random.randint(0, 1)
        train_rows.append((make_review(label), label))

    write_csv(os.path.join(OUT_DIR,"train.csv"), train_rows, ["text", "label"])

    test_rows = []
    for _ in range(60):
        label = random.randint(0, 1)
        test_rows.append((make_review(label), label))
    write_csv(os.path.join(OUT_DIR,"test.csv"), test_rows, ["text", "label"])
    
    stream_rows = []
    row_id = 1
    for day in range(1, 11):
        for _ in range(20):
            label = random.randint(0, 1)
            if day <= 5:
                text = make_review(label)
            else:
                drift_share = min(0.9, 0.3+(day-6) *0.15)
                if random.random() < drift_share:
                    text = make_drift_review(label)
                else:
                    text = make_review(label)
            stream_rows.append((row_id, day, text, label))
            row_id += 1
    write_csv(
        os.path.join(OUT_DIR, "production_stream.csv"),
        stream_rows, 
        ["id", "day", "text","true_label"]
    )

    print(f"train.csv: {len(train_rows)} строк")
    print(f"test.csv: {len(test_rows)} строк")
    print(f"production_stream.csv: {len(stream_rows)} строк (10 дней, дрифт с 6-го дня)")


if __name__ == "__main__":
    main()
