"""Build the downloadable FCLF complaint dataset (CSV + Excel + data card).

Run:
    python make_dataset.py

Outputs (in ./dataset):
    fclf_complaints_train.csv   - 3000 labeled complaints (train split)
    fclf_complaints_test.csv    - 1000 labeled complaints (test split)
    fclf_complaints_full.csv    - all 4000 rows with a `split` column
    fclf_complaints.xlsx        - same data as Excel (Train/Test/LabelMap sheets)
    DATASET_CARD.md             - Persian data card describing the dataset
"""

import os

import pandas as pd

from fclf.config import CATEGORIES
from fclf.data import generate_dataset

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
N_TRAIN, N_TEST = 3000, 1000
LABEL_DESCRIPTIONS = [
    "Complaints about staff behavior or attitude",
    "Complaints about misdiagnosis or treatment errors",
    "Complaints about facilities, appointments, or medication",
    "Complaints about staff unaccountability",
]


def build_split(n, seed, split_name, id_start):
    df = generate_dataset(n, seed=seed, n_centers=5, n_weeks=12, noise=0.15)
    out = pd.DataFrame({
        "patient_id": [f"P{id_start + i:06d}" for i in range(n)],
        "complaint_text": df["text"],
        "label": df["true_label"].astype(int),
        "category": df["label_name"],
        "center_id": df["center_id"].astype(int),
        "week": df["week"].astype(int),
        "department": df["department"],
        "split": split_name,
    })
    return out


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    train = build_split(N_TRAIN, seed=100, split_name="train", id_start=1)
    test = build_split(N_TEST, seed=200, split_name="test", id_start=N_TRAIN + 1)
    full = pd.concat([train, test], ignore_index=True)

    # ------------------------------------------------------------- CSV files
    p_train = os.path.join(DATASET_DIR, "fclf_complaints_train.csv")
    p_test = os.path.join(DATASET_DIR, "fclf_complaints_test.csv")
    p_full = os.path.join(DATASET_DIR, "fclf_complaints_full.csv")
    train.to_csv(p_train, index=False, encoding="utf-8-sig")
    test.to_csv(p_test, index=False, encoding="utf-8-sig")
    full.to_csv(p_full, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------- Excel file
    label_map = pd.DataFrame({
        "label": [0, 1, 2, 3],
        "category": CATEGORIES,
        "description": LABEL_DESCRIPTIONS,
    })
    p_xlsx = os.path.join(DATASET_DIR, "fclf_complaints.xlsx")
    with pd.ExcelWriter(p_xlsx, engine="openpyxl") as writer:
        train.to_excel(writer, sheet_name="Train", index=False)
        test.to_excel(writer, sheet_name="Test", index=False)
        label_map.to_excel(writer, sheet_name="LabelMap", index=False)
        for sheet in ("Train", "Test"):
            ws = writer.sheets[sheet]
            ws.column_dimensions["B"].width = 90
            ws.column_dimensions["D"].width = 28
            ws.column_dimensions["G"].width = 12
            ws.column_dimensions["H"].width = 8

    # ------------------------------------------------------------- data card
    dist = full.groupby(["split", "category"]).size().unstack(fill_value=0)
    card_lines = [
        "# کارت معرفی دیتاست شکایت بیماران FCLF",
        "",
        "دیتاست مصنوعی (سنتتیک) شکایت بیماران در ۴ دسته‌ی مقاله‌ی",
        "«A Distributed LLM Framework for Patient Complaint Analysis in Fog-Cloud Environments»",
        "بدون هیچ داده‌ی واقعی بیمار — مناسب آموزش و ارزیابی مدل‌های طبقه‌بندی شکایت.",
        "",
        "## فایل‌ها",
        "",
        "| فایل | تعداد ردیف | توضیح |",
        "|---|---|---|",
        "| `fclf_complaints_train.csv` | 3000 | مجموعه‌ی آموزشی |",
        "| `fclf_complaints_test.csv` | 1000 | مجموعه‌ی آزمون |",
        "| `fclf_complaints_full.csv` | 4000 | همه‌ی داده‌ها + ستون `split` |",
        "| `fclf_complaints.xlsx` | 4000 | همین داده‌ها در اکسل (شیت‌های Train/Test/LabelMap) |",
        "",
        "## ستون‌ها",
        "",
        "| ستون | نوع | توضیح |",
        "|---|---|---|",
        "| `patient_id` | متن | شناسه‌ی یکتای بیمار (P000001 تا P004000) |",
        "| `complaint_text` | متن | متن انگلیسی شکایت بیمار |",
        "| `label` | عدد ۰ تا ۳ | برچسب عددی دسته |",
        "| `category` | متن | نام دسته (۴ دسته‌ی Table 1 مقاله) |",
        "| `center_id` | عدد ۱ تا ۵ | شناسه‌ی مرکز درمانی (متادیتای غیرحساس) |",
        "| `week` | عدد ۱ تا ۱۲ | شماره‌ی هفته (برچسب زمانی) |",
        "| `department` | متن | دپارتمان: Emergency/Surgery/Internal/Pediatrics/Radiology |",
        "| `split` | متن | train یا test (فقط در فایل full) |",
        "",
        "## نگاشت برچسب‌ها",
        "",
        "| label | category | توضیح |",
        "|---|---|---|",
    ]
    for i, (cat, desc) in enumerate(zip(CATEGORIES, LABEL_DESCRIPTIONS)):
        card_lines.append(f"| {i} | {cat} | {desc} |")
    card_lines += [
        "",
        "## توزیع کلاس‌ها (تعداد)",
        "",
        "| split | " + " | ".join(CATEGORIES) + " | جمع |",
        "|" + "|".join(["---"] * 6) + "|",
    ]
    for split in ("train", "test"):
        row = dist.loc[split]
        card_lines.append(
            f"| {split} | " + " | ".join(str(int(row[c])) for c in CATEGORIES)
            + f" | {int(row.sum())} |"
        )
    card_lines += [
        "",
        "## روش تولید",
        "",
        "- ۳۰ قالب جمله‌ی پایه برای هر دسته (۱۲۰ قالب) + جمله‌ی پایانی تصادفی.",
        "- حدود ۱۵٪ نمونه‌ها یک جمله‌ی انحرافی (distractor) از دسته‌ی دیگر دارند",
        "  تا بازتاب واقعیت باشند (شکایت‌های واقعی اغلب چندوجهی‌اند).",
        "- توزیع طبیعی نامتوازن کلاس‌ها: ۳۰٪ / ۲۷٪ / ۲۳٪ / ۲۰٪.",
        "- کاملاً قطعی با seed ثابت — با اجرای `python make_dataset.py` عیناً بازتولید می‌شود.",
        "",
        "## مثال استفاده (Python)",
        "",
        "```python",
        "import pandas as pd",
        "train = pd.read_csv('fclf_complaints_train.csv')",
        "test = pd.read_csv('fclf_complaints_test.csv')",
        "from sklearn.feature_extraction.text import TfidfVectorizer",
        "from sklearn.linear_model import LogisticRegression",
        "Xtr = TfidfVectorizer().fit_transform(train['complaint_text'])",
        "Xte = TfidfVectorizer().fit(train['complaint_text']).transform(test['complaint_text'])",
        "clf = LogisticRegression(max_iter=1000).fit(Xtr, train['label'])",
        "print('test accuracy:', clf.score(Xte, test['label']))",
        "```",
    ]
    with open(os.path.join(DATASET_DIR, "DATASET_CARD.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines) + "\n")

    # ------------------------------------------------------------- report
    print(f"train: {len(train)} rows -> {p_train}")
    print(f"test : {len(test)} rows -> {p_test}")
    print(f"full : {len(full)} rows -> {p_full}")
    print(f"xlsx : -> {p_xlsx}")
    print("\nLabel distribution:\n", dist.to_string())
    print("\nSample row:")
    print(full.iloc[0].to_string())
    print("\nData card ->", os.path.join(DATASET_DIR, "DATASET_CARD.md"))


if __name__ == "__main__":
    main()
