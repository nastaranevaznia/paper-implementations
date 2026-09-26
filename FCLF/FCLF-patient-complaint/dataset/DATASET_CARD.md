# کارت معرفی دیتاست شکایت بیماران FCLF

دیتاست مصنوعی (سنتتیک) شکایت بیماران در ۴ دسته‌ی مقاله‌ی
«A Distributed LLM Framework for Patient Complaint Analysis in Fog-Cloud Environments»
بدون هیچ داده‌ی واقعی بیمار — مناسب آموزش و ارزیابی مدل‌های طبقه‌بندی شکایت.

## فایل‌ها

| فایل | تعداد ردیف | توضیح |
|---|---|---|
| `fclf_complaints_train.csv` | 3000 | مجموعه‌ی آموزشی |
| `fclf_complaints_test.csv` | 1000 | مجموعه‌ی آزمون |
| `fclf_complaints_full.csv` | 4000 | همه‌ی داده‌ها + ستون `split` |
| `fclf_complaints.xlsx` | 4000 | همین داده‌ها در اکسل (شیت‌های Train/Test/LabelMap) |

## ستون‌ها

| ستون | نوع | توضیح |
|---|---|---|
| `patient_id` | متن | شناسه‌ی یکتای بیمار (P000001 تا P004000) |
| `complaint_text` | متن | متن انگلیسی شکایت بیمار |
| `label` | عدد ۰ تا ۳ | برچسب عددی دسته |
| `category` | متن | نام دسته (۴ دسته‌ی Table 1 مقاله) |
| `center_id` | عدد ۱ تا ۵ | شناسه‌ی مرکز درمانی (متادیتای غیرحساس) |
| `week` | عدد ۱ تا ۱۲ | شماره‌ی هفته (برچسب زمانی) |
| `department` | متن | دپارتمان: Emergency/Surgery/Internal/Pediatrics/Radiology |
| `split` | متن | train یا test (فقط در فایل full) |

## نگاشت برچسب‌ها

| label | category | توضیح |
|---|---|---|
| 0 | Communication problems | Complaints about staff behavior or attitude |
| 1 | Diagnosis/Treatment issues | Complaints about misdiagnosis or treatment errors |
| 2 | Management problems | Complaints about facilities, appointments, or medication |
| 3 | Responsibility concerns | Complaints about staff unaccountability |

## توزیع کلاس‌ها (تعداد)

| split | Communication problems | Diagnosis/Treatment issues | Management problems | Responsibility concerns | جمع |
|---|---|---|---|---|---|
| train | 924 | 802 | 702 | 572 | 3000 |
| test | 286 | 274 | 232 | 208 | 1000 |

## روش تولید

- ۳۰ قالب جمله‌ی پایه برای هر دسته (۱۲۰ قالب) + جمله‌ی پایانی تصادفی.
- حدود ۱۵٪ نمونه‌ها یک جمله‌ی انحرافی (distractor) از دسته‌ی دیگر دارند
  تا بازتاب واقعیت باشند (شکایت‌های واقعی اغلب چندوجهی‌اند).
- توزیع طبیعی نامتوازن کلاس‌ها: ۳۰٪ / ۲۷٪ / ۲۳٪ / ۲۰٪.
- کاملاً قطعی با seed ثابت — با اجرای `python make_dataset.py` عیناً بازتولید می‌شود.

## مثال استفاده (Python)

```python
import pandas as pd
train = pd.read_csv('fclf_complaints_train.csv')
test = pd.read_csv('fclf_complaints_test.csv')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
Xtr = TfidfVectorizer().fit_transform(train['complaint_text'])
Xte = TfidfVectorizer().fit(train['complaint_text']).transform(test['complaint_text'])
clf = LogisticRegression(max_iter=1000).fit(Xtr, train['label'])
print('test accuracy:', clf.score(Xte, test['label']))
```
