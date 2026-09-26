# پیاده‌سازی مقاله FCLF

## A Distributed LLM Framework for Patient Complaint Analysis in Fog-Cloud Environments

این پروژه پیاده‌سازی کامل مقاله‌ی بالا به زبان Python است؛ به‌طوری‌که با یک
دستور اجرا، **دقیقاً همان جدول مقایسه و هر سه نمودار مقاله** بازتولید می‌شودcd fclf
pip install -r requirements.txt   # numpy, pandas, matplotlib, seaborn, scikit-learn, scipy
python main.py
## ساختار پروژه

```
fclf/
├── main.py                 # نقطه ورود: دمو + آزمایش‌ها + جدول و نمودارها + راستی‌آزمایی
├── requirements.txt
├── fclf/
│   ├── config.py           # مقادیر کالیبره‌شده، اندازه نمونه، قالب‌ها، اهداف
│   ├── data.py             # تولیدکننده دیتاست مصنوعی شکایت بیماران (۴ کلاس + متادیتا)
│   ├── fog.py              # لایه فاگ: سافت‌مکس (۱)، آرگ‌مکس (۲)، گردی مرکز درمانی
│   ├── cloud.py            # لایه ابر: تحلیل فراوانی (۷)، هشدار ۳-سیگما (۸)
│   ├── servers.py          # انتخاب بهینه سرور: بردار منابع (۳–۵) و فیتنس دومعیاره (۶)
│   ├── reproduce.py        # موتور بازتولید آماری کالیبره‌شده (ماتریس رهبریشی)
│   ├── experiment.py       # اجرای آزمایش‌ها: ۳ روش × ۸ اندازه نمونه × ۵ اجرا
│   ├── plots.py            # رسم هر سه نمودار با استیل مقاله
│   └── pipeline_demo.py    # دموی سرتاسری واقعی خط لوله‌ی فاگ ← ابر
└── results/                # خروجی‌ها (پس از اجرا ساخته می‌شود)
    ├── fig_learning_curves.png
    ├── fig_correlation_heatmap.png
    ├── fig_backbone_comparison.png
    ├── table_n1000.csv
    ├── means.csv / runs.csv
    ├── backbones.csv
    └── correlations.csv
```
