DEFAULT_AGENTS = [
    {
        "name": "Inference Agent",
        "mission": "تحلیل متن کاربر و تولید نتیجه‌ی قابل‌فهم با confidence و توضیح عملیاتی.",
        "status": "active",
    },
    {
        "name": "Feedback Guard Agent",
        "mission": "امتیازدهی کیفیت فیدبک و جلوگیری از data poisoning و افت کیفیت مدل.",
        "status": "active",
    },
    {
        "name": "Retraining Agent",
        "mission": "ساخت batchهای آموزشی امن، آموزش shadow model و مقایسه با نسخه‌ی فعلی.",
        "status": "planned",
    },
    {
        "name": "Drift Monitor Agent",
        "mission": "پایش drift، تغییر distribution و هشدار افت عملکرد.",
        "status": "planned",
    },
    {
        "name": "Admin Insight Agent",
        "mission": "خلاصه‌سازی روندها، صف بازبینی و KPIهای حیاتی برای ادمین.",
        "status": "active",
    },
]
