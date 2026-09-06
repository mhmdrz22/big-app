from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from stressguard.features import build_feature_vector


class StressModel:
    """Hybrid ensemble: text TF-IDF branch + numeric LIWC branch."""

    def __init__(self, model_dir: Path, default_low: float = 0.35, default_high: float = 0.60, default_decision: float = 0.48):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "stress_model.joblib"
        self.metrics_path = self.model_dir / "metrics.json"
        self.metrics = json.loads(self.metrics_path.read_text(encoding="utf-8")) if self.metrics_path.exists() else {}
        self.model_name = self.metrics.get("selected_model", "unknown")
        self.thresholds = {
            "low": float(self.metrics.get("thresholds", {}).get("low", default_low)),
            "high": float(self.metrics.get("thresholds", {}).get("high", default_high)),
            "decision": float(self.metrics.get("thresholds", {}).get("decision", default_decision)),
        }
        self.artifact = None
        if self.model_path.exists():
            self.artifact = joblib.load(self.model_path)

    @staticmethod
    def _script_profile(text: str):
        latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        return {"latin": latin, "arabic": arabic}

    def _predict_proba(self, text: str) -> float:
        if self.artifact is None:
            raise RuntimeError("Model is not trained yet. Run train_model.py first.")

        members = self.artifact["members"]
        numeric = build_feature_vector(text).reshape(1, -1)

        probas = []
        for name, model in members.items():
            if name.startswith("text_"):
                probas.append(float(model.predict_proba([text])[0][1]))
            else:
                probas.append(float(model.predict_proba(numeric)[0][1]))

        # soft-voting ensemble (matches training-time selection)
        return float(np.mean(probas))

    @staticmethod
    def _score_to_band(score: float) -> str:
        if score >= 75:
            return "خیلی بالا"
        if score >= 55:
            return "بالا"
        if score >= 35:
            return "متوسط"
        return "پایین"

    @staticmethod
    def _score_label(score: float) -> str:
        band = StressModel._score_to_band(score)
        return f"نشانه‌های استرس: {band}"

    def predict_text(self, text: str):
        if self.artifact is None:
            raise RuntimeError("Model is not trained yet. Run train_model.py first.")

        script = self._script_profile(text)
        if script["arabic"] > script["latin"] and script["arabic"] >= 12:
            return {
                "analysis_mode": "support_only",
                "triage_level": "unsupported_language",
                "predicted_label": None,
                "probability": None,
                "stress_score": None,
                "stress_band": None,
                "label_text": "این مدل فعلی برای متن فارسی اعتبارسنجی نشده است.",
                "support_message": "فعلاً فقط پیام حمایتی امن نمایش داده می‌شود و نباید این خروجی به‌عنوان تحلیل معتبر مدل برای فارسی تفسیر شود.",
                "recommendations": [
                    "اگر نسخه فارسی می‌خواهی، باید مدل روی داده فارسی رضایت‌محور و برچسب‌خورده آموزش ببیند.",
                    "برای این ورودی، سامانه فقط نقش غربالگری حمایتی دارد و نه تصمیم‌گیری خودکار.",
                ],
                "disclaimer": "این خروجی تشخیص پزشکی نیست.",
                "urgent_support_flag": False,
                "urgent_support_message": None,
                "model_name": self.model_name,
            }

        probability = self._predict_proba(text)
        label = int(probability >= self.thresholds["decision"])
        stress_score = round(probability * 100, 1)
        stress_band = self._score_to_band(stress_score)

        if probability >= self.thresholds["high"]:
            triage_level = "elevated"
            label_text = f"نشانه‌های استرس: {stress_band}"
            support_message = "ممکن است متن نشانه‌هایی از فشار روانی را نشان دهد. بهتر است خروجی با زبان حمایتی و منابع کمکی نمایش داده شود."
        elif probability >= self.thresholds["low"]:
            triage_level = "uncertain"
            label_text = f"نشانه‌های استرس: {stress_band} — مدل مطمئن نیست"
            support_message = "برای این متن بهتر است سؤال‌های تکمیلی کوتاه یا بافت بیشتری از کاربر گرفته شود."
        else:
            triage_level = "low"
            label_text = f"نشانه‌های استرس: {stress_band}"
            support_message = "در متن فعلی سیگنال غالبی دیده نشد، اما این خروجی جای قضاوت تخصصی یا پزشکی را نمی‌گیرد."

        # Hard guardrail: crisis text must co-render urgent support with the
        # result — never on a separate page.
        crisis_patterns = [
            "kill myself", "kill my self", "suicide", "suicidal", "end my life",
            "want to die", "hurt myself", "self harm", "self-harm", "take my life",
            "no reason to live", "better off dead", "end it all", "dont want to live",
            "don't want to live", "wanna die", "want to kill",
        ]
        _lower = text.lower()
        urgent = any(p in _lower for p in crisis_patterns)
        urgent_message = (
            "اگر همین الان احساس خطر یا فشار شدید داری، لطفاً با یک نفر مورد اعتماد یا خدمات اضطراری محل زندگی‌ات صحبت کن. تو تنها نیستی."
            if urgent else None
        )

        recommendations = {
            "elevated": [
                "منابع حمایتی، check-in اختیاری و در صورت نیاز مسیر تماس اضطراری باید نمایش داده شود.",
                "اگر این سامانه در محیط واقعی استفاده شود، این مورد می‌تواند وارد صف بازبینی اولویت‌دار شود.",
            ],
            "uncertain": [
                "نتیجه باید غیرقطعی نمایش داده شود، نه به‌صورت بله/خیر مطلق.",
                "چند سؤال کوتاه اختیاری می‌تواند برای مرحله بعد کمک‌کننده باشد.",
            ],
            "low": [
                "در متن فعلی نشانه‌ی واضحی دیده نشد، اما این به معنای نبود قطعی فشار روانی نیست.",
                "در نسخه نهایی بهتر است کاربر امکان افزودن زمینه بیشتر را داشته باشد.",
            ],
        }

        return {
            "analysis_mode": "model_inference",
            "predicted_label": label,
            "probability": round(probability, 4),
            "stress_score": stress_score,
            "stress_band": stress_band,
            "triage_level": triage_level,
            "label_text": label_text,
            "support_message": support_message,
            "recommendations": recommendations[triage_level],
            "disclaimer": "این خروجی تشخیص پزشکی نیست و فقط برای غربالگری حمایتی استفاده می‌شود.",
            "urgent_support_flag": urgent,
            "urgent_support_message": urgent_message,
            "model_name": self.model_name,
        }
