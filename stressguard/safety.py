def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate_feedback(
    text: str,
    proposed_label: int,
    model_label: int,
    model_probability: float,
    note: str,
    duplicate_count: int,
    consent_to_research: bool,
    analysis_mode: str,
):
    reasons = []
    text_len = len(text.strip())
    note_len = len(note.strip())
    disagreement = int(proposed_label != model_label)
    uncertainty = 1 - abs(model_probability - 0.5) * 2

    length_score = clamp(text_len / 180, 0.0, 1.0)
    note_score = clamp(note_len / 60, 0.0, 1.0)
    novelty_score = 0.0 if duplicate_count >= 2 else 1.0 if duplicate_count == 0 else 0.4
    disagreement_penalty = 0.35 if disagreement and model_probability >= 0.82 else 0.0

    trust_score = clamp(
        0.35 * length_score
        + 0.20 * note_score
        + 0.25 * uncertainty
        + 0.20 * novelty_score
        - disagreement_penalty,
        0.0,
        1.0,
    )

    if text_len < 20:
        reasons.append("متن خیلی کوتاه است و برای بازآموزی قابل اعتماد نیست.")
    if duplicate_count >= 2:
        reasons.append("نمونه تکراری شناسایی شد.")
    if disagreement and model_probability >= 0.82:
        reasons.append("بازخورد مخالفِ پیش‌بینی با confidence بالا است؛ نیاز به بررسی انسانی دارد.")
    if disagreement and note_len < 20:
        reasons.append("برای بازخورد مخالف باید توضیح کافی ثبت شود.")
    if note_len >= 20:
        reasons.append("توضیح تکمیلی کافی ثبت شده است.")
    if uncertainty >= 0.5:
        reasons.append("این نمونه برای یادگیری مرزی مفید است چون مدل خیلی مطمئن نبوده است.")
    if not consent_to_research:
        reasons.append("رضایت پژوهشی برای استفاده آموزشی داده نشده است؛ این بازخورد فقط در قرنطینه می‌ماند.")
    if analysis_mode != "model_inference":
        reasons.append("این بازخورد مربوط به حالت بدون inference معتبر است و نباید وارد بازآموزی خودکار شود.")

    if text_len < 20 or duplicate_count >= 3:
        action = "reject"
    elif not consent_to_research or analysis_mode != "model_inference":
        action = "review_required"
    elif disagreement and model_probability >= 0.82 and note_len < 20:
        action = "review_required"
    elif trust_score >= 0.78 and note_len >= 10 and duplicate_count == 0:
        action = "approved_limited_weight"
    else:
        # Agreement feedback with low trust is still human signal worth reviewing,
        # not garbage — only truly invalid input (short/duplicate) is rejected.
        action = "review_required"

    sample_weight = 0.0 if action != "approved_limited_weight" else round(clamp(0.10 + trust_score * 0.25, 0.10, 0.35), 3)

    return {
        "trust_score": round(trust_score, 4),
        "action": action,
        "sample_weight": sample_weight,
        "reasons": reasons,
    }
