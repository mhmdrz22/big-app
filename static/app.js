const analyzeBtn = document.getElementById('analyzeBtn');
const resultPanel = document.getElementById('resultPanel');
const feedbackPanel = document.getElementById('feedbackPanel');
const feedbackResult = document.getElementById('feedbackResult');
const verdictBanner = document.getElementById('verdictBanner');
const urgentBox = document.getElementById('urgentBox');

let latestPrediction = null;
let feedbackSent = false;

function renderGauge(score, level) {
  const gauge = document.getElementById('gauge');
  const pct = Math.round(score || 0);
  const colors = { elevated: '#c96b6b', uncertain: '#c9a25e', low: '#6fa287' };
  gauge.style.setProperty('--gauge', `${pct}%`);
  gauge.style.setProperty('--gauge-color', colors[level] || '#7ba7bc');
  const persianDigits = ['۰','۱','۲','۳','۴','۵','۶','۷','۸','۹'];
  gauge.innerHTML = `<span>${String(pct).split('').map(d => persianDigits[+d] || d).join('')}٪</span>`;
}

function bannerClass(level) {
  return {
    elevated: 'v-elevated',
    uncertain: 'v-uncertain',
    low: 'v-low',
    unsupported_language: 'v-support',
  }[level] || 'v-support';
}

function levelText(level) {
  return {
    elevated: 'نشانه‌هایی از فشار روانی دیده می‌شود',
    uncertain: 'مدل مطمئن نیست',
    low: 'نشانه‌ی واضحی دیده نشد',
    unsupported_language: 'فعلاً فقط پیام حمایتی',
  }[level] || 'نامشخص';
}

analyzeBtn?.addEventListener('click', async () => {
  const text = document.getElementById('userText').value.trim();
  feedbackResult.classList.add('hidden');
  feedbackSent = false;
  document.querySelectorAll('.direction-btn').forEach(b => b.disabled = false);

  const res = await fetch('/api/v1/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, locale: 'fa', store_text_consent: false })
  });
  const data = await res.json();

  if (!res.ok) {
    alert(data.detail || data.error || 'خطا در تحلیل');
    return;
  }

  latestPrediction = { ...data, text };
  renderGauge(data.stress_score != null ? data.stress_score : (data.probability * 100), data.triage_level);
  verdictBanner.className = 'verdict-banner ' + bannerClass(data.triage_level);
  document.getElementById('resultLabel').textContent = levelText(data.triage_level);
  if (data.stress_score != null) {
    document.getElementById('resultScore').textContent = `شدت استرس: ${data.stress_score.toFixed(0)}٪ (${data.stress_band})`;
  } else {
    document.getElementById('resultScore').textContent = '';
  }
  document.getElementById('resultProbability').textContent = data.probability != null
    ? `سطح اطمینان مدل: ${Math.round(data.probability * 100)}٪`
    : '';
  document.getElementById('recommendations').innerHTML = data.recommendations.map(item => `<li>${item}</li>`).join('');

  // Safety message must appear together with the result — never on a separate page.
  if (data.urgent_support_flag || data.triage_level === 'elevated') {
    urgentBox.textContent = data.urgent_support_message
      || 'اگر همین الان احساس خطر یا فشار شدید داری، لطفاً با یک نفر مورد اعتماد یا خدمات اضطراری صحبت کن.';
    urgentBox.classList.remove('hidden');
  } else {
    urgentBox.classList.add('hidden');
  }

  resultPanel.classList.remove('hidden');
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

async function sendVerdictFeedback(direction) {
  if (!latestPrediction || feedbackSent) return;
  const res = await fetch('/api/v1/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      assessment_id: latestPrediction.assessment_id,
      text: latestPrediction.text,
      direction: direction,
      note: '',
      consent_to_research: false
    })
  });

  const data = await res.json();
  feedbackResult.classList.remove('hidden');
  if (!res.ok) {
    feedbackResult.textContent = data.detail || 'خطا در ثبت بازخورد';
    return;
  }
  feedbackSent = true;
  feedbackResult.textContent = data.message;
  document.querySelectorAll('.direction-btn').forEach(b => b.disabled = true);
}

document.querySelectorAll('.direction-btn').forEach(btn => {
  btn.addEventListener('click', () => sendVerdictFeedback(btn.dataset.direction));
});
const unsureBtn = document.getElementById('feedbackUnsure');
if (unsureBtn) unsureBtn.addEventListener('click', () => sendVerdictFeedback('unsure'));

