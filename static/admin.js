const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
};

let charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function drawChart(id, config) {
  destroyChart(id);
  const ctx = $(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx, config);
}

// ── Tab switching ────────────────────────────────────────────────────
document.querySelectorAll('.admin-tabs .tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.admin-tabs .tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    $('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'technical') loadTechnicalTab();
  });
});

// ── Executive tab ────────────────────────────────────────────────────
async function loadExecutiveTab() {
  try {
    const [summary, series, feedbackSeries] = await Promise.all([
      api('/api/v1/admin/summary'),
      api('/api/v1/admin/assessment-series'),
      api('/api/v1/admin/feedback-series'),
    ]);
    renderKPIs(summary);
    drawAssessmentChart(series.items || []);
    drawFeedbackChart(feedbackSeries.items || []);
    renderFeedbackQueue(summary.feedback_queue || []);
    renderAgents(summary.agents || []);
  } catch (e) { console.error(e); }
}

function renderKPIs(summary) {
  const store = summary.store || {};
  const metrics = summary.training_metrics || {};
  const best = (metrics.candidates || []).find(c => c.model === metrics.selected_model) || {};
  const kpis = [
    { label: 'ارزیابی‌ها', value: store.assessment_count || 0, icon: '📊' },
    { label: 'بازخوردها', value: store.feedback_count || 0, icon: '💬' },
    { label: 'F1 مدل', value: best.f1 ? (best.f1 * 100).toFixed(1) + '٪' : '—', icon: '🎯' },
    { label: 'AUC', value: best.roc_auc ? best.roc_auc.toFixed(3) : '—', icon: '📈' },
    { label: 'مدل‌های ثبت‌شده', value: store.model_count || 0, icon: '🤖' },
    { label: 'رانش', value: summary.drift_report?.drift_level || '—', icon: '⚠️' },
  ];
  $('kpiGrid').innerHTML = kpis.map(k => `
    <div class="kpi-card">
      <div class="kpi-icon">${k.icon}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-label">${k.label}</div>
    </div>
  `).join('');
}

function drawAssessmentChart(items) {
  const labels = items.map(i => new Date(i.created_at).toLocaleDateString('fa-IR'));
  const data = items.map(i => i.probability ? i.probability * 100 : 0);
  drawChart('assessmentChart', {
    type: 'line',
    data: { labels, datasets: [{ label: 'سطح استرس (٪)', data, borderColor: '#2563eb', tension: 0.3, fill: true, backgroundColor: 'rgba(37,99,235,0.1)' }] },
    options: { responsive: true, scales: { y: { min: 0, max: 100 } } },
  });
}

function drawFeedbackChart(items) {
  const agree = items.filter(i => i.user_agrees_with_result === true).length;
  const disagree = items.filter(i => i.user_agrees_with_result === false).length;
  const unsure = items.filter(i => i.user_agrees_with_result === null).length;
  drawChart('feedbackChart', {
    type: 'doughnut',
    data: { labels: ['موافق', 'مخالف', 'نامطمئن'], datasets: [{ data: [agree, disagree, unsure], backgroundColor: ['#10b981', '#ef4444', '#f59e0b'] }] },
    options: { responsive: true },
  });
}

function renderFeedbackQueue(items) {
  $('feedbackQueue').innerHTML = items.length ? items.map(it => `
    <div class="queue-item">
      <div class="queue-text">${it.text_excerpt || '—'}</div>
      <div class="queue-meta">
        <span class="badge ${it.direction}">${it.direction || '—'}</span>
        <span class="badge">${it.review_state}</span>
        <span>وزن: ${it.sample_weight?.toFixed(2) || '—'}</span>
      </div>
    </div>
  `).join('') : '<p class="empty">صف بازخورد خالی است.</p>';
}

function renderAgents(agents) {
  $('agentList').innerHTML = agents.map(a => `
    <div class="agent-card">
      <strong>${a.name}</strong>
      <span class="agent-role">${a.role}</span>
      <p>${a.description}</p>
    </div>
  `).join('');
}

// ── Technical tab ────────────────────────────────────────────────────
async function loadTechnicalTab() {
  try {
    const [summary, runs, registry] = await Promise.all([
      api('/api/v1/admin/summary'),
      api('/api/v1/admin/retraining/runs'),
      api('/api/v1/admin/model-registry'),
    ]);
    drawConfusionMatrix(summary.training_metrics);
    drawDriftChart(summary.drift_report);
    drawWeightChart(summary.store);
    renderModelRegistry(registry.items || []);
    renderRetrainingRuns(runs.items || []);
  } catch (e) { console.error(e); }
}

function drawConfusionMatrix(metrics) {
  const best = (metrics?.candidates || []).find(c => c.model === metrics?.selected_model) || {};
  const cm = best.confusion_matrix || [[0, 0], [0, 0]];
  drawChart('confusionMatrixChart', {
    type: 'bar',
    data: {
      labels: ['TN', 'FP', 'FN', 'TP'],
      datasets: [{ label: 'تعداد', data: [cm[0][0], cm[0][1], cm[1][0], cm[1][1]], backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#3b82f6'] }],
    },
    options: { responsive: true, plugins: { title: { display: true, text: `Confusion Matrix — ${metrics?.selected_model || 'N/A'}` } } },
  });
}

function drawDriftChart(drift) {
  const metrics = drift?.metrics || {};
  const labels = Object.keys(metrics);
  const values = Object.values(metrics);
  drawChart('driftChart', {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Drift Metric', data: values, backgroundColor: values.map(v => v > 0.2 ? '#ef4444' : v > 0.1 ? '#f59e0b' : '#10b981') }] },
    options: { responsive: true, scales: { y: { min: 0 } } },
  });
}

function drawWeightChart(store) {
  drawChart('weightChart', {
    type: 'pie',
    data: {
      labels: ['تأیید شده', 'در انتظار', 'رد شده'],
      datasets: [{ data: [store.approved_feedback || 0, store.pending_feedback || 0, store.rejected_feedback || 0], backgroundColor: ['#10b981', '#f59e0b', '#ef4444'] }],
    },
    options: { responsive: true },
  });
}

function renderModelRegistry(items) {
  $('modelRegistryList').innerHTML = items.length ? items.map(m => `
    <div class="registry-item">
      <strong>${m.version_name}</strong>
      <span class="badge">${m.lifecycle_stage}</span>
      <span>F1: ${(m.f1 * 100).toFixed(1)}٪</span>
    </div>
  `).join('') : '<p class="empty">مدلی ثبت نشده است.</p>';
}

function renderRetrainingRuns(items) {
  $('retrainingRuns').innerHTML = items.length ? items.map(r => `
    <div class="run-item">
      <strong>${r.run_name}</strong>
      <span class="badge ${r.status}">${r.status}</span>
      <span>${r.approved_feedback_count} بازخورد</span>
      <span>${new Date(r.created_at).toLocaleString('fa-IR')}</span>
    </div>
  `).join('') : '<p class="empty">هیچ اجرای بازآموزی وجود ندارد.</p>';
}

// ── Boot ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadExecutiveTab();
  $('loginForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const res = await api('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ email: $('adminEmail').value, password: $('adminPassword').value }) });
      localStorage.setItem('aramnegar_token', res.access_token);
      location.reload();
    } catch (err) { $('loginError').textContent = err.message; }
  });
});
