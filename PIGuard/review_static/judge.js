const $ = (selector) => document.querySelector(selector);
const state = { items: [], total: 0, offset: 0, limit: 12, selected: null, request: 0, running: false, jobActive: false };
const number = value => new Intl.NumberFormat('pt-BR').format(value);
let toastTimer;

function toast(message) {
  const el = $('#toast'); el.textContent = message; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 4000);
}
async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Não foi possível concluir a operação.');
  return body;
}
async function loadMeta() {
  const meta = await api('/api/judge/meta');
  const dataset = $('#dataset'), old = dataset.value;
  dataset.replaceChildren(new Option('Todos os datasets', ''));
  meta.datasets.forEach(name => dataset.add(new Option(name, name)));
  dataset.value = old;
  if (!$('#model').value) $('#model').value = meta.model;
  $('#key-status').textContent = meta.configured
    ? 'Chave da API configurada no servidor · a avaliação utiliza chamadas pagas.'
    : 'Chave ausente: defina TYPESAFE_API_KEY em .env na raiz do projeto.';
  const stats = meta.stats;
  $('#stat-total').textContent = number(stats.total);
  $('#stat-reviewed').textContent = number(stats.reviewed);
  $('#stat-malicious').textContent = number(stats.malicious);
  $('#stat-benign').textContent = number(stats['non-malicious']);
  $('#reviewed-count').textContent = `${number(stats.reviewed)} avaliados`;
  $('#total-count').textContent = `${number(stats.total)} no total`;
  const percent = stats.total ? Math.round(stats.reviewed * 100 / stats.total) : 0;
  $('#progress-percent').textContent = `${percent}%`;
  $('#progress-bar').style.width = `${percent}%`;
}
function query() {
  const params = new URLSearchParams({ offset: state.offset, limit: state.limit });
  for (const key of ['dataset', 'status', 'label', 'search']) if ($(`#${key}`).value.trim()) params.set(key, $(`#${key}`).value.trim());
  return params;
}
async function loadPage(preferredId) {
  const request = ++state.request;
  try {
    const page = await api(`/api/judge/prompts?${query()}`);
    if (request !== state.request) return;
    state.items = page.items; state.total = page.total;
    if (!state.items.length && state.offset > 0) { state.offset = Math.max(0, state.offset - state.limit); return loadPage(preferredId); }
    selectPrompt(state.items.find(item => item.id === preferredId) || state.items[0] || null);
    renderList();
  } catch (error) { toast(error.message); }
}
function renderList() {
  const list = $('#prompt-list'); list.replaceChildren();
  $('#queue-count').textContent = number(state.total);
  $('#list-range').textContent = state.total ? `${number(state.offset + 1)}–${number(state.offset + state.items.length)} de ${number(state.total)}` : '0 resultados';
  $('#page-info').textContent = state.total ? `${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)}` : '0 / 0';
  $('#previous-page').disabled = state.offset === 0 || state.running;
  $('#next-page').disabled = state.offset + state.limit >= state.total || state.running;
  if (!state.items.length) { const empty = document.createElement('p'); empty.className = 'empty-state'; empty.style.padding = '48px 16px'; empty.textContent = 'Nenhum prompt encontrado com esses filtros.'; list.append(empty); }
  for (const item of state.items) {
    const button = document.createElement('button'); button.type = 'button';
    button.className = `prompt-item${state.selected?.id === item.id ? ' active' : ''}`;
    const top = document.createElement('div'); top.className = 'item-top';
    const source = document.createElement('span'); source.className = 'item-source'; source.textContent = item.source || item.dataset;
    const status = document.createElement('span'); status.className = `item-state${item.review ? ` done ${item.review.verdict}` : ''}`;
    status.textContent = item.review ? ({ malicious: '● Malicious', 'non-malicious': '● Non-malicious', uncertain: '● Incerto' })[item.review.verdict] : '○ Pendente';
    top.append(source, status);
    const text = document.createElement('div'); text.className = 'item-text'; text.textContent = item.prompt;
    const id = document.createElement('div'); id.className = 'item-id'; id.textContent = `#${item.index + 1} · ${item.dataset}`;
    button.append(top, text, id);
    button.addEventListener('click', () => { selectPrompt(item); renderList(); });
    list.append(button);
  }
}
function selectPrompt(item) {
  state.selected = item;
  $('#detail-empty').hidden = !!item;
  $('#detail-content').hidden = !item;
  if (!item) return;
  $('#prompt-dataset').textContent = item.dataset;
  $('#prompt-position').textContent = `Prompt #${item.index + 1}`;
  $('#prompt-text').textContent = item.prompt;
  $('#original-label').textContent = item.original_label === 0 ? '0 · Non-malicious' : item.original_label === 1 ? '1 · Malicious' : 'Não informado';
  $('#original-category').textContent = item.original_category || 'Não informada';
  $('#original-source').textContent = item.source || item.dataset.replace(/\.json$/, '');
  const review = item.review;
  $('#review-state').textContent = review ? '✓ Avaliado pela IA' : '○ Não avaliado';
  $('#review-state').classList.toggle('reviewed', !!review);
  $('#judge-result').hidden = !review;
  $('#no-result').hidden = !!review;
  $('#run-single').textContent = review ? 'Reavaliar com Jev ↻' : 'Avaliar com Jev →';
  if (review) {
    $('#result-model').textContent = `${review.model} · ${review.prompt_version} · ${new Date(review.updated_at).toLocaleString('pt-BR')}`;
    $('#result-verdict').textContent = ({ malicious: '✕ Malicious', 'non-malicious': '✓ Non-malicious', uncertain: '? Incerto' })[review.verdict];
    $('#result-verdict').className = `result-verdict ${review.verdict}`;
    $('#result-confidence').textContent = `Confiança: ${Math.round(review.confidence * 100)}%`;
    $('#result-probabilities').replaceChildren();
    for (const [label, probability] of Object.entries(review.probabilities || {})) {
      const row = document.createElement('div'); row.className = 'probability-row';
      const title = document.createElement('span'); title.textContent = label;
      const track = document.createElement('div'); track.className = 'probability-track';
      const fill = document.createElement('div'); fill.style.width = `${Math.round(probability * 100)}%`; track.append(fill);
      const percent = document.createElement('strong'); percent.textContent = `${Math.round(probability * 100)}%`;
      row.append(title, track, percent); $('#result-probabilities').append(row);
    }
    $('#result-categories').replaceChildren();
    if (!review.categories.length) $('#result-categories').textContent = 'Nenhuma categoria identificada';
    for (const category of review.categories) {
      const chip = document.createElement('span'); chip.className = 'result-chip';
      chip.textContent = `${category} · ${Math.round((review.category_probabilities?.[category] || 0) * 100)}%`;
      $('#result-categories').append(chip);
    }
  }
  $('#run-status').textContent = review ? 'Reavaliar substituirá apenas esta avaliação da IA.' : 'O texto do prompt será enviado à API configurada.';
}
async function run(ids, overwrite) {
  if (state.running || state.jobActive || !ids.length) return;
  state.running = true;
  $('#run-single').disabled = true; $('#run-batch').disabled = true;
  $('#run-status').textContent = `Avaliando ${ids.length} prompt${ids.length === 1 ? '' : 's'}... Aguarde.`;
  renderList();
  try {
    const result = await api('/api/judge/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, model: $('#model').value.trim(), overwrite }) });
    const completed = Object.keys(result.results).length;
    const failed = Object.keys(result.errors).length;
    toast(`${completed} avaliado(s), ${result.skipped.length} ignorado(s), ${failed} erro(s).`);
    if (failed) $('#run-status').textContent = Object.values(result.errors)[0];
    await loadMeta();
    await loadPage(state.selected?.id);
    if (failed) $('#run-status').textContent = Object.values(result.errors)[0];
  } catch (error) { toast(error.message); $('#run-status').textContent = error.message; }
  finally { state.running = false; updateJobButtons(); renderList(); }
}
$('#run-single').addEventListener('click', () => { if (state.selected) run([state.selected.id], true); });
$('#run-batch').addEventListener('click', () => {
  const ids = state.items.filter(item => !item.review).slice(0, 10).map(item => item.id);
  if (!ids.length) return toast('Não há prompts pendentes nesta página. Avance ou filtre por “Não avaliados”.');
  run(ids, false);
});
function updateJobButtons() {
  $('#auto-start').disabled = state.jobActive || state.running;
  $('#auto-stop').hidden = !state.jobActive;
  $('#run-single').disabled = state.jobActive || state.running;
  $('#run-batch').disabled = state.jobActive || state.running;
}
function showJob(job) {
  state.jobActive = job.active;
  updateJobButtons();
  const total = job.requested || 0;
  $('#auto-progress').textContent = total
    ? `${job.message} · ${number(job.completed)} de ${number(total)} salvos · ${number(job.skipped)} ignorados · ${number(job.failed)} erros${job.last_error ? ` · ${job.last_error}` : ''}`
    : job.message;
}
let jobPolling = false;
async function pollJob() {
  if (jobPolling) return;
  jobPolling = true;
  try {
    const wasActive = state.jobActive;
    const job = await api('/api/judge/job');
    showJob(job);
    if (wasActive && !job.active) { await loadMeta(); await loadPage(state.selected?.id); }
  } catch (error) { toast(error.message); }
  finally { jobPolling = false; }
}
$('#auto-start').addEventListener('click', async () => {
  const maxRequests = Number($('#auto-limit').value);
  if (!Number.isInteger(maxRequests) || maxRequests < 1 || maxRequests > 80000) return toast('Escolha entre 1 e 80.000 chamadas.');
  const dataset = $('#dataset').value, label = $('#label').value;
  const scope = `${dataset || 'todos os datasets'}${label ? `, rótulo original ${label}` : ''}`;
  if (!window.confirm(`Iniciar até ${maxRequests} chamadas pagas à API TypeSafe Jev para ${scope}? Resultados anteriores não serão substituídos.`)) return;
  $('#auto-start').disabled = true;
  try {
    const job = await api('/api/judge/job/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, label, max_requests: maxRequests, model: $('#model').value.trim() }) });
    showJob(job);
  } catch (error) { toast(error.message); $('#auto-progress').textContent = error.message; }
  finally { updateJobButtons(); }
});
$('#auto-stop').addEventListener('click', async () => {
  try {
    showJob(await api('/api/judge/job/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }));
  } catch (error) { toast(error.message); }
});
let searchTimer;
for (const key of ['dataset', 'status', 'label']) $(`#${key}`).addEventListener('change', () => { state.offset = 0; loadPage(); });
$('#search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadPage(); }, 300); });
$('#previous-page').addEventListener('click', () => { state.offset -= state.limit; loadPage(); });
$('#next-page').addEventListener('click', () => { state.offset += state.limit; loadPage(); });
(async () => { try { await loadMeta(); await loadPage(); await pollJob(); } catch (error) { toast(error.message); } })();
setInterval(() => { if (state.jobActive) pollJob(); }, 2500);
