const $ = (selector) => document.querySelector(selector);
const state = { items: [], total: 0, offset: 0, limit: 12, selected: null, verdict: null, categories: new Set(), categoryOptions: [], request: 0 };
const number = (value) => new Intl.NumberFormat('pt-BR').format(value);
let toastTimer;

function toast(message) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

async function api(url, options) {
  if (options?.method?.toUpperCase() === 'POST') {
    const sessionResponse = await fetch('/api/session');
    if (!sessionResponse.ok) throw new Error('Sessão expirada. Atualize a página e entre novamente.');
    const session = await sessionResponse.json();
    options.headers = { ...options.headers, 'X-CSRF-Token': session.csrf };
  }
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Não foi possível concluir a operação.');
  return body;
}

async function loadMeta() {
  const meta = await api('/api/meta');
  state.categoryOptions = meta.categories;
  const dataset = $('#dataset');
  const previous = dataset.value;
  dataset.replaceChildren(new Option('Todos os datasets', ''));
  meta.datasets.forEach(name => dataset.add(new Option(name, name)));
  dataset.value = previous;
  const stats = meta.stats;
  $('#stat-total').textContent = number(stats.total);
  $('#stat-reviewed').textContent = number(stats.reviewed);
  $('#stat-malicious').textContent = number(stats.malicious);
  $('#stat-benign').textContent = number(stats['non-malicious']);
  $('#reviewed-count').textContent = `${number(stats.reviewed)} revisados`;
  $('#total-count').textContent = `${number(stats.total)} no total`;
  const percent = stats.total ? Math.round(stats.reviewed * 100 / stats.total) : 0;
  $('#progress-percent').textContent = `${percent}%`;
  $('#progress-bar').style.width = `${percent}%`;
}

function filters() {
  const params = new URLSearchParams({ offset: state.offset, limit: state.limit });
  for (const key of ['dataset', 'status', 'label', 'search']) {
    if ($(`#${key}`).value.trim()) params.set(key, $(`#${key}`).value.trim());
  }
  return params;
}

async function loadPage(preferredId) {
  const request = ++state.request;
  try {
    const data = await api(`/api/prompts?${filters()}`);
    if (request !== state.request) return;
    state.items = data.items;
    state.total = data.total;
    if (!data.items.length && state.offset > 0) {
      state.offset = Math.max(0, state.offset - state.limit);
      return loadPage(preferredId);
    }
    const current = state.items.find(item => item.id === preferredId) || state.items[0] || null;
    selectPrompt(current);
    renderList();
  } catch (error) { toast(error.message); }
}

function renderList() {
  const list = $('#prompt-list');
  list.replaceChildren();
  $('#queue-count').textContent = number(state.total);
  $('#list-range').textContent = state.total ? `${number(state.offset + 1)}–${number(state.offset + state.items.length)} de ${number(state.total)}` : '0 resultados';
  $('#page-info').textContent = state.total ? `${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)}` : '0 / 0';
  $('#previous-page').disabled = state.offset === 0;
  $('#next-page').disabled = state.offset + state.limit >= state.total;
  if (!state.items.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.style.padding = '48px 16px';
    empty.textContent = 'Nenhum prompt encontrado com esses filtros.';
    list.append(empty);
  }
  for (const item of state.items) {
    const button = document.createElement('button');
    button.type = 'button';
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
  const review = item.review;
  state.verdict = review?.verdict || null;
  state.categories = new Set(review?.categories || []);
  $('#notes').value = review?.notes || '';
  $('#prompt-dataset').textContent = item.dataset;
  $('#prompt-position').textContent = `Prompt #${item.index + 1}`;
  $('#prompt-text').textContent = item.prompt;
  $('#original-label').textContent = item.original_label === 0 ? '0 · Non-malicious' : item.original_label === 1 ? '1 · Malicious' : 'Não informado';
  $('#original-category').textContent = item.original_category || 'Não informada';
  $('#original-source').textContent = item.source || item.dataset.replace(/\.json$/, '');
  $('#review-state').textContent = review ? '✓ Revisado' : '○ Não revisado';
  $('#review-state').classList.toggle('reviewed', !!review);
  $('#save-status').textContent = review ? 'Avaliação salva · pode ser alterada' : 'Alterações não salvas';
  renderVerdict();
  renderCategories();
}

function renderVerdict() {
  document.querySelectorAll('[data-verdict]').forEach(button => {
    const selected = button.dataset.verdict === state.verdict;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  $('#category-hint').textContent = state.verdict === 'non-malicious'
    ? 'Categorias de injeção não se aplicam a prompts non-malicious.'
    : 'Selecione uma ou mais categorias, se houver uma técnica identificável.';
}

function renderCategories() {
  const box = $('#categories');
  box.replaceChildren();
  for (const category of state.categoryOptions) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'chip'; button.textContent = category.name;
    button.title = category.description || category.name;
    button.disabled = state.verdict === 'non-malicious';
    button.classList.toggle('selected', state.categories.has(category.name));
    button.setAttribute('aria-pressed', String(state.categories.has(category.name)));
    button.addEventListener('click', () => {
      if (state.categories.has(category.name)) state.categories.delete(category.name);
      else state.categories.add(category.name);
      $('#save-status').textContent = 'Alterações não salvas';
      renderCategories();
    });
    box.append(button);
  }
}

document.querySelectorAll('[data-verdict]').forEach(button => button.addEventListener('click', () => {
  state.verdict = button.dataset.verdict;
  $('#save-status').textContent = 'Alterações não salvas';
  renderVerdict(); renderCategories();
}));
$('#notes').addEventListener('input', () => { $('#save-status').textContent = 'Alterações não salvas'; });
$('#save-review').addEventListener('click', async () => {
  if (!state.selected) return;
  if (!state.verdict) return toast('Selecione um veredito antes de salvar.');
  const button = $('#save-review'); button.disabled = true;
  const id = state.selected.id;
  try {
    const review = await api('/api/reviews', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id, verdict: state.verdict, categories: [...state.categories], notes: $('#notes').value }) });
    toast('Avaliação salva com sucesso.');
    await loadMeta();
    await loadPage(id);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
});
$('#copy-prompt').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(state.selected.prompt); toast('Prompt copiado.'); }
  catch { toast('Não foi possível copiar o prompt.'); }
});
$('#add-category').addEventListener('click', () => { $('#category-dialog').showModal(); $('#category-name').focus(); });
$('#cancel-category').addEventListener('click', () => $('#category-dialog').close());
$('#category-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    const category = await api('/api/categories', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('#category-name').value }) });
    state.categoryOptions.push(category);
    state.categoryOptions.sort((a, b) => a.name.localeCompare(b.name, 'pt-BR'));
    if (state.selected && state.verdict !== 'non-malicious') state.categories.add(category.name);
    renderCategories();
    $('#category-dialog').close(); $('#category-form').reset();
    $('#save-status').textContent = 'Alterações não salvas';
    toast('Categoria adicionada.');
  } catch (error) { toast(error.message); }
});
let searchTimer;
for (const key of ['dataset', 'status', 'label']) $(`#${key}`).addEventListener('change', () => { state.offset = 0; loadPage(); });
$('#search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadPage(); }, 300); });
$('#previous-page').addEventListener('click', () => { state.offset -= state.limit; loadPage(); });
$('#next-page').addEventListener('click', () => { state.offset += state.limit; loadPage(); });

(async () => { try { await loadMeta(); await loadPage(); } catch (error) { toast(error.message); } })();
