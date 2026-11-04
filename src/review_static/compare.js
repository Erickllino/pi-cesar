const $ = selector => document.querySelector(selector);
const number = value => new Intl.NumberFormat('pt-BR').format(value);
const state = { items: [], selected: null, total: 0, offset: 0, limit: 12, request: 0 };
const labels = { malicious: '✕ Malicious', 'non-malicious': '✓ Non-malicious', uncertain: '? Incerto' };
let toastTimer;

function toast(message) {
  const box = $('#toast'); box.textContent = message; box.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => box.classList.remove('show'), 3500);
}
async function api(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Não foi possível carregar a comparação.');
  return data;
}
async function loadMeta() {
  const meta = await api('/api/compare/meta');
  const select = $('#dataset');
  select.replaceChildren(new Option('Todos os datasets', ''));
  meta.datasets.forEach(name => select.add(new Option(name, name)));
  for (const key of ['total', 'complete', 'agree', 'disagree']) $(`#stat-${key}`).textContent = number(meta.summary[key]);
}
function params() {
  const query = new URLSearchParams({ offset: state.offset, limit: state.limit });
  for (const key of ['dataset', 'status', 'label', 'search']) {
    const value = $(`#${key}`).value.trim();
    if (value) query.set(key, value);
  }
  return query;
}
async function loadPage(preferredId) {
  const request = ++state.request;
  try {
    const data = await api(`/api/compare/prompts?${params()}`);
    if (request !== state.request) return;
    state.items = data.items; state.total = data.total;
    if (!state.items.length && state.offset > 0) {
      state.offset = Math.max(0, state.offset - state.limit);
      return loadPage(preferredId);
    }
    selectPrompt(state.items.find(item => item.id === preferredId) || state.items[0] || null);
    renderList();
  } catch (error) { toast(error.message); }
}
function smallBadge(text, kind) {
  const badge = document.createElement('span');
  badge.className = `item-badge${kind ? ` ${kind}` : ''}`;
  badge.textContent = text;
  return badge;
}
function renderList() {
  const list = $('#prompt-list'); list.replaceChildren();
  $('#queue-count').textContent = number(state.total);
  $('#list-range').textContent = state.total ? `${number(state.offset + 1)}–${number(state.offset + state.items.length)} de ${number(state.total)}` : '0 resultados';
  $('#page-info').textContent = state.total ? `${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)}` : '0 / 0';
  $('#previous-page').disabled = !state.offset;
  $('#next-page').disabled = state.offset + state.limit >= state.total;
  if (!state.items.length) {
    const empty = document.createElement('p'); empty.className = 'empty-state';
    empty.style.padding = '40px 16px'; empty.textContent = 'Nenhum prompt encontrado com esses filtros.';
    list.append(empty);
  }
  for (const item of state.items) {
    const button = document.createElement('button'); button.type = 'button';
    button.className = `prompt-item${state.selected?.id === item.id ? ' active' : ''}`;
    const top = document.createElement('div'); top.className = 'item-top';
    const source = document.createElement('span'); source.className = 'item-source';
    source.textContent = item.source || item.dataset;
    const id = document.createElement('span'); id.className = 'item-id'; id.textContent = `#${item.index + 1}`;
    top.append(source, id);
    const text = document.createElement('div'); text.className = 'item-text'; text.textContent = item.prompt;
    const badges = document.createElement('div'); badges.className = 'item-badges';
    badges.append(smallBadge(item.original_label === null ? 'Orig. —' : `Orig. ${item.original_label}`, item.original_label === null ? '' : 'done'));
    badges.append(smallBadge(item.jev ? 'Jev ✓' : 'Jev —', item.jev ? 'done' : ''));
    badges.append(smallBadge(item.human ? 'Manual ✓' : 'Manual —', item.human ? 'done' : ''));
    if (Object.values(item.comparison.disagreements).some(Boolean)) badges.append(smallBadge('≠ Divergência', 'disagree'));
    button.append(top, text, badges);
    button.addEventListener('click', () => { selectPrompt(item); renderList(); });
    list.append(button);
  }
}
function verdict(element, value, missing = 'Não informado') {
  element.textContent = value ? labels[value] || value : missing;
  element.className = `comparison-verdict${value ? ` ${value}` : ''}`;
}
function chips(container, values, probabilities) {
  container.replaceChildren();
  if (!values?.length) {
    const empty = document.createElement('span'); empty.className = 'compare-muted';
    empty.textContent = 'Nenhuma categoria registrada'; container.append(empty); return;
  }
  for (const value of values) {
    const chip = document.createElement('span'); chip.className = 'compare-chip';
    const probability = probabilities?.[value];
    chip.textContent = `${value}${typeof probability === 'number' ? ` · ${Math.round(probability * 100)}%` : ''}`;
    container.append(chip);
  }
}
function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('pt-BR');
}
function selectPrompt(item) {
  state.selected = item;
  $('#detail-empty').hidden = !!item; $('#detail-content').hidden = !item;
  if (!item) return;
  $('#detail-heading').textContent = `Prompt #${item.index + 1}`;
  $('#prompt-dataset').textContent = item.dataset;
  $('#prompt-position').textContent = `Registro #${item.index + 1}`;
  $('#prompt-source').textContent = item.source ? `· ${item.source}` : '';
  $('#prompt-text').textContent = item.prompt;
  const badge = $('#comparison-badge');
  const disagreement = Object.values(item.comparison.disagreements).some(Boolean);
  badge.textContent = disagreement ? '≠ Divergência' : item.comparison.agree ? '✓ Concordância total' : 'Comparação incompleta';
  badge.className = `comparison-badge ${disagreement ? 'disagree' : item.comparison.agree ? 'agree' : 'partial'}`;
  verdict($('#original-verdict'), item.original_label === 1 ? 'malicious' : item.original_label === 0 ? 'non-malicious' : null, 'Rótulo não informado');
  $('#original-category').textContent = item.original_category || 'Não informada';
  $('#original-source').textContent = item.source || item.dataset.replace(/\.json$/, '');
  $('#original-words-wrap').hidden = !item.word_list?.length;
  chips($('#original-words'), item.word_list || []);
  verdict($('#jev-verdict'), item.jev?.verdict, 'Sem avaliação');
  $('#jev-details').hidden = !item.jev; $('#jev-missing').hidden = !!item.jev;
  if (item.jev) {
    $('#jev-confidence').textContent = `Confiança da escolha: ${Math.round(item.jev.confidence * 100)}%`;
    const probabilities = $('#jev-probabilities'); probabilities.replaceChildren();
    for (const [name, probability] of Object.entries(item.jev.probabilities || {})) {
      const row = document.createElement('div'); row.className = 'compare-probability';
      const label = document.createElement('span'); label.textContent = name;
      const track = document.createElement('div'); track.className = 'bar';
      const fill = document.createElement('div'); fill.className = 'fill';
      fill.style.width = `${Math.round(probability * 100)}%`; track.append(fill);
      const percent = document.createElement('strong'); percent.textContent = `${Math.round(probability * 100)}%`;
      row.append(label, track, percent); probabilities.append(row);
    }
    chips($('#jev-categories'), item.jev.categories, item.jev.category_probabilities);
    $('#jev-meta').textContent = `${item.jev.model} · ${formatDate(item.jev.updated_at)}`;
  }
  verdict($('#human-verdict'), item.human?.verdict, 'Sem revisão');
  $('#human-details').hidden = !item.human; $('#human-missing').hidden = !!item.human;
  if (item.human) {
    chips($('#human-categories'), item.human.categories);
    $('#human-notes').textContent = item.human.notes?.trim() || 'Nenhuma observação registrada.';
    $('#human-updated').textContent = formatDate(item.human.updated_at);
  }
}
let searchTimer;
for (const key of ['dataset', 'status', 'label']) $(`#${key}`).addEventListener('change', () => { state.offset = 0; loadPage(); });
$('#search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadPage(); }, 300); });
$('#previous-page').addEventListener('click', () => { state.offset -= state.limit; loadPage(); });
$('#next-page').addEventListener('click', () => { state.offset += state.limit; loadPage(); });
(async () => { try { await loadMeta(); await loadPage(); } catch (error) { toast(error.message); } })();
