/* Chess Game Review – frontend logic
 *
 * Uses chessground (Lichess board library) loaded as an ES module.
 * No build step, no jQuery, no external image files.
 */

import { Chessground } from 'https://cdn.jsdelivr.net/npm/chessground@9.1.1/+esm';

// ── State ────────────────────────────────────────────────────────────────────

let cg           = null;   // Chessground instance
let currentGame  = null;   // full game object from /api/game/{id}
let currentIndex = -1;     // -1 = start, 0..N-1 = after move[index]
let flipped      = false;
let analysisCache = {};    // "fen|uci" → analysis response

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract piece-placement portion of FEN (chessground only needs this part). */
function fenPieces(fullFen) {
  return fullFen.split(' ')[0];
}

/** Convert UCI move string ("e2e4") to chessground lastMove array (["e2","e4"]). */
function uciToLastMove(uci) {
  if (!uci || uci.length < 4) return undefined;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

/** Return whose turn it is from a full FEN string. */
function fenTurn(fullFen) {
  return fullFen.split(' ')[1] === 'b' ? 'black' : 'white';
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  cg = Chessground(document.getElementById('board'), {
    fen: 'start',
    orientation: 'white',
    movable:   { color: 'none', free: false },   // read-only
    draggable: { enabled: false },
    selectable: { enabled: false },
    animation: { enabled: true, duration: 200 },
    highlight: { lastMove: true, check: true },
  });

  bindControls();
  await loadGameList();
});

// ── Game list ─────────────────────────────────────────────────────────────────

async function loadGameList() {
  const [gamesRes, userRes] = await Promise.all([
    fetch('/api/games'),
    fetch('/api/username'),
  ]);
  const games = await gamesRes.json();
  const { username } = await userRes.json();

  document.getElementById('username-label').textContent = username ? `@${username}` : '';

  const sel = document.getElementById('game-select');
  sel.innerHTML = '';

  if (!games.length) {
    sel.innerHTML = '<option value="">No games found</option>';
    return;
  }

  games.forEach((g, i) => {
    const opt = document.createElement('option');
    opt.value = g.id;
    opt.textContent = `${i + 1}. ${g.title}`;
    sel.appendChild(opt);
  });

  sel.addEventListener('change', () => { if (sel.value) loadGame(sel.value); });
  await loadGame(games[0].id);
}

// ── Load game ─────────────────────────────────────────────────────────────────

async function loadGame(gameId) {
  const res = await fetch(`/api/game/${gameId}`);
  currentGame   = await res.json();
  currentIndex  = -1;
  analysisCache = {};
  renderMoveList();
  updateGameInfo();
  goTo(-1);
}

// ── Board update ──────────────────────────────────────────────────────────────

function setBoard(fen, lastMoveUci) {
  cg.set({
    fen:         fenPieces(fen),
    turnColor:   fenTurn(fen),
    lastMove:    uciToLastMove(lastMoveUci),
    orientation: flipped ? 'black' : 'white',
  });
}

// ── Navigation ────────────────────────────────────────────────────────────────

function goTo(index) {
  if (!currentGame) return;
  const max = currentGame.moves.length - 1;
  index = Math.max(-1, Math.min(index, max));
  currentIndex = index;

  if (index === -1) {
    const startFen = currentGame.moves[0]?.fen_before
      ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    setBoard(startFen, null);
  } else {
    const move = currentGame.moves[index];
    const fen  = index < max
      ? currentGame.moves[index + 1].fen_before
      : currentGame.final_fen;
    setBoard(fen, move.uci);
  }

  highlightActive();
  updatePositionInfo();
  updateButtons();

  if (index >= 0) {
    const m = currentGame.moves[index];
    fetchAnalysis(m.fen_before, m.uci);
  } else {
    clearAnalysis();
  }
}

function bindControls() {
  document.getElementById('btn-start').addEventListener('click', () => goTo(-1));
  document.getElementById('btn-prev') .addEventListener('click', () => goTo(currentIndex - 1));
  document.getElementById('btn-next') .addEventListener('click', () => goTo(currentIndex + 1));
  document.getElementById('btn-end')  .addEventListener('click', () => currentGame && goTo(currentGame.moves.length - 1));

  document.getElementById('btn-flip').addEventListener('click', () => {
    flipped = !flipped;
    goTo(currentIndex);
  });

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft')  { e.preventDefault(); goTo(currentIndex - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); goTo(currentIndex + 1); }
    if (e.key === 'ArrowUp')    { e.preventDefault(); goTo(-1); }
    if (e.key === 'ArrowDown')  { e.preventDefault(); currentGame && goTo(currentGame.moves.length - 1); }
  });
}

// ── Move list ─────────────────────────────────────────────────────────────────

function renderMoveList() {
  const el = document.getElementById('move-list');
  el.innerHTML = '';
  if (!currentGame?.moves.length) return;

  currentGame.moves.forEach((m, i) => {
    if (m.color === 'white') {
      const num = document.createElement('span');
      num.className = 'move-num';
      num.textContent = m.move_number + '.';
      el.appendChild(num);
    }
    const cell = document.createElement('span');
    cell.className = 'move-cell';
    cell.dataset.index = i;
    cell.textContent = m.san;
    cell.addEventListener('click', () => goTo(i));
    el.appendChild(cell);

    // Fill empty column when white's move ends the game
    if (m.color === 'white' && i === currentGame.moves.length - 1) {
      el.appendChild(document.createElement('span'));
    }
  });
}

function highlightActive() {
  document.querySelectorAll('.move-cell').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.index) === currentIndex);
  });
  document.querySelector('.move-cell.active')
    ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ── Info bar ──────────────────────────────────────────────────────────────────

function updatePositionInfo() {
  const el = document.getElementById('move-counter');
  if (!currentGame) { el.textContent = '—'; return; }
  const total = currentGame.moves.length;
  el.textContent = currentIndex === -1
    ? 'Start position'
    : `Move ${currentIndex + 1} / ${total} — ${currentGame.moves[currentIndex].san}`;
}

function updateButtons() {
  const total = currentGame ? currentGame.moves.length : 0;
  document.getElementById('btn-start').disabled = currentIndex <= -1;
  document.getElementById('btn-prev') .disabled = currentIndex <= -1;
  document.getElementById('btn-next') .disabled = currentIndex >= total - 1;
  document.getElementById('btn-end')  .disabled = currentIndex >= total - 1;
}

function updateGameInfo() {
  const el = document.getElementById('game-info');
  if (!currentGame) { el.textContent = ''; return; }
  const resultText = {
    '1-0': 'White wins', '0-1': 'Black wins', '1/2-1/2': 'Draw',
  }[currentGame.result] || currentGame.result;
  el.innerHTML =
    `<strong>${currentGame.white}</strong> vs <strong>${currentGame.black}</strong> &nbsp;·&nbsp; ` +
    `${resultText} by ${currentGame.result_detail} &nbsp;·&nbsp; ` +
    `${currentGame.date} &nbsp;·&nbsp; TC: ${currentGame.time_control}s`;

  document.getElementById('result-badge').textContent = currentGame.result;
}

// ── Analysis ──────────────────────────────────────────────────────────────────

async function fetchAnalysis(fen, moveUci) {
  const key = `${fen}|${moveUci}`;
  showAnalysisLoading();

  if (analysisCache[key]) { renderAnalysis(analysisCache[key]); return; }

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, move_uci: moveUci }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    analysisCache[key] = data;

    // Only render if still the current move
    const cur = currentIndex >= 0 ? currentGame.moves[currentIndex] : null;
    if (cur && `${cur.fen_before}|${cur.uci}` === key) renderAnalysis(data);
  } catch (err) {
    showAnalysisError(err.message);
  }
}

function renderAnalysis(data) {
  const el  = document.getElementById('analysis-content');
  const cls = data.classification;

  // Tag move cell for CSS annotation badge (★ ?? ? !?)
  const activeCell = document.querySelector('.move-cell.active');
  if (activeCell) activeCell.dataset.class = cls;

  const sourceBadge = data.comment_source === 'llm'
    ? '<span class="source-badge source-llm">AI</span>'
    : '<span class="source-badge source-template">engine</span>';

  el.innerHTML = `
    <div class="analysis-row">
      <span class="class-pill class-${cls}">${cls}</span>
      <span class="eval-pill">${data.eval_before}</span>
      ${sourceBadge}
    </div>
    ${!data.is_best
      ? `<div class="best-move-row">Best: <strong>${data.best_move}</strong>${data.cp_loss > 0 ? ` (−${data.cp_loss} cp)` : ''}</div>`
      : ''}
    <p class="comment-text">${data.comment}</p>
  `;
}

function showAnalysisLoading() {
  document.getElementById('analysis-content').innerHTML =
    '<span class="analysis-loading">Analyzing...</span>';
}

function showAnalysisError(msg) {
  document.getElementById('analysis-content').innerHTML =
    `<span class="placeholder">Analysis unavailable: ${msg}</span>`;
}

function clearAnalysis() {
  document.getElementById('analysis-content').innerHTML =
    '<p class="placeholder">Select a move to see analysis.</p>';
}
