const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (x, d = 3) => x == null ? "–" : (+x).toFixed(d);
const ago = t => {
  const s = Date.now() / 1000 - t;
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(t * 1000).toLocaleDateString();
};

let runs = [], detail = null, sel = decodeURIComponent(location.hash.slice(1)) || null;
let idx = -1, follow = true, playing = null;

async function loadRuns() {
  try { runs = await (await fetch("/api/runs")).json(); } catch { return; }
  renderRuns();
  const s = runs.find(r => r.name === sel);
  if (sel && s && (!detail || detail.name !== sel || s.mtime !== detail.mtime)) loadDetail();
}

function renderRuns() {
  const q = $("#filter").value.toLowerCase();
  const list = runs.filter(r => !q || r.name.toLowerCase().includes(q) || (r.target || "").toLowerCase().includes(q));
  const live = runs.filter(r => r.running).length;
  $("#count").textContent = `${runs.length} runs` + (live ? ` · ${live} running` : "");
  // Elements are updated in place, so that the thumbnails do not reload and flicker.
  const box = $("#runs"), keep = new Set(list.map(r => r.name));
  for (const el of [...box.children]) if (!keep.has(el.dataset.name)) el.remove();
  list.forEach((r, i) => {
    let el = [...box.children].find(el => el.dataset.name === r.name);
    if (!el) {
      el = document.createElement("div");
      el.dataset.name = r.name;
      el.innerHTML = `<img class="thumb" alt=""><div class="info" style="min-width:0"></div>`;
    }
    if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
    el.className = "run" + (r.name === sel ? " sel" : "");
    const img = $(".thumb", el);
    img.classList.toggle("noimg", r.image == null);
    if (r.image != null) setSrc(img, imgUrl(r.name, r.image, r.version));
    const info = `
        <div class="name"><span class="dot ${dotClass(r)}" title="${esc(r.status ?? "not run")}"></span>${esc(r.name)}
          ${r.hits ? `<span class="badge ok" title="checks with the exact target">${r.hits}/${r.checks}</span>` : ""}</div>
        <div class="bar"><div style="width:${r.steps ? 100 * (r.step || 0) / r.steps : 0}%"></div></div>
        <div class="meta">${r.step ?? "–"}/${r.steps ?? "?"} · loss ${fmt(r.loss)}${r.ms_step == null ? "" : ` · ${Math.round(r.ms_step)} ms`} · ${esc(r.method ?? "")} · ${ago(r.mtime)}</div>
        <div class="out" title="${esc(r.output)}">${esc(r.output)}</div>`;
    patch($(".info", el), info);
  });
}

// Sets the contents or an attribute only if they differ, to keep the updates local.
function patch(el, html) { if (el && el._html !== html) { el.innerHTML = html; el._html = html; } }
function setText(el, t) { t = String(t); if (el && el.textContent !== t) el.textContent = t; }
function setAttr(el, k, v) { v = String(v); if (el && el.getAttribute(k) !== v) el.setAttribute(k, v); }

const dotClass = r => ({running: "live", stopped: "stopped"})[r.status] ?? "";

const imgUrl = (name, step, version) => `/img/${encodeURIComponent(name)}/${step}.png?v=${version}`;

// Sets the source of an image once the new one is loaded, keeping the old one until then.
function setSrc(img, url) {
  if (img.dataset.src === url) return;
  img.dataset.src = url;
  const pre = new Image();
  pre.onload = () => { if (img.dataset.src === url) img.src = url; };
  pre.src = url;
}

$("#runs").addEventListener("click", e => {
  const el = e.target.closest(".run");
  if (el) select(el.dataset.name);
});
$("#filter").addEventListener("input", renderRuns);

function select(name) {
  if (name === sel && detail) return;
  sel = name; detail = null; idx = -1; follow = true; stop();
  history.replaceState(null, "", "#" + encodeURIComponent(name));
  renderRuns(); loadDetail();
}

async function loadDetail() {
  const name = sel;
  let d;
  try { d = await (await fetch(`/api/runs/${encodeURIComponent(name)}`)).json(); } catch { return; }
  if (name !== sel) return;
  const atEnd = !detail || idx >= detail.images.length - 1;
  const same = detail && detail.name === d.name && detail.version === d.version && $("#view");
  detail = d;
  if (idx < 0 || (follow && atEnd)) idx = d.images.length - 1;
  idx = Math.min(idx, d.images.length - 1);
  same ? updateDetail() : renderDetail();
}

function rowAt(step) {
  return detail.log.find(r => r.step === step);
}

function renderDetail() {
  const d = detail;
  if (!d) return;
  $("#main").innerHTML = `
    <div class="title">
      <h2><span class="dot" id="livedot" style="display:inline-block;margin-right:8px"></span>${esc(d.name)}</h2>
      <span class="target">“${esc(d.target)}”</span>
    </div>
    <div class="grid">
      <div>
        <div class="card">
          <img id="view" alt="">
          <div class="controls">
            <button id="play" title="play (space)">▶</button>
            <input type="range" id="slider" min="0" value="${idx}">
            <span class="stepinfo" id="stepinfo"></span>
          </div>
          <div class="strip" id="strip"></div>
          <div class="output" id="output"></div>
        </div>
        <div class="legend" style="margin-top:8px"><span><kbd>←</kbd><kbd>→</kbd> step</span><span><kbd>space</kbd> play</span><span><kbd>↑</kbd><kbd>↓</kbd> run</span><span><kbd>End</kbd> latest</span></div>
      </div>
      <div class="right">
        <div class="card">
          <div class="stats">
            <div class="stat"><div class="k">step</div><div class="v" id="s_step"></div></div>
            <div class="stat"><div class="k">loss</div><div class="v" id="s_loss"></div></div>
            <div class="stat"><div class="k" title="checks with the exact target">hits</div><div class="v" id="s_hits"></div></div>
            <div class="stat"><div class="k">first hit</div><div class="v" id="s_first"></div></div>
            <div class="stat"><div class="k">ms/step</div><div class="v" id="s_ms"></div></div>
          </div>
          <div id="chart"></div>
          <div class="hover" id="hover"></div>
        </div>
        <div class="card"><h3>log</h3><div class="logtab"><table id="logtab"></table></div></div>
        <div class="card"><h3>config</h3><table id="config"></table></div>
        <div class="card" id="runsh"><h3>run.sh</h3><pre></pre></div>
      </div>
    </div>`;
  $("#slider").addEventListener("input", e => { stop(); setIdx(+e.target.value); });
  $("#play").addEventListener("click", () => playing ? stop() : play());
  $("#strip").addEventListener("click", e => { if (e.target.dataset.i) { stop(); setIdx(+e.target.dataset.i); } });
  $("#logtab").addEventListener("click", e => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    const i = detail.images.indexOf(+tr.dataset.step);
    if (i >= 0) { stop(); setIdx(i); }
  });
  if (playing) { $("#play").textContent = "❚❚"; $("#play").classList.add("on"); }
  updateDetail();
}

// Updates the parts of the detail view that change while a case runs.
function updateDetail() {
  const d = detail, cfg = d.config || {};
  setAttr($("#livedot"), "class", `dot ${dotClass(d)}`);
  setAttr($("#livedot"), "title", d.status ?? "not run");
  setAttr($("#slider"), "max", Math.max(0, d.images.length - 1));
  setText($("#s_step"), `${d.step ?? "–"}/${d.steps ?? "?"}`);
  setText($("#s_loss"), fmt(d.loss));
  setText($("#s_hits"), `${d.hits}/${d.checks}`);
  setText($("#s_first"), d.first_hit ?? "–");
  setText($("#s_ms"), d.ms_step == null ? "–" : Math.round(d.ms_step));
  patch($("#config"), Object.keys(cfg).map(k => `<tr><td>${esc(k)}</td><td class="v">${esc(JSON.stringify(cfg[k]))}</td></tr>`).join(""));
  $("#runsh").style.display = d.run_sh ? "" : "none";
  setText($("pre", $("#runsh")), d.run_sh || "");
  renderStrip(); renderChart(); renderLog(); showStep();
}

function setIdx(i) {
  if (!detail || !detail.images.length) return;
  idx = Math.max(0, Math.min(detail.images.length - 1, i));
  follow = idx === detail.images.length - 1;
  showStep();
}

function showStep() {
  const d = detail;
  if (!d.images.length) { $("#stepinfo").textContent = "no images"; return; }
  const step = d.images[idx];
  setSrc($("#view"), imgUrl(d.name, step, d.version));
  if (+$("#slider").value !== idx) $("#slider").value = idx;
  setText($("#stepinfo"), `step ${step}`);
  const r = rowAt(step);
  const hit = r && r.output === d.target;
  const o = $("#output");
  o.classList.toggle("hit", !!hit);
  patch(o, r ? `<div class="row">loss ${fmt(r.loss, 4)} · tokens_ok ${fmt(r.tokens_ok, 2)} · linf ${fmt(r.linf)} ${hit ? "· exact target" : ""}</div>${esc(r.output)}` : "");
  document.querySelectorAll("#strip img").forEach(im => im.classList.toggle("cur", +im.dataset.i === idx));
  document.querySelectorAll("#logtab tr").forEach(tr => tr.classList.toggle("cur", +tr.dataset.step === step));
  const cur = $("#logtab tr.cur");
  // Scrolls only the log box, scrollIntoView() would also scroll the page.
  const box = $(".logtab");
  if (cur && box && !playing) {
    const r = cur.getBoundingClientRect(), b = box.getBoundingClientRect();
    if (r.top < b.top) box.scrollTop -= b.top - r.top;
    else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom;
  }
  if (chart) chart.redraw(false, false);
}

function renderStrip() {
  const d = detail, n = d.images.length, k = Math.min(n, 12);
  const picks = [...new Set(Array.from({length: k}, (_, j) => Math.round(j * (n - 1) / Math.max(1, k - 1))))];
  const strip = $("#strip");
  while (strip.children.length > picks.length) strip.lastChild.remove();
  while (strip.children.length < picks.length) strip.appendChild(document.createElement("img"));
  picks.forEach((i, j) => {
    const im = strip.children[j];
    im.dataset.i = i;
    im.title = `step ${d.images[i]}`;
    im.classList.toggle("cur", i === idx);
    setSrc(im, imgUrl(d.name, d.images[i], d.version));
  });
}

const logRow = (r, target) => `<tr data-step="${r.step}" class="${r.output === target ? "hit" : ""}"><td>${r.step}</td><td>${fmt(r.loss, 3)}</td><td>${fmt(r.tokens_ok, 2)}</td><td>${esc(r.output)}</td></tr>`;

// Rows are newest first. New rows are inserted at the top, the table is rebuilt only if the log was restarted.
function renderLog() {
  const d = detail, tab = $("#logtab"), n = tab.rows.length;
  const prefix = n && n <= d.log.length && +tab.rows[n - 1].dataset.step === d.log[0].step
    && +tab.rows[0].dataset.step === d.log[n - 1].step;
  if (!prefix) { tab.innerHTML = d.log.slice().reverse().map(r => logRow(r, d.target)).join(""); return; }
  if (d.log.length > n) tab.insertAdjacentHTML("afterbegin", d.log.slice(n).reverse().map(r => logRow(r, d.target)).join(""));
}

// Chart with uPlot, created once per run and updated with setData().
let chart = null, zoomed = false;
const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();

function chartData(d) {
  const log = d.log;
  return [
    log.map(r => r.step),
    log.map(r => r.loss > 0 ? r.loss : null),
    log.map(r => r.tokens_ok),
    log.map(r => r.output === d.target ? 0 : null),
  ];
}

function makeChart(el) {
  const axis = {stroke: css("--dim"), grid: {stroke: css("--line"), width: 1}, ticks: {stroke: css("--line"), width: 1},
    font: `10px ${css("--mono")}`};
  let down = null;
  const u = new uPlot({
    width: el.clientWidth, height: 220,
    scales: {
      x: {time: false, range: (u, lo, hi) => [0, Math.max(detail?.steps || 0, hi, 1)]},
      loss: {distr: 3},
      tok: {range: [0, 1.05]},
    },
    axes: [
      {...axis},
      {...axis, scale: "loss", size: 44},
      {...axis, scale: "tok", side: 1, size: 34, grid: {show: false}, splits: () => [0, 0.5, 1]},
    ],
    series: [
      {label: "step", value: (u, v) => v ?? "–"},
      {label: "loss", scale: "loss", stroke: css("--accent"), width: 1.8, value: (u, v) => v == null ? "–" : v.toFixed(4)},
      {label: "tokens_ok", scale: "tok", stroke: css("--warn"), width: 1.3, value: (u, v) => v == null ? "–" : v.toFixed(2)},
      {label: "exact target", scale: "tok", stroke: css("--ok"), paths: () => null,
        points: {show: true, size: 7, fill: css("--ok"), stroke: css("--ok")}, value: (u, v) => v == null ? "–" : "yes"},
    ],
    cursor: {points: {size: 6}, drag: {x: true, y: false}},
    legend: {live: true},
    hooks: {
      // Marker of the step shown in the image.
      draw: [u => {
        if (!detail || !detail.images.length) return;
        const x = u.valToPos(detail.images[idx], "x", true), c = u.ctx;
        c.save(); c.strokeStyle = css("--text"); c.globalAlpha = 0.5; c.lineWidth = devicePixelRatio;
        c.beginPath(); c.moveTo(x, u.bbox.top); c.lineTo(x, u.bbox.top + u.bbox.height); c.stroke(); c.restore();
      }],
      setCursor: [u => {
        const i = u.cursor.idx, r = i != null ? detail?.log[i] : null;
        patch($("#hover"), r ? `step ${r.step}: ${esc(r.output)}` : "");
      }],
      setSelect: [u => { if (u.select.width > 0) zoomed = true; }],
    },
  }, [[], [], [], []], el);
  // Click without dragging shows the image of the nearest step, double click resets the zoom.
  u.over.addEventListener("mousedown", e => { down = [e.clientX, e.clientY]; });
  u.over.addEventListener("click", e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 3 || !detail.images.length) return;
    const s = u.posToVal(u.cursor.left, "x");
    let best = 0;
    detail.images.forEach((v, i) => { if (Math.abs(v - s) < Math.abs(detail.images[best] - s)) best = i; });
    stop(); setIdx(best);
  });
  u.over.addEventListener("dblclick", () => { zoomed = false; });
  new ResizeObserver(() => u.setSize({width: el.clientWidth, height: 220})).observe(el);
  return u;
}

function renderChart() {
  const el = $("#chart");
  if (!chart || !el.contains(chart.root)) {
    if (chart) chart.destroy();
    chart = makeChart(el);
    zoomed = false;
  }
  const {min, max} = chart.scales.x;
  chart.setData(chartData(detail));
  // Keeps the zoom while a running case updates.
  if (zoomed) chart.setScale("x", {min, max});
}

function play() {
  if (!detail || !detail.images.length) return;
  if (idx >= detail.images.length - 1) setIdx(0);
  $("#play").textContent = "❚❚"; $("#play").classList.add("on");
  playing = setInterval(() => {
    if (idx >= detail.images.length - 1) return stop();
    setIdx(idx + 1);
  }, 120);
}

function stop() {
  if (playing) clearInterval(playing);
  playing = null;
  const b = $("#play");
  if (b) { b.textContent = "▶"; b.classList.remove("on"); }
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (e.key === "ArrowRight") { e.preventDefault(); stop(); setIdx(idx + 1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); stop(); setIdx(idx - 1); }
  else if (e.key === "End") { stop(); setIdx(Infinity); }
  else if (e.key === "Home") { stop(); setIdx(0); }
  else if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const names = [...document.querySelectorAll(".run")].map(el => el.dataset.name);
    const i = names.indexOf(sel) + (e.key === "ArrowDown" ? 1 : -1);
    if (names[i]) { select(names[i]); document.querySelector(".run.sel")?.scrollIntoView({block: "nearest"}); }
  }
});

loadRuns();
setInterval(loadRuns, 2000);
