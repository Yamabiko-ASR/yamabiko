(() => {
  // Telop overlay for the browser/OBS page: the same design as the Qt
  // overlay (yamabiko/overlay_app/telop_*.py). Numbers kept in sync by hand.
  const PT_TO_PX = 4 / 3;
  const RESHAPE_MS = 350;
  const AGE_SCALE = [1, 0.84, 0.70];   // echo ages 0, 1, 2 (older keep age 2)
  const GENRE_BANNER = {
    variety: ["バラエティ", "Variety"], news: ["ニュース", "News"],
    anime: ["アニメ", "Anime"], drama: ["ドラマ", "Drama"],
  };
  const PAUSED_BANNER = ["一時停止", "Paused"];
  const CHOICES = {
    mode: ["watch", "learn"], alignment: ["left", "center"],
    echo: ["priority", "both"], banner: ["jp", "jp_en"],
  };
  const KINDS = ["translator", "audio"];   // display order (problems.KIND_ORDER)
  const NOTICE_LABEL = "お知らせ";
  const NOTICE_GAP = 8;                      // telop_look.NOTICE_GAP
  const RECOVERY_MS = 5000;                  // telop_look.NOTICE_RECOVERY_MS

  const state = {
    holdMs: 5000, fadeInMs: 160, fadeOutMs: 420, maxLines: 3, reduceMotion: false,
    genre: "variety", paused: false, seriesId: null, seriesLabel: "", seriesNative: "",
  };

  // The pipeline's problems ("problems" frame, yamabiko/problems.py): the
  // last set seen, the recovery lines showing and the carried recoveries
  // already shown. Page lifetime: a reconnect keeps them, like the Qt
  // overlay's ProblemTracker (Decision 47).
  const problemLines = new Map();   // kind -> entry
  const recoveries = new Map();     // kind -> { text, timer }
  const carriedShown = new Set();

  const body = document.body;
  const stage = document.getElementById("stage");
  const linesBox = stage.querySelector(".lines");
  const plateJp = stage.querySelector(".plate-jp");
  const plateEn = stage.querySelector(".plate-en");
  const banner = stage.querySelector(".vt");
  const noticesBox = stage.querySelector(".notices");
  const status = document.getElementById("status");

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  function jpHtml(msg) {
    if (Array.isArray(msg.jp_ruby) && msg.jp_ruby.length > 0) {
      return msg.jp_ruby.map(seg => {
        const text = escapeHtml(seg.text || "");
        if (seg.reading) {
          return `<ruby>${text}<rt>${escapeHtml(seg.reading)}</rt></ruby>`;
        }
        return text;
      }).join("");
    }
    return escapeHtml(msg.jp || "");
  }

  // Defense-in-depth: the server (browser_server._sanitize_style)
  // is the authoritative gate -- it strips CSS-injection AND control
  // chars before broadcasting. This client-side pass is a
  // belt-and-suspenders guard so a value that ever slipped through (or a
  // non-server sender) cannot break out of a CSS custom-property
  // declaration. Drops the CSS-breaking chars ; { } < > from a string.
  function sanitizeCss(s) {
    return String(s).replace(/[;{}<>]/g, "");
  }

  function num(value, fallback) {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
  }
  function px(points) { return Math.max(1, Math.round(points * PT_TO_PX)); }
  function rgb(hex, fallback) {
    const m = /^#([0-9a-fA-F]{6})$/.exec(String(hex || ""));
    if (!m) return fallback;
    const v = parseInt(m[1], 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  function choice(msg, key) {
    return CHOICES[key].includes(msg[key]) ? msg[key] : CHOICES[key][0];
  }

  // Pixel sizes rounded exactly as telop_look.py and telop_text.py round
  // them (Math.round is their round-half-up for these positive values), so
  // both renderers draw the same pixels: fonts, furigana and row heights
  // per echo age, and paddings.
  function applySizes(msg) {
    const r = document.documentElement.style;
    const learn = choice(msg, "mode") === "learn";
    const jpPt = num(msg.jp_size, 16), enPt = num(msg.en_size, 23);
    const jp = learn ? Math.max(1, Math.round(enPt * PT_TO_PX * 33 / 31)) : px(jpPt);
    const kana = Math.max(1, Math.round(jp * (learn ? 0.46 : 0.5)));
    const en = learn ? px(jpPt) : px(enPt);
    const jpRow = learn ? 1.9 : 1.75;
    AGE_SCALE.forEach((k, age) => {
      const jpPx = Math.max(1, Math.round(jp * k)), enPx = Math.max(1, Math.round(en * k));
      r.setProperty(`--jp-px-${age}`, `${jpPx}px`);
      r.setProperty(`--kana-px-${age}`, `${Math.max(1, Math.round(kana * k))}px`);
      r.setProperty(`--en-px-${age}`, `${enPx}px`);
      r.setProperty(`--jp-row-${age}`, `${Math.round(jpPx * jpRow)}px`);
      r.setProperty(`--en-row-${age}`, `${Math.round(enPx * 1.2)}px`);
    });
    const k = Math.max(0, num(msg.padding, 10)) / 10;
    const pad = v => `${Math.round(v * k)}px`;
    const [jt, js, jb] = learn ? [4, 22, 2] : [3, 18, 2];
    const [et, es, eb] = learn ? [6, 22, 8] : [9, 20, 11];
    r.setProperty('--jp-pad', `${pad(jt)} ${pad(js)} ${pad(jb)}`);
    r.setProperty('--en-pad', `${pad(et)} ${pad(es)} ${pad(eb)}`);
    r.setProperty('--echo-jp-pad', `0px ${pad(js)} 0px`);
    r.setProperty('--echo-en-pad', `${pad(1)} ${pad(es)} ${pad(3)}`);
  }

  function applyStyle(msg) {
    const r = document.documentElement.style;
    r.setProperty('--jp-font', `"${sanitizeCss(msg.jp_font)}", sans-serif`);
    r.setProperty('--en-font', `"${sanitizeCss(msg.en_font)}", "Segoe UI", sans-serif`);
    applySizes(msg);
    const [tr, tg, tb] = rgb(msg.text_color, [255, 255, 255]);
    r.setProperty('--text-color', `rgb(${tr}, ${tg}, ${tb})`);
    r.setProperty('--echo-kana', `rgba(${tr}, ${tg}, ${tb}, 0.85)`);
    r.setProperty('--learn-en-color', `rgba(${tr}, ${tg}, ${tb}, 0.88)`);
    const [br, bg, bb] = rgb(msg.bg_color, [14, 16, 20]);
    const a = (Math.min(255, Math.max(0, num(msg.bg_opacity, 237))) / 255).toFixed(3);
    r.setProperty('--bar-color', `rgba(${br}, ${bg}, ${bb}, ${a})`);
    // Width cap (browser/Qt parity). vw, not %: a percentage cap inside the
    // shrink-to-fit stack is cyclic and caps nothing. Guard on typeof
    // number, not Number() (Number(null) is 0 and would clamp to 30vw while
    // Qt falls back to 70); remove the property on garbage so the CSS
    // var(--max-width, 70vw) default recovers.
    if (typeof msg.max_width_pct === "number" && Number.isFinite(msg.max_width_pct)) {
      r.setProperty('--max-width', `${Math.min(100, Math.max(30, msg.max_width_pct))}vw`);
    } else {
      r.removeProperty('--max-width');
    }
    r.setProperty('--fade-in-ms', `${num(msg.fade_in_ms, 160)}ms`);
    r.setProperty('--fade-out-ms', `${num(msg.fade_out_ms, 420)}ms`);
    state.holdMs = num(msg.hold_ms, 5000);
    state.fadeInMs = num(msg.fade_in_ms, 160);
    state.fadeOutMs = num(msg.fade_out_ms, 420);
    state.maxLines = Math.max(1, Math.floor(num(msg.max_lines, 3)));
    state.reduceMotion = msg.reduce_motion === true;
    body.classList.toggle("learn", choice(msg, "mode") === "learn");
    const align = choice(msg, "alignment");
    body.classList.toggle("align-left", align === "left");
    body.classList.toggle("align-center", align === "center");
    body.classList.toggle("echo-priority", choice(msg, "echo") === "priority");
    body.classList.toggle("tag-en", choice(msg, "banner") === "jp_en");
    body.classList.toggle("reduce-motion", state.reduceMotion);
    trim();
    sync(false);
  }

  function bannerText() {
    if (state.paused) return PAUSED_BANNER;
    if (state.seriesId !== null && state.seriesNative) {
      return [state.seriesNative, state.seriesLabel.replace(/\s*\(\d{4}\)\s*$/, "")];
    }
    return GENRE_BANNER[state.genre] || GENRE_BANNER.variety;
  }

  function span(cls) {
    const el = document.createElement("span");
    el.className = cls;
    return el;
  }

  function retag() {
    const tops = stackTops();
    const [jp, en] = bannerText();
    banner.replaceChildren();
    if (state.paused) {
      const glyph = span("pg");
      glyph.append(document.createElement("i"), document.createElement("i"));
      glyph.setAttribute("aria-hidden", "true");
      banner.append(glyph);
    }
    const vj = span("vj");
    vj.textContent = jp;
    const ve = span("ve");
    ve.textContent = en;
    banner.append(vj, ve);
    body.classList.toggle("paused", state.paused);
    sync(false);
    glide(tops);
  }

  function liveLines() {
    return [...linesBox.querySelectorAll(".line:not(.leaving):not(.expiring)")];
  }

  function reage() {
    const live = liveLines();
    live.forEach((el, i) => { el.dataset.age = String(live.length - 1 - i); });
  }

  // Box of el relative to the stage's bottom-left (layout offsets, so the
  // FLIP and entrance transforms are ignored: where the line settles).
  function boxOf(el) {
    let x = 0, y = 0, n = el;
    while (n && n !== stage) { x += n.offsetLeft; y += n.offsetTop; n = n.offsetParent; }
    return { left: x, bottom: stage.clientHeight - y - el.offsetHeight,
             width: el.offsetWidth, height: el.offsetHeight };
  }

  function place(plate, el) {
    if (!el || !el.offsetParent) { plate.style.display = "none"; return; }
    const b = boxOf(el);
    plate.style.display = "";
    plate.style.left = `${b.left}px`;
    plate.style.bottom = `${b.bottom}px`;
    plate.style.width = `${b.width}px`;
    plate.style.height = `${b.height}px`;
  }

  // Whether the banner showed at the last sync (a line on screen, or paused).
  let bannerShown = false;

  // The banner's own height (content and padding), apart from the
  // min-height that stretches it to the newest line.
  function bannerNatural() {
    const parts = [...banner.children].filter(el => el.offsetParent !== null);
    return 24 + parts.reduce((h, el) => h + el.offsetHeight, 0)
      + 8 * Math.max(0, parts.length - 1);
  }

  // The notice column's bottom margin (Decision 40): 8 px above the newest
  // line's banner, which can rise past the line; above the paused banner
  // when paused with nothing on screen; otherwise on the anchor.
  function fitNotices(newest) {
    let under = 0;
    if (newest) under = NOTICE_GAP + Math.max(0, bannerNatural() - newest.offsetHeight);
    else if (state.paused) under = bannerNatural() + NOTICE_GAP;
    noticesBox.style.marginBottom = `${under}px`;
  }

  // Viewport tops of the live lines and the notices, taken before a change
  // that moves them, for glide().
  function stackTops() {
    return new Map([...liveLines(), ...noticesBox.children]
      .map(el => [el, el.getBoundingClientRect().top]));
  }

  // FLIP: each element in `tops` starts where it was and glides to where
  // the change put it (the .35 s glide, the Qt overlay's RESHAPE_MS). Only
  // the transform jumps: a running opacity fade keeps running (transition:
  // none would cancel it), and a glide in flight goes on from where it is.
  // Lines on their way out keep their own motion.
  function glide(tops) {
    if (state.reduceMotion) return;
    tops.forEach((top, el) => {
      if (!el.isConnected || el.classList.contains("leaving")
          || el.classList.contains("expiring")) return;
      const moved = top - el.getBoundingClientRect().top;
      if (!moved) return;
      const flight = new DOMMatrixReadOnly(getComputedStyle(el).transform).m42;
      el.style.transitionProperty = "opacity";
      el.style.transform = `translateY(${moved + flight}px)`;
      el.getBoundingClientRect();
      el.style.transitionProperty = "";
      el.style.transform = "";
    });
  }

  // Fit the anchored layer (plates and banner) to the newest line. With
  // `appearing` (the stack was empty, or the page was just unhidden) the
  // plates land at once and only their fade-in runs. The banner lands at
  // once when it appears and glides while it stays (a paused banner glides
  // from the anchor to a new line); when it goes it fades where it is. Same
  // rules as the Qt overlay.
  function sync(appearing) {
    const live = liveLines();
    const newest = live[live.length - 1];
    stage.classList.toggle("empty", !newest);
    // Nothing to measure while Ctrl+Alt+O hides the stage; unhiding re-syncs.
    if (stage.classList.contains("hidden")) return;
    fitNotices(newest);
    const wasShown = bannerShown;
    bannerShown = Boolean(newest) || state.paused;
    const snap = [];
    if (appearing || state.reduceMotion) snap.push(plateJp, plateEn);
    if (!wasShown || state.reduceMotion) snap.push(banner);
    snap.forEach(a => a.classList.add("instant"));
    if (newest) {
      const L = boxOf(newest);
      banner.style.left = `${L.left}px`;
      banner.style.bottom = `${L.bottom}px`;
      banner.style.minHeight = `${L.height}px`;
      place(plateJp, newest.querySelector(".jp"));
      place(plateEn, newest.querySelector(".en"));
    } else if (state.paused) {
      // Paused with nothing on screen: the banner waits at the anchor.
      banner.style.bottom = "0px";
      banner.style.minHeight = "0px";
      banner.style.left = body.classList.contains("align-center")
        ? "calc(50% - var(--gutter) / 2)" : "0px";
    }
    if (snap.length) {
      stage.getBoundingClientRect();
      snap.forEach(a => a.classList.remove("instant"));
    }
  }

  // Text blocks also change size without a new line: a Mode or size change
  // (the font-size transition runs after sync measured), a late web-font
  // swap. Refit the anchored layer whenever a block resizes so the plates
  // never keep a stale shape. A line's blocks are unobserved as it leaves.
  // The stage is observed too: a viewport resize moves a centered stack
  // without resizing any block.
  const resizer = new ResizeObserver(() => sync(false));
  resizer.observe(stage);

  function drop(el) {
    el.querySelectorAll(".jp, .en").forEach(block => resizer.unobserve(block));
    el.remove();
  }

  function lineEl(msg, en) {
    const el = document.createElement("div");
    el.className = "line";
    const html = jpHtml(msg);
    if (html) {
      const jp = document.createElement("div");
      jp.className = "jp";
      jp.innerHTML = html;  // escaped in jpHtml
      el.appendChild(jp);
    }
    const enEl = document.createElement("div");
    enEl.className = "en";
    enEl.textContent = en;
    el.appendChild(enEl);
    return el;
  }

  // A leaving line (pushed past max_lines, or faded after its hold) leaves
  // the layout and fades where it is, like the Qt overlay's ghosts: pinned
  // at its layout box, measured from the bottom of the lines box (which
  // stays put as lines come and go). A glide still running keeps running
  // (the leaving classes keep the transform transition).
  function detach(el, cls) {
    if (el.offsetParent) {  // nothing to measure while the stage is hidden
      const box = linesBox.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      const glide = new DOMMatrixReadOnly(getComputedStyle(el).transform).m42;
      el.style.left = `${r.left - box.left}px`;
      el.style.bottom = `${box.bottom - r.bottom + glide}px`;
      el.style.width = `${r.width}px`;
    }
    el.classList.add(cls);
  }

  function leave(el, ms) {
    clearTimeout(el._hold);
    detach(el, "leaving");
    setTimeout(() => drop(el), state.reduceMotion ? 0 : ms + 50);
  }

  // Lines past max_lines leave; `incoming` makes room for lines about to
  // be added.
  function trim(incoming = 0) {
    const live = liveLines();
    while (live.length + incoming > state.maxLines) leave(live.shift(), RESHAPE_MS);
    reage();
  }

  function expire(el) {
    if (!el.isConnected || el.classList.contains("leaving")) return;
    const tops = stackTops();
    detach(el, "expiring");
    setTimeout(() => drop(el), state.reduceMotion ? 0 : state.fadeOutMs + 50);
    reage();
    sync(false);
    glide(tops);
  }

  function addSubtitle(msg, en) {
    const before = liveLines();
    // The lines and the notices glide (FLIP below).
    const tops = stackTops();
    // Lines pushed past max_lines leave before the new line lands, so they
    // fade from where they are, like a line whose hold ended.
    trim(1);
    const ghosts = [...linesBox.querySelectorAll(".line.leaving, .line.expiring")];
    const el = lineEl(msg, en);
    if (!state.reduceMotion) el.classList.add("enter");
    linesBox.appendChild(el);
    el.querySelectorAll(".jp, .en").forEach(block => resizer.observe(block));
    reage();
    fitNotices(el);  // before the FLIP measures where everything lands
    if (!state.reduceMotion) {
      // Leaving lines rise with the stack by the new line's height, as in
      // the Qt overlay.
      const rise = el.getBoundingClientRect().height;
      ghosts.forEach(g => { g.style.bottom = `${parseFloat(g.style.bottom) + rise}px`; });
      // The lines still on screen and the notices start where they were and
      // glide up.
      glide(tops);
      requestAnimationFrame(() => requestAnimationFrame(() => el.classList.remove("enter")));
    }
    sync(before.length === 0);
    el._hold = setTimeout(() => expire(el), state.holdMs);
  }

  function isProblem(e) {
    return Boolean(e) && typeof e === "object" && KINDS.includes(e.kind)
      && ["title", "body", "line", "recovery"].every(k => typeof e[k] === "string");
  }

  function endRecovery(kind) {
    const shown = recoveries.get(kind);
    if (shown) {
      clearTimeout(shown.timer);
      recoveries.delete(kind);
    }
  }

  function showRecovery(kind, text) {
    endRecovery(kind);
    const timer = setTimeout(() => { recoveries.delete(kind); renderNotices(); }, RECOVERY_MS);
    recoveries.set(kind, { text, timer });
  }

  // One problems frame, diffed by kind against the last set (lead decision
  // 1): a kind that left shows its recovery line; a restarted pipeline's
  // carried recovery shows once per kind (Decision 18); a kind that is
  // active ends its recovery line.
  function onProblems(payload) {
    const next = new Map();
    (Array.isArray(payload.active) ? payload.active : []).forEach(e => {
      if (isProblem(e)) next.set(e.kind, e);
    });
    const gone = KINDS.filter(kind => problemLines.has(kind) && !next.has(kind));
    gone.forEach(kind => showRecovery(kind, problemLines.get(kind).recovery));
    (Array.isArray(payload.recovered) ? payload.recovered : []).forEach(e => {
      if (!e || typeof e !== "object" || !KINDS.includes(e.kind)) return;
      if (typeof e.recovery !== "string" || next.has(e.kind)) return;
      if (carriedShown.has(e.kind)) return;
      // A kind that left in this frame already shows its recovery line: the
      // carry is that recovery, so it counts as shown and a later reconnect
      // inside the carried window does not show it again.
      carriedShown.add(e.kind);
      if (!gone.includes(e.kind)) showRecovery(e.kind, e.recovery);
    });
    next.forEach((_, kind) => endRecovery(kind));
    problemLines.clear();
    next.forEach((e, kind) => problemLines.set(kind, e));
    renderNotices();
  }

  function noticeEl(kind, text, recovery) {
    const el = document.createElement("div");
    el.className = recovery ? "notice recovery" : "notice";
    el.dataset.kind = kind;
    const tag = span("nb");
    tag.textContent = NOTICE_LABEL;
    const bar = span("nt");
    bar.textContent = text;  // a device name from the frame: text, never markup
    el.append(tag, bar);
    return el;
  }

  // Fit the notice column to the problem and recovery lines, translator
  // above audio. An entry whose text or line kind changed is replaced: the
  // new one fades in, the old one goes at once (Decision 25).
  function renderNotices() {
    const tops = stackTops();
    const have = new Map([...noticesBox.children].map(el => [el.dataset.kind, el]));
    const want = [];
    KINDS.forEach(kind => {
      const problem = problemLines.get(kind);
      const shown = recoveries.get(kind);
      if (!problem && !shown) return;
      const text = problem ? problem.line : shown.text;
      const recovery = !problem;
      let el = have.get(kind);
      if (el && (el.querySelector(".nt").textContent !== text
                 || el.classList.contains("recovery") !== recovery)) {
        el = null;
      }
      if (!el) {
        el = noticeEl(kind, text, recovery);
        if (!state.reduceMotion) {
          el.classList.add("enter");
          requestAnimationFrame(() => requestAnimationFrame(() => el.classList.remove("enter")));
        }
      }
      want.push(el);
    });
    noticesBox.replaceChildren(...want);
    sync(false);
    glide(tops);
  }

  // The status pill is feedback, not furniture: OBS captures this page,
  // so a healthy badge must not sit in the corner of the stream forever.
  // After a short confirmation window a healthy ("connected"-class)
  // status fades out via the .faded CSS transition. Every status change
  // rebuilds className below (dropping .faded) and clears the pending
  // timer, so a disconnect/error re-shows the pill immediately.
  let statusFadeTimer = null;
  const STATUS_FADE_DELAY_MS = 4000;

  function setStatus(text, cls) {
    status.textContent = text;
    status.className = "status " + cls;
    // Stay hidden while the overlay is toggled off (Ctrl+Alt+O): a state
    // frame arriving mid-hide must not pop the badge back onto the capture.
    if (stage.classList.contains("hidden")) {
      status.classList.add("hidden");
    }
    clearTimeout(statusFadeTimer);
    if (cls === "connected") {
      statusFadeTimer = setTimeout(
        () => status.classList.add("faded"), STATUS_FADE_DELAY_MS
      );
    }
  }

  // NOTE: one reconnect counter per page load -- intentionally module-scoped.
  // If a second WebSocket is ever added to this file, convert these to
  // closure-captured state on the connection factory.
  let reconnectDelayMs = 1000;
  const RECONNECT_CAP_MS = 30000;

  function scheduleReconnect() {
    setTimeout(connect, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, RECONNECT_CAP_MS);
  }

  async function fetchToken() {
    // Per-session WS auth token. Served same-origin only -- the
    // /ws-token response carries no CORS headers, so a page from any
    // other local origin cannot read it. If the fetch fails (server
    // restarting), connect with an empty token; the WS open then fails
    // and the normal reconnect/backoff path retries everything.
    try {
      const resp = await fetch("/ws-token");
      if (resp.ok) {
        const body = await resp.json();
        return body.token || "";
      }
    } catch (e) { /* fall through to empty token */ }
    return "";
  }

  async function connect() {
    const token = await fetchToken();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${proto}//${location.host}/ws?t=${encodeURIComponent(token)}`
    );
    ws.addEventListener("open", () => {
      reconnectDelayMs = 1000;
      setStatus("connected", "connected");
    });
    ws.addEventListener("close", () => {
      setStatus("disconnected -- retrying...", "disconnected");
      scheduleReconnect();
    });
    ws.addEventListener("error", () => ws.close());
    ws.addEventListener("message", (ev) => {
      let payload;
      try { payload = JSON.parse(ev.data); } catch (e) { return; }
      if (payload.type === "style") { applyStyle(payload); return; }
      if (payload.type === "problems") { onProblems(payload); return; }
      // A frame with no English (whitespace only) is skipped here; the Qt
      // overlay shows its Japanese alone.
      if (payload.type === "subtitle" && String(payload.en ?? "").trim()) {
        addSubtitle(payload, payload.en);
        return;
      }
      if (payload.type === "toggle_overlay") {
        stage.classList.toggle("hidden");
        // Ctrl+Alt+O means "nothing from the overlay on the capture":
        // the status badge must vanish and return with the stage.
        status.classList.toggle("hidden", stage.classList.contains("hidden"));
        // Lines may have come and gone while hidden: the anchored layer
        // reappears with the stage, fitted at once.
        if (!stage.classList.contains("hidden")) {
          bannerShown = false;
          sync(true);
        }
        return;
      }
      if (payload.type === "state") {
        state.paused = payload.paused === true;
        if (typeof payload.genre === "string") state.genre = payload.genre;
        if ("series_media_id" in payload) {
          state.seriesId = payload.series_media_id ?? null;
          state.seriesLabel = typeof payload.series_label === "string" ? payload.series_label : "";
          state.seriesNative = typeof payload.series_native === "string" ? payload.series_native : "";
        }
        retag();
        // Pause stops new audio intake, but a subtitle already emitted
        // downstream may still finish. Keep accepting subtitle frames;
        // this status only explains the capture boundary.
        if (payload.paused) {
          setStatus("paused: new audio stopped, in-flight lines may appear", "connected");
        } else {
          setStatus("connected", "connected");
        }
        return;
      }
    });
  }

  applySizes({});  // style defaults until the server's style frame arrives
  retag();
  setStatus("connecting...", "");
  connect();
})();
