/* ---- screen snapping with a spring ----
   A quick wheel flick (or PageDown, PageUp, Space) springs to the next or previous screen with a small overshoot.
   Slow trackpad scrolling, arrow keys and the scrollbar stay free; a slow scroll that ends close to a screen edge is pulled onto it.
   Inside a section taller than the screen, scrolling stays native until the next edge is within reach.
   Off on touch screens, narrow windows and with reduced motion. */
(() => {
  const NAV = 64, FLICK = 34, IDLE = 170;
  const fine = window.matchMedia('(pointer: fine)'), calm = window.matchMedia('(prefers-reduced-motion: reduce)');
  const on = () => fine.matches && !calm.matches && window.innerWidth > 860 && window.innerHeight >= 560 && !document.documentElement.classList.contains('modal-open');
  const sel = document.currentScript && document.currentScript.dataset.snap || 'main > section, main > .snap';
  let anim = 0, g = null, endT = 0;
  const vhc = () => window.innerHeight - NAV;
  function points(){
    const max = document.documentElement.scrollHeight - window.innerHeight, out = [];
    document.querySelectorAll(sel).forEach((s, i) => { const y = i === 0 ? 0 : Math.round(s.getBoundingClientRect().top + window.scrollY - NAV); out.push(Math.max(0, Math.min(max, y))); });
    out.push(max);
    return out.sort((a, b) => a - b).filter((p, i, a) => !i || p - a[i - 1] > 4);
  }
  function pick(dir){
    const pts = points(), y = window.scrollY, reach = vhc() * 1.12;
    if (dir > 0){ const n = pts.find(p => p > y + 2); return n !== undefined && n - y <= reach ? n : null; }
    const p = pts.slice().reverse().find(q => q < y - 2); return p !== undefined && y - p <= reach ? p : null;
  }
  const root = document.documentElement, to = y => window.scrollTo(0, y);
  function stop(){ if (anim){ cancelAnimationFrame(anim); anim = 0; root.style.scrollBehavior = ''; } }
  function spring(target, v0){
    stop(); root.style.scrollBehavior = 'auto';
    const k = 190, c = 2 * .74 * Math.sqrt(k), st = { x: window.scrollY, v: v0 };
    let last = performance.now();
    const step = now => {
      const dt = Math.min(.034, (now - last) / 1000); last = now;
      for (let i = 0; i < 4; i++){ const h = dt / 4, a = -k * (st.x - target) - c * st.v; st.v += a * h; st.x += st.v * h; }
      to(st.x);
      if (Math.abs(st.x - target) < .6 && Math.abs(st.v) < 10){ to(target); anim = 0; root.style.scrollBehavior = ''; return; }
      anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }
  function settle(){
    if (!on() || anim) return;
    const y = window.scrollY, pts = points();
    let best = null, bd = 1e9; pts.forEach(p => { const d = Math.abs(p - y); if (d < bd){ bd = d; best = p; } });
    if (best !== null && bd > 2 && bd < vhc() * .1) spring(best, 0);
  }
  window.addEventListener('wheel', e => {
    if (!on() || e.ctrlKey || Math.abs(e.deltaX) > Math.abs(e.deltaY) || (e.target.closest && e.target.closest('[data-nosnap]'))) return;
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? window.innerHeight : 1), now = performance.now();
    if (!dy) return;
    if (!g || now - g.t > IDLE || (g.mode === 'free' && Math.sign(dy) !== g.dir)) g = { dir: Math.sign(dy), mode: 'free', t0: now, n: 0, sum: 0 };
    g.t = now; g.n++; g.sum += dy;
    clearTimeout(endT);
    if (g.mode === 'jump'){ e.preventDefault(); return; }
    if (anim){ stop(); }
    const flick = Math.abs(dy) >= FLICK || (now - g.t0 < 140 && Math.abs(g.sum) > 70);
    if (flick){
      const t = pick(g.dir);
      if (t !== null){ e.preventDefault(); g.mode = 'jump'; spring(t, g.dir * 700); return; }
    }
    endT = setTimeout(settle, IDLE + 40);
  }, { passive: false });
  window.addEventListener('keydown', e => {
    if (!on() || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target, typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(t.tagName));
    const dir = e.key === 'PageDown' || (e.key === ' ' && !e.shiftKey) ? 1 : e.key === 'PageUp' || (e.key === ' ' && e.shiftKey) ? -1 : 0;
    if (!dir || (typing && e.key === ' ')) { if (anim) stop(); return; }
    const tg = pick(dir); if (tg === null) return;
    e.preventDefault(); spring(tg, dir * 500);
  });
  ['mousedown', 'touchstart'].forEach(ev => window.addEventListener(ev, stop, { passive: true }));
})();
