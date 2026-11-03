/* ---- six axis scenes: one drawing per axis, used as a live thumbnail and as the full stage ---- */
const S = { tr: '#45453F', g: '#8B8B88', cr: '#F4F3EE', li: '#D7EE5B', lb: '#BDBDB8' };
const ease = t => 1 - Math.pow(1 - t, 3);
const wait = ms => new Promise(r => setTimeout(r, ms));
const tw = (ms, f) => new Promise(res => { if (reduce){ f(1); return res(); } const t0 = performance.now(); const st = n => { const k = Math.min(1, (n - t0) / ms); f(ease(k)); if (k < 1) requestAnimationFrame(st); else res(); }; requestAnimationFrame(st); });
const mk = (p, n, a) => { const e = el(n, a); p.appendChild(e); return e; };
const tx = (p, x, y, s, a = {}) => { const t = mk(p, 'text', Object.assign({ x, y, class: 'lb', fill: S.lb, 'font-size': 15, 'text-anchor': 'middle' }, a)); t.dataset.en = s; t.textContent = I18N.t(s); return t; };
I18N.on(() => document.querySelectorAll('svg text[data-en]').forEach(t => { t.textContent = I18N.t(t.dataset.en); }));
const arrow = (p, x, y, dx, dy, col) => { const l = Math.hypot(dx, dy), ux = dx / l, uy = dy / l, s = 9; return mk(p, 'path', { d: `M${x - ux * s - uy * s * .6} ${y - uy * s + ux * s * .6} L${x} ${y} L${x - ux * s + uy * s * .6} ${y - uy * s - ux * s * .6}`, fill: 'none', stroke: col, 'stroke-width': 2.5 }); };

const SCENES = {
  1(s){ // reachability
    const y = 104, xs = 70, xt = 525, xp = 405;
    mk(s, 'line', { x1: xs, y1: y, x2: xt, y2: y, stroke: S.tr, 'stroke-width': 3 });
    const ln = mk(s, 'line', { x1: xs, y1: y, x2: xs, y2: y, stroke: S.li, 'stroke-width': 4 });
    mk(s, 'circle', { cx: xs, cy: y, r: 8, fill: S.g });
    mk(s, 'circle', { cx: xt, cy: y, r: 17, fill: 'none', stroke: S.cr, 'stroke-width': 3 });
    const ah = arrow(s, 0, y, 1, 0, S.li);
    const dot = mk(s, 'circle', { cx: xs, cy: y, r: 11, fill: S.li });
    tx(s, xs, y + 42, 'Source'); tx(s, xt, y + 46, 'Target');
    const pl = tx(s, xp, y - 26, 'Patched', { fill: S.cr });
    const res = mk(s, 'g', { opacity: 0 });
    mk(res, 'path', { d: `M${xp} ${y - 70}v14M${xp} ${y - 63}H${xt - 17}M${xt - 17} ${y - 70}v14`, stroke: S.li, 'stroke-width': 2.5, 'stroke-dasharray': '6 5', fill: 'none' });
    tx(res, (xp + xt) / 2, y - 80, 'Residual');
    const at = x => { dot.setAttribute('cx', x); ln.setAttribute('x2', Math.max(xs, x - 14)); pl.setAttribute('x', x); ah.setAttribute('transform', `translate(${x - 13} 0)`); ah.setAttribute('opacity', x - xs > 30 ? 1 : 0); };
    return {
      async play(){ res.setAttribute('opacity', 0); pl.setAttribute('opacity', 0); at(xs); await tw(1100, k => at(lerp(xs, xp, k))); pl.setAttribute('opacity', 1); await tw(500, k => res.setAttribute('opacity', k)); },
      final(){ at(xp); res.setAttribute('opacity', 1); pl.setAttribute('opacity', 1); }
    };
  },
  2(s){ // signed reversal
    const y = 112, xs = 70, xt = 525, xe = 190, ex = 118, ey = y - 26;
    mk(s, 'line', { x1: xs, y1: y, x2: xt, y2: y, stroke: S.tr, 'stroke-width': 3 });
    mk(s, 'circle', { cx: xs, cy: y, r: 8, fill: S.g });
    mk(s, 'circle', { cx: xt, cy: y, r: 17, fill: 'none', stroke: S.cr, 'stroke-width': 3 });
    const arc = mk(s, 'path', { d: `M${xt - 12} ${y - 30} Q ${(xs + xt) / 2 + 20} ${y - 128} ${ex} ${ey}`, fill: 'none', stroke: S.li, 'stroke-width': 3, pathLength: 1, 'stroke-dasharray': '1 1', 'stroke-dashoffset': 1 });
    const head = arrow(s, ex, ey, ex - ((xs + xt) / 2 + 20), ey - (y - 128), S.li); head.setAttribute('opacity', 0);
    const b = tx(s, (xs + xt) / 2 + 20, 24, '−β', { fill: S.cr, 'font-size': 22 }); b.setAttribute('opacity', 0);
    const dot = mk(s, 'circle', { cx: xt, cy: y, r: 11, fill: S.li });
    tx(s, xs, y + 42, 'Source'); tx(s, xt, y + 46, 'Target');
    const set = (a, d) => { arc.setAttribute('stroke-dashoffset', 1 - a); head.setAttribute('opacity', a > .98 ? 1 : 0); b.setAttribute('opacity', a); dot.setAttribute('cx', lerp(xt, xe, d)); };
    return {
      async play(){ set(0, 0); await wait(200); await tw(800, k => set(k, 0)); await tw(1000, k => set(1, k)); },
      final(){ set(1, 1); }
    };
  },
  3(s, onSet){ // dose response
    const X0 = 92, X1 = 560, Y0 = 136, Y1 = 40;
    const P = g => 1 / (1 + Math.exp(-8.5 * (g - .42))), p0 = P(0), p1 = P(1), Pn = g => (P(g) - p0) / (p1 - p0);
    const fx = g => X0 + g * (X1 - X0), fy = g => Y0 - Pn(g) * (Y0 - Y1);
    const path = (a, b) => { let d = ''; for (let i = 0; i <= 60; i++){ const g = a + (b - a) * i / 60; d += (i ? 'L' : 'M') + fx(g).toFixed(1) + ' ' + fy(g).toFixed(1); } return d; };
    mk(s, 'path', { d: `M${X0} ${Y1 - 12}V${Y0}H${X1 + 8}`, fill: 'none', stroke: S.tr, 'stroke-width': 2.5 });
    for (let i = 0; i < 10; i++){ const g = .05 + i * .1; mk(s, 'line', { x1: fx(g), y1: Y0 + 3, x2: fx(g), y2: Y0 + 10, stroke: S.g, 'stroke-width': 2 }); }
    mk(s, 'path', { d: path(0, 1), fill: 'none', stroke: S.tr, 'stroke-width': 3 });
    const act = mk(s, 'path', { d: path(0, .01), fill: 'none', stroke: S.li, 'stroke-width': 3.5 });
    const vl = mk(s, 'line', { stroke: S.cr, 'stroke-width': 1.5, 'stroke-dasharray': '4 4' });
    const dot = mk(s, 'circle', { r: 9, fill: S.li });
    tx(s, X0 + 10, Y1 - 18, 'Target progress', { 'text-anchor': 'start' }); tx(s, X1 + 8, Y0 + 30, 'Dose γ', { 'text-anchor': 'end' });
    for (let i = 0; i < 9; i++) mk(s, 'rect', { x: 424 + i * 13, y: 8, width: 9, height: 9, fill: i === 2 || i === 6 ? S.li : S.g });
    tx(s, 414, 17, 'Support fixed', { 'text-anchor': 'end', 'font-size': 13 });
    const set = g => { g = clamp(g, .005, 1); act.setAttribute('d', path(0, g)); dot.setAttribute('cx', fx(g)); dot.setAttribute('cy', fy(g)); vl.setAttribute('x1', fx(g)); vl.setAttribute('x2', fx(g)); vl.setAttribute('y1', fy(g) + 9); vl.setAttribute('y2', Y0); if (onSet) onSet(g); };
    set(.8);
    return {
      set,
      async play(){ await tw(1700, k => set(.04 + .92 * k)); await wait(500); await tw(1100, k => set(.96 - .76 * k)); },
      final(){ set(.8); }
    };
  },
  4(s, onStep){ // feedback controllability
    const ty = 30, xa = 92, xT = 470;
    mk(s, 'line', { x1: xa, y1: ty, x2: 540, y2: ty, stroke: S.tr, 'stroke-width': 2.5 });
    mk(s, 'circle', { cx: xT, cy: ty, r: 13, fill: 'none', stroke: S.cr, 'stroke-width': 3 });
    const err = mk(s, 'line', { y1: ty - 20, y2: ty - 20, x2: xT, stroke: S.li, 'stroke-width': 2, 'stroke-dasharray': '6 5' });
    const dot = mk(s, 'circle', { cy: ty, r: 9, fill: S.li });
    tx(s, 548, ty + 5, 'Target', { 'text-anchor': 'start', 'font-size': 14 });
    tx(s, xa, ty + 30, 'Target error', { 'text-anchor': 'start', 'font-size': 14 });
    const ly = 98;
    mk(s, 'circle', { cx: 112, cy: ly, r: 18, fill: 'none', stroke: S.cr, 'stroke-width': 2.5 });
    tx(s, 112, ly + 6, 'e', { fill: S.cr, 'font-size': 18 });
    mk(s, 'line', { x1: 130, y1: ly, x2: 214, y2: ly, stroke: S.g, 'stroke-width': 2.5 }); arrow(s, 214, ly, 1, 0, S.g);
    mk(s, 'rect', { x: 218, y: ly - 21, width: 112, height: 42, fill: 'none', stroke: S.cr, 'stroke-width': 2.5 }); tx(s, 274, ly + 6, 'Dose γ', { fill: S.cr, 'font-size': 17 });
    mk(s, 'line', { x1: 330, y1: ly, x2: 384, y2: ly, stroke: S.li, 'stroke-width': 2.5 }); arrow(s, 384, ly, 1, 0, S.li);
    mk(s, 'rect', { x: 388, y: ly - 21, width: 112, height: 42, fill: 'none', stroke: S.cr, 'stroke-width': 2.5 }); tx(s, 444, ly + 6, 'Model', { fill: S.cr, 'font-size': 17 });
    mk(s, 'path', { d: `M444 ${ly + 21}V${ly + 46}H112V${ly + 20}`, fill: 'none', stroke: S.li, 'stroke-width': 2.5 }); arrow(s, 112, ly + 20, 0, -1, S.li);
    tx(s, 278, 166, 'Measured output', { 'font-size': 14 });
    const loop = el('path', { d: `M130 ${ly}H444V${ly + 46}H112V${ly + 18}` });
    const pulse = mk(s, 'circle', { r: 6, fill: S.li, opacity: 0 });
    s.appendChild(loop); loop.setAttribute('fill', 'none'); loop.setAttribute('stroke', 'none');
    let m = .12, n = 0, busy = false;
    const px = v => lerp(xa + 14, xT, v);
    const show = () => { dot.setAttribute('cx', px(m)); err.setAttribute('x1', px(m) + 10); err.setAttribute('opacity', 1 - m > .04 ? 1 : 0); if (onStep) onStep(n); };
    show();
    const reset = () => { m = .12; n = 0; show(); };
    async function step(){
      if (busy) return; busy = true;
      if (1 - m < .04) { reset(); await wait(250); }
      const L = loop.getTotalLength(); pulse.setAttribute('opacity', 1);
      await tw(950, k => { const p = loop.getPointAtLength(k * L); pulse.setAttribute('cx', p.x); pulse.setAttribute('cy', p.y); });
      pulse.setAttribute('opacity', 0);
      const a = m, b = m + .58 * (1 - m); n++;
      await tw(550, k => { m = lerp(a, b, k); dot.setAttribute('cx', px(m)); err.setAttribute('x1', px(m) + 10); });
      show(); busy = false;
    }
    return {
      step,
      async play(){ reset(); await wait(300); for (let i = 0; i < 4; i++){ await step(); await wait(200); } },
      final(){ m = .97; n = 4; show(); }
    };
  },
  5(s){ // off-target preservation
    const base = 146, Hh = 112, xs = [170, 270, 370, 470], b = [.86, .58, .38, .2], q = [.8, .65, .42, .23];
    mk(s, 'line', { x1: 100, y1: base, x2: 540, y2: base, stroke: S.tr, 'stroke-width': 2.5 });
    mk(s, 'line', { x1: 100, y1: base, x2: 100, y2: 22, stroke: S.tr, 'stroke-width': 2.5 });
    tx(s, 110, 24, 'Probability', { 'text-anchor': 'start', 'font-size': 14 });
    const bars = [], marks = [];
    xs.forEach((x, i) => {
      mk(s, 'rect', { x: x - 26, y: base - b[i] * Hh, width: 22, height: b[i] * Hh, fill: S.g });
      bars.push(mk(s, 'rect', { x: x + 2, width: 22, fill: S.li }));
      const g = mk(s, 'g', { opacity: 0 });
      mk(g, 'line', { x1: x - 30, x2: x + 30, y1: base - b[i] * Hh, y2: base - b[i] * Hh, stroke: S.cr, 'stroke-width': 1.6, 'stroke-dasharray': '3 3' });
      mk(g, 'line', { x1: x + 34, x2: x + 34, y1: base - b[i] * Hh, y2: base - q[i] * Hh, stroke: S.cr, 'stroke-width': 2.5 });
      marks.push(g);
      tx(s, x, base + 20, 'ABCD'[i], { 'font-size': 15 });
    });
    const set = (k, mk2) => { bars.forEach((r, i) => { const h = lerp(b[i], q[i], k) * Hh; r.setAttribute('y', base - h); r.setAttribute('height', h); }); marks.forEach(g => g.setAttribute('opacity', mk2)); };
    set(0, 0);
    return {
      async play(){ set(0, 0); await wait(250); await tw(900, k => set(k, 0)); await tw(450, k => set(1, k)); },
      final(){ set(1, 1); }
    };
  },
  6(s){ // held-out reuse
    const T = 17, G = 4, x0 = 46, y0 = 40, on = [2, 5, 11];
    for (let i = 0; i < 16; i++) mk(s, 'rect', { x: x0 + (i % 4) * (T + G), y: y0 + Math.floor(i / 4) * (T + G), width: T, height: T, fill: on.includes(i) ? S.li : S.g });
    tx(s, x0 + 40, y0 + 112, 'Same sites', { 'font-size': 14 });
    mk(s, 'path', { d: `M136 ${y0 + 40}H176M176 ${y0 + 2}V${y0 + 82}M176 ${y0 + 2}H204M176 ${y0 + 82}H204`, fill: 'none', stroke: S.cr, 'stroke-width': 2 });
    arrow(s, 206, y0 + 2, 1, 0, S.cr); arrow(s, 206, y0 + 82, 1, 0, S.cr);
    const rows = [[22, 'Fixed sites · refit strengths', '0.992', [.99, .96, .93]], [104, 'Full patch fixed', '0.021', [.05, .02, .08]]];
    const fills = [];
    rows.forEach(([ly, lab, val, vs]) => {
      tx(s, 222, ly, lab, { 'text-anchor': 'start', fill: S.cr, 'font-size': 15, 'font-weight': 600 });
      tx(s, 560, ly, val, { 'text-anchor': 'end', fill: S.li, 'font-size': 14, class: 'lb mono' });
      vs.forEach((v, j) => { const y = ly + 12 + j * 11; mk(s, 'rect', { x: 222, y, width: 338, height: 6, fill: S.tr }); fills.push([mk(s, 'rect', { x: 222, y, width: 0, height: 6, fill: S.li }), v]); });
    });
    const set = k => fills.forEach(([r, v], i) => r.setAttribute('width', 338 * v * clamp(k * 1.25 - (i % 3) * .1, 0, 1)));
    return {
      async play(){ set(0); await wait(250); await tw(1300, set); },
      final(){ set(1); }
    };
  }
};
