/* ---- language: English by default, Chinese for visitors in mainland China ----
   Order of precedence: ?lang= in the URL, then the visitor's own choice on the switch (remembered),
   then the IP country (CN shows Chinese), with the time zone as an instant first guess and as the fallback
   when the lookup is blocked or slow. Translations swap whole elements, so sentences keep their own order
   in each language; callers re-render math in the swapped elements and rebuild drawn labels. */
const I18N = (() => {
  const ZH = /*ZH_DICT*/{};
  const KEY = 'sic-lang';
  const norm = s => s.replace(/\s+/g, ' ').trim();
  const INLINE = new Set(['B', 'STRONG', 'I', 'EM', 'BR', 'CODE', 'SPAN', 'SUB', 'SUP', 'SMALL']);
  const ownText = el => Array.from(el.childNodes).some(n => n.nodeType === 3 && /\S/.test(n.nodeValue));
  let lang = 'en', items = [], ready = false, chosen = false;
  const subs = [];
  const store = (k, v) => { try { window.localStorage.setItem(k, v); } catch (e) {} };
  const read = (area, k) => { try { return window[area].getItem(k); } catch (e) { return null; } };
  function inline(el){ for (const c of el.children){ if (c.tagName === 'svg' ? !c.classList.contains('ic') : !INLINE.has(c.tagName) || !inline(c)) return false; } return true; }
  function collect(){
    const taken = new Set();
    document.querySelectorAll('body *').forEach(el => {
      if (el.closest('script, style, svg, .katex, [data-noi18n]')) return;
      for (let a = el.parentElement; a; a = a.parentElement) if (taken.has(a)) return;
      if (!inline(el) || !ownText(el)) return;
      const en = el.innerHTML, zh = ZH[norm(en)];
      if (zh === undefined) return;
      items.push({ el, en, zh }); taken.add(el);
    });
    items.titleEn = document.title;
  }
  function set(next, opts = {}){
    if (next !== 'zh') next = 'en';
    if (opts.user){ chosen = true; store(KEY, next); }
    if (next === lang && ready) return;
    lang = next;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    items.forEach(it => { it.el.innerHTML = lang === 'zh' ? it.zh : it.en; });
    document.title = lang === 'zh' ? (ZH[norm(items.titleEn)] || items.titleEn) : items.titleEn;
    document.querySelectorAll('[data-lang-btn]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.langBtn === lang)));
    if (ready) subs.forEach(f => { try { f(lang, items.map(i => i.el)); } catch (e) {} });
  }
  function geo(){
    if (!window.fetch) return Promise.resolve(null);
    const ctl = window.AbortController ? new AbortController() : null, timer = setTimeout(() => ctl && ctl.abort(), 1800);
    return fetch('https://api.country.is/', { signal: ctl ? ctl.signal : undefined, cache: 'no-store' })
      .then(r => (r.ok ? r.json() : null)).then(j => (j && typeof j.country === 'string' ? j.country.toUpperCase() : null))
      .catch(() => null).finally(() => clearTimeout(timer));
  }
  function init(){
    collect();
    const q = new URLSearchParams(location.search).get('lang'), saved = read('localStorage', KEY);
    let first = 'en', ask = false;
    if (q === 'zh' || q === 'en'){ first = q; chosen = true; }
    else if (saved === 'zh' || saved === 'en'){ first = saved; chosen = true; }
    else {
      const cc = read('sessionStorage', KEY + '-cc');
      if (cc) first = cc === 'CN' ? 'zh' : 'en';
      else {
        let tz = ''; try { tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch (e) {}
        first = /^(Asia\/(Shanghai|Chongqing|Chungking|Harbin|Urumqi|Kashgar)|PRC)$/.test(tz) ? 'zh' : 'en';
        ask = true;
      }
    }
    set(first);
    ready = true;
    document.querySelectorAll('[data-lang-btn]').forEach(b => b.addEventListener('click', () => set(b.dataset.langBtn, { user: true })));
    if (ask) geo().then(cc => { if (!cc) return; try { sessionStorage.setItem(KEY + '-cc', cc); } catch (e) {} if (!chosen) set(cc === 'CN' ? 'zh' : 'en'); });
    return lang;
  }
  return {
    init, set,
    on: f => subs.push(f),
    get lang(){ return lang; },
    t: s => (lang === 'zh' && ZH[s] !== undefined ? ZH[s] : s),
    L: (en, zh) => (lang === 'zh' ? zh : en)
  };
})();
