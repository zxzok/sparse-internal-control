# Project website

This branch serves the project website for *Sparse Internal Control of Language Models* (NeurIPS 2026) at https://sparsecontrol.xizhe.net through GitHub Pages.

- `index.html`: home page (framework, six-axis protocol, experiments, resources)
- `theory.html`: the mathematical model, local controllability and the six results
- `protocol.html`: the six-axis protocol, axis by axis: definitions, thresholds, results and code
- `404.html`, `favicon.svg`, `og.png`, `paper.pdf`, `robots.txt`, `sitemap.xml`: supporting files
- `CNAME`: custom domain
- `source/page.html`, `source/theory.html`, `source/protocol.html`: page sources
- `source/snap.js`: screen snapping shared by the pages, inlined at build time
- `source/scenes.js`: the six axis animations, shared by the home and protocol pages
- `source/i18n.js`, `source/zh.json`: English and Chinese. The switch in the navigation overrides the default and is remembered; without a choice, visitors whose IP is in mainland China (looked up through api.country.is, falling back to the time zone) see Chinese, everyone else English. `?lang=zh` or `?lang=en` forces either.
- `source/experiments.json`: per-task six-axis numbers and curves for the experiment results panel, extracted from `results/` and `data/measurements/` on `main`
- `source/build.py`: rebuilds the site from the sources

To rebuild, copy the page sources, `snap.js`, `scenes.js`, `i18n.js` and `zh.json` to `src/`, `source/experiments.json` to `data/`, and `source/og.png` to `assets/og.png`, next to `build.py` (the KaTeX stylesheet and fonts go in `assets/katex/`), then run `python3 build.py path/to/paper.pdf [arxiv_id]`. Passing an arXiv ID adds the BibTeX eprint. The output lands in `dist/site/`.

The code and measurement records for the paper live on the `main` branch.
