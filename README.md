# Project website

This branch serves the project website for *Sparse Internal Control of Language Models* (NeurIPS 2026) at https://sparsecontrol.xizhe.net through GitHub Pages.

- `index.html`: home page (framework, six-axis protocol, experiments, resources)
- `theory.html`: the mathematical model, local controllability and the six results
- `404.html`, `favicon.svg`, `og.png`, `paper.pdf`, `robots.txt`, `sitemap.xml`: supporting files
- `CNAME`: custom domain
- `source/page.html`, `source/theory.html`: page sources
- `source/snap.js`: screen snapping shared by both pages, inlined at build time
- `source/build.py`: rebuilds the site from the sources

To rebuild, copy `source/page.html`, `source/theory.html` and `source/snap.js` to `src/`, and `source/og.png` to `assets/og.png`, next to `build.py` (the KaTeX stylesheet and fonts go in `assets/katex/`), then run `python3 build.py path/to/paper.pdf [arxiv_id]`. Passing an arXiv ID adds the BibTeX eprint. The output lands in `dist/site/`.

The code and measurement records for the paper live on the `main` branch.
