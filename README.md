# Project website

This branch serves the project website for *Sparse Internal Control of Language Models* (NeurIPS 2026) at https://sparsecontrol.xizhe.net through GitHub Pages.

- `index.html`, `404.html`, `favicon.svg`, `og.png`, `paper.pdf`: the published site
- `CNAME`: custom domain
- `source/page.html`: the single source of the page
- `source/build.py`: rebuilds the site from the source

To rebuild, copy `source/page.html` to `src/page.html` and `source/og.png` to `assets/og.png` next to `build.py`, then run `python3 build.py path/to/paper.pdf`. The output lands in `dist/site/`.

The code and measurement records for the paper live on the `main` branch.
