#!/usr/bin/env python3
"""Build the Sparse Internal Control site.

src/page.html is the single source. It produces:
  dist/artifact.html  preview version (no document skeleton, absolute paper link)
  dist/site/          deployable static site for GitHub Pages
"""
import base64, json, pathlib, re, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src" / "page.html"
DIST = ROOT / "dist"
SITE = DIST / "site"
DOMAIN = "sparsecontrol.xizhe.net"
URL = f"https://{DOMAIN}/"
PAPER_PDF = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
ARXIV_ID = sys.argv[2] if len(sys.argv) > 2 else ""

DESC = ("Sparse internal control steers a frozen language model through a few internal sites at "
        "inference time. A six axis protocol measures reachability, signed reversal, dose response, "
        "feedback controllability, off-target preservation and held-out reuse.")

FAQ = [
    ("Does sparse internal control change the model's weights?",
     "No. The parameters stay frozen. An intervention adds scaled directions to a few activations during the forward pass, and setting beta to zero recovers the original model exactly."),
    ("What is a driver node?",
     "A candidate control node is a triple of a physical site, an intervention direction and a selection cost. The candidates that receive a nonzero strength form the driver set."),
    ("How is this different from activation steering?",
     "Activation steering shows that internal directions can change behavior. Sparse internal control adds the control problem around it: which compact set of sites to use, under a bounded energy, with explicit requirements for target recovery and off-target preservation, followed by an evaluation on six axes."),
    ("Why six axes instead of one score?",
     "Because the properties separate in practice. Refusal responds to dose in perfect order and still fails reversal. MCQA sites transfer when strengths are refit, while the fully frozen intervention barely moves held out prompts."),
    ("Can I evaluate my own steering method with the protocol?",
     "Yes. Each axis is defined on any intervention with a support, base strengths and a direction rule, and the released code has one metrics module per axis."),
]

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect width="24" height="24" rx="3" fill="#111111"/><rect x="4.5" y="4.5" width="6.5" height="6.5" fill="#D7EE5B"/><rect x="13" y="4.5" width="6.5" height="6.5" fill="#D7EE5B"/><rect x="4.5" y="13" width="6.5" height="6.5" fill="#D7EE5B"/><rect x="13" y="13" width="6.5" height="6.5" fill="#D7EE5B"/></svg>
"""

NOT_FOUND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Page not found · Sparse Internal Control</title><meta name="robots" content="noindex"><link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Anton&family=IBM+Plex+Mono:wght@600&family=Poppins:wght@400;600&display=swap">
<style>
:root{--black:#111111;--cream:#F4F3EE;--lime:#D7EE5B;--g3:#BDBDB8;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;align-content:center;background:var(--black);color:var(--cream);font-family:"Poppins",system-ui,sans-serif;padding:32px max(24px,env(safe-area-inset-right,0px)) 32px max(24px,env(safe-area-inset-left,0px))}
main{display:grid;gap:24px;max-width:1100px}
.k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:600;font-size:14px;letter-spacing:.08em;text-transform:uppercase;color:var(--lime)}
h1{margin:0;font-family:"Anton",sans-serif;font-weight:400;font-size:clamp(64px,13vw,190px);line-height:.95;text-transform:uppercase}
p{margin:0;font-size:20px;line-height:1.45;color:var(--g3);max-width:640px}
a{justify-self:start;display:inline-flex;align-items:center;gap:10px;padding:14px 26px;background:var(--lime);color:var(--black);text-decoration:none;font-weight:600;font-size:18px;transition:transform 300ms cubic-bezier(0.22,1,0.36,1)}
a:hover{transform:translateY(-2px)}a:active{transform:scale(.98)}a:focus-visible{outline:3px solid var(--cream);outline-offset:3px}
</style></head>
<body><main><span class="k">404 / Target not reached</span><h1>Outside the target set.</h1><p>The address does not match any page on this site. The framework, the six axes and the theory are one click away.</p><a href="/">Back to the home page</a></main></body></html>
"""


KATEX_V = "0.16.11"
KATEX_LINK = f'<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@{KATEX_V}/dist/katex.min.css" crossorigin="anonymous">'


def katex_inline_css():
    """KaTeX stylesheet with the fonts it needs embedded as data URIs (the preview host blocks CDN stylesheets)."""
    kdir = ROOT / "assets" / "katex"
    css = (kdir / "katex.min.css").read_text(encoding="utf-8")

    def face(m):
        block = m.group(0)
        f = re.search(r"url\(fonts/(KaTeX_[A-Za-z0-9_-]+)\.woff2\)", block)
        path = kdir / "fonts" / (f.group(1) + ".woff2") if f else None
        if not path or not path.exists():
            return ""
        data = base64.b64encode(path.read_bytes()).decode()
        return re.sub(r"src:[^}]*", f'src:url(data:font/woff2;base64,{data}) format("woff2")', block)

    css = re.sub(r"@font-face\{[^}]*\}", face, css)
    return "<style>" + css + "</style>"


def main():
    src = SRC.read_text(encoding="utf-8")
    # arXiv: set ARXIV_ID (e.g. "2611.01234") to add the arXiv button, footer link and BibTeX eprint
    if ARXIV_ID:
        src = re.sub(r"<!--/?ARXIV-->", "", src).replace("{{ARXIV_ID}}", ARXIV_ID)
        src = src.replace("{{ARXIV_BIB}}", f",\n  eprint    = {{{ARXIV_ID}}},\n  archivePrefix = {{arXiv}}")
        src = src.replace("{{ARXIV_BIB_JS}}", f",\\n  eprint    = {{{ARXIV_ID}}},\\n  archivePrefix = {{arXiv}}")
    else:
        src = re.sub(r"\s*<!--ARXIV-->.*?<!--/ARXIV-->", "", src, flags=re.S).replace("{{ARXIV_BIB}}", "").replace("{{ARXIV_BIB_JS}}", "")
    zh_path = ROOT / "src" / "zh.json"
    zh = json.loads(zh_path.read_text(encoding="utf-8")) if zh_path.exists() else {}
    src = src.replace("/*ZH_DICT*/{}", json.dumps(zh, ensure_ascii=False).replace("</", "<\\/"))
    snap = (ROOT / "src" / "snap.js").read_text(encoding="utf-8")
    src = src.replace("/*SNAP_JS*/", snap)
    theory = (ROOT / "src" / "theory.html").read_text(encoding="utf-8").replace("/*SNAP_JS*/", snap)
    DIST.mkdir(exist_ok=True)
    (DIST / "theory.html").write_text(
        theory.replace("{{PAPER_URL}}", URL + "paper.pdf").replace("<!--KATEX_CSS-->", katex_inline_css()).replace("<!--HEAD_META-->", ""), encoding="utf-8")

    # Preview version for the Artifact tool
    (DIST / "artifact.html").write_text(
        src.replace("{{PAPER_URL}}", URL + "paper.pdf").replace("<!--KATEX_CSS-->", katex_inline_css()), encoding="utf-8")
    src = src.replace("<!--KATEX_CSS-->", KATEX_LINK)

    # Deployable version
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    split = src.index('<svg width="0" height="0"')
    head_part, body_part = src[:split], src[split:]
    head_part = head_part.replace("<title>Sparse Internal Control</title>",
                                  "<title>Sparse Internal Control of Language Models</title>")
    head_part = head_part.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    ld = [
        {"@context": "https://schema.org", "@type": "ScholarlyArticle",
         "headline": "Sparse Internal Control of Language Models",
         "author": {"@type": "Person", "name": "Xizhe Zhang", "identifier": "https://orcid.org/0000-0002-8684-4591"},
         "datePublished": "2026", "description": DESC, "url": URL,
         "isPartOf": {"@type": "Event", "name": "NeurIPS 2026"},
         "codeRepository": "https://github.com/zxzok/sparse-internal-control"},
        {"@context": "https://schema.org", "@type": "FAQPage",
         "mainEntity": [{"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]},
    ]
    meta = f"""<link rel="canonical" href="{URL}">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<meta name="author" content="Xizhe Zhang">
<meta name="theme-color" content="#111111">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Sparse Internal Control">
<meta property="og:title" content="Sparse Internal Control of Language Models">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{URL}og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Sparse Internal Control of Language Models, NeurIPS 2026">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Sparse Internal Control of Language Models">
<meta name="twitter:description" content="{DESC}">
<meta name="twitter:image" content="{URL}og.png">
<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>
<style>body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}</style>
"""
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           + head_part + meta + "</head>\n<body>\n" + body_part + "\n</body>\n</html>\n")
    (SITE / "index.html").write_text(doc.replace("{{PAPER_URL}}", "paper.pdf"), encoding="utf-8")
    t_meta = f"""<link rel="canonical" href="{URL}theory.html">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<meta name="author" content="Xizhe Zhang">
<meta name="theme-color" content="#111111">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Sparse Internal Control">
<meta property="og:title" content="Control theory · Sparse Internal Control of Language Models">
<meta property="og:description" content="The intervention, the sparse control problem, local controllability of a frozen Transformer and the results that link them.">
<meta property="og:url" content="{URL}theory.html">
<meta property="og:image" content="{URL}og.png">
<meta name="twitter:card" content="summary_large_image">
<style>[hidden]{{display:none!important}}</style>"""
    (SITE / "theory.html").write_text(
        theory.replace("{{PAPER_URL}}", "paper.pdf").replace("<!--KATEX_CSS-->", KATEX_LINK).replace("<!--HEAD_META-->", t_meta), encoding="utf-8")
    (SITE / "404.html").write_text(NOT_FOUND, encoding="utf-8")
    (SITE / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (SITE / "CNAME").write_text(DOMAIN + "\n", encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    (SITE / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {URL}sitemap.xml\n", encoding="utf-8")
    (SITE / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{URL}</loc></url><url><loc>{URL}theory.html</loc></url></urlset>\n", encoding="utf-8")
    if (ROOT / "assets" / "og.png").exists():
        shutil.copyfile(ROOT / "assets" / "og.png", SITE / "og.png")
    if (ROOT / "assets" / "readme").is_dir():
        shutil.copytree(ROOT / "assets" / "readme", SITE / "readme")
    if PAPER_PDF and PAPER_PDF.exists():
        shutil.copyfile(PAPER_PDF, SITE / "paper.pdf")
    print("built", sorted(p.name for p in SITE.iterdir()))


if __name__ == "__main__":
    main()
