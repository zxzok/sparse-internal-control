#!/usr/bin/env python3
"""Build the Sparse Internal Control site.

src/page.html is the single source. It produces:
  dist/artifact.html  preview version (no document skeleton, absolute paper link)
  dist/site/          deployable static site for GitHub Pages
"""
import json, pathlib, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src" / "page.html"
DIST = ROOT / "dist"
SITE = DIST / "site"
DOMAIN = "sparsecontrol.xizhe.net"
URL = f"https://{DOMAIN}/"
PAPER_PDF = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None

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

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><style>.s{stroke:#0E0E0C}.f{fill:#0E0E0C}@media (prefers-color-scheme:dark){.s{stroke:#F2F2EE}.f{fill:#F2F2EE}}</style><path class="s" d="M12 2.6 20.1 7.3v9.4L12 21.4 3.9 16.7V7.3Z" fill="none" stroke-width="1.6" stroke-linejoin="round"/><path class="s" d="M12 12 12 2.6" stroke-width="1.6" opacity=".45"/><circle class="f" cx="12" cy="12" r="1.9"/><circle cx="12" cy="2.9" r="2.6" fill="#E8590C"/></svg>
"""

NOT_FOUND = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Page not found · Sparse Internal Control</title><meta name="robots" content="noindex"><link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@500&display=swap">
<style>
:root{--bg:#FAFAF8;--ink:#0E0E0C;--muted:#5B5B55;--line:#C8C8C1;--drive:#C9480C}
@media (prefers-color-scheme:dark){:root{--bg:#181818;--ink:#F2F2EE;--muted:#A8A8A1;--line:#4A4A45;--drive:#FF8A3D;color-scheme:dark}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font-family:"Geist",system-ui,sans-serif;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{display:grid;gap:24px;max-width:680px}
.k{font-family:"Geist Mono",ui-monospace,monospace;font-size:12px;line-height:16px;letter-spacing:.08em;color:var(--drive)}
h1{margin:0;font-size:48px;line-height:48px;font-weight:600;letter-spacing:-.02em;text-wrap:balance}
p{margin:0;font-size:18px;line-height:28px;color:var(--muted);text-wrap:pretty}
a{justify-self:start;display:inline-flex;padding:8px 16px;border-radius:9999px;background:var(--ink);color:var(--bg);text-decoration:none;font-weight:600;transition:transform 300ms cubic-bezier(0.22,1,0.36,1)}
a:hover{transform:translateY(-1px)}a:active{transform:scale(.98)}a:focus-visible{outline:2px solid var(--drive);outline-offset:3px}
@media (max-width:720px){h1{font-size:30px;line-height:36px}}
</style></head>
<body><main><span class="k">404 · TARGET NOT REACHED</span><h1>This page is outside the target set</h1><p>The address does not match any page on this site. The paper, the framework and the six axes are on the home page.</p><a href="/">Back to the home page</a></main></body></html>
"""


def main():
    src = SRC.read_text(encoding="utf-8")
    DIST.mkdir(exist_ok=True)

    # Preview version for the Artifact tool
    (DIST / "artifact.html").write_text(src.replace("{{PAPER_URL}}", URL + "paper.pdf"), encoding="utf-8")

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
<meta name="theme-color" content="#FAFAF8" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#181818" media="(prefers-color-scheme: dark)">
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
    (SITE / "404.html").write_text(NOT_FOUND, encoding="utf-8")
    (SITE / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (SITE / "CNAME").write_text(DOMAIN + "\n", encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    (SITE / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {URL}sitemap.xml\n", encoding="utf-8")
    (SITE / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{URL}</loc></url></urlset>\n", encoding="utf-8")
    if (ROOT / "assets" / "og.png").exists():
        shutil.copyfile(ROOT / "assets" / "og.png", SITE / "og.png")
    if PAPER_PDF and PAPER_PDF.exists():
        shutil.copyfile(PAPER_PDF, SITE / "paper.pdf")
    print("built", sorted(p.name for p in SITE.iterdir()))


if __name__ == "__main__":
    main()
