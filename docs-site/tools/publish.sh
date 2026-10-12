#!/usr/bin/env bash
# Build the docs for the installed mojolearn release and assemble the published site:
#   <site>/<version>/  this release, kept across later publishes
#   <site>/stable/     a copy of the newest release
#   <site>/_redirects  / -> /stable/
# The site directory persists in ~/mojolearn-evidence/docs-site-published so older
# releases stay published; deploy with `npx wrangler deploy` from docs-site/.
# Usage: tools/publish.sh [python]   (python defaults to ~/mojolearn-evidence/docs-venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-$HOME/mojolearn-evidence/docs-venv/bin/python}"
SITE="$HOME/mojolearn-evidence/docs-site-published"
VERSION="$("$PY" -I -c 'import mojolearn; print(mojolearn.__version__)')"
"$PY" -I tools/gen_api.py
rm -rf _build/html
mkdir -p _build
"$PY" -m sphinx -b html . _build/html -q -j 4 -w _build/warnings.log
echo "built $VERSION; $(wc -l < _build/warnings.log | tr -d ' ') warnings in _build/warnings.log"
mkdir -p "$SITE"
rm -rf "$SITE/$VERSION" "$SITE/stable"
cp -R _build/html "$SITE/$VERSION"
cp -R _build/html "$SITE/stable"
cat > "$SITE/404.html" <<'HTML'
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not found | mojolearn docs</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:36rem;margin:15vh auto;padding:0 16px}a{color:#30215A}@media(prefers-color-scheme:dark){body{background:#14181b;color:#e6e8e8}a{color:#A99BE0}}</style></head>
<body><h1>Page not found</h1><p>This page is not in the mojolearn documentation. Try the <a href="/stable/">current docs</a> or <a href="/stable/search.html">search</a>.</p></body></html>
HTML
printf '/ /stable/ 302\n' > "$SITE/_redirects"
# Version switcher: every published release, newest first, with stable/ marked preferred.
"$PY" -I - "$SITE" "$VERSION" <<'PYEOF'
import json, pathlib, sys
site, newest = pathlib.Path(sys.argv[1]), sys.argv[2]
key = lambda v: tuple(int(x) for x in v.split(".") if x.isdigit())
versions = sorted((d.name for d in site.iterdir() if d.is_dir() and d.name[0].isdigit()), key=key, reverse=True)
entries = [{"name": f"{v} (stable)" if v == newest else v, "version": v,
            "url": f"https://docs.mojolearn.org/{'stable' if v == newest else v}/",
            **({"preferred": True} if v == newest else {})} for v in versions]
(site / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")
(site / "robots.txt").write_text("User-agent: *\nAllow: /\nSitemap: https://docs.mojolearn.org/stable/sitemap.xml\n")
PYEOF
echo "site: $SITE ($(ls "$SITE" | tr '\n' ' '))"
rm -rf _site && cp -R "$SITE" _site
echo "staged _site for: npx wrangler deploy"
