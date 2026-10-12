# docs-site

Source for docs.mojolearn.org (Sphinx, PyData theme, MyST Markdown). Distinct from `docs/`,
which holds the project's internal engineering notes.

Build against the release being documented, never a source tree:

```sh
python -m venv ~/mojolearn-evidence/docs-venv
~/mojolearn-evidence/docs-venv/bin/pip install -r docs-site/requirements.txt "mojolearn==<release>"
cd docs-site
../../docs-venv/bin/python -I tools/gen_api.py          # api/*.rst from mojolearn.__all__; fails on an unplaced name
~/mojolearn-evidence/docs-venv/bin/sphinx-build -b html . _build/html -q -w _build/warnings.log
```

`tools/gen_api.py` places every public name on one family page; add a new module or name to
its `FAMILIES` table when it fails. Deployed as the `mojolearn-docs` Worker (static assets)
on docs.mojolearn.org: `npx wrangler deploy` from `docs-site/`.
