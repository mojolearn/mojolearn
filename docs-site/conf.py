# Sphinx configuration for docs.mojolearn.org. Built against the installed mojolearn
# release (pip install mojolearn==<version>), so the API reference is what users install.
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import mojolearn

project = "mojolearn"
author = "Andrew Hendel"
copyright = "2026, Andrew Hendel"
release = mojolearn.__version__
version = release


# Release facts the prose restates, filled in at build time so no page hand-types them: {{release}} is the
# documented version; {{nvidia_arches}} / {{amd_arches}} (and *_examples) are the GPU architectures that
# release's vendor packages ship, read from their PyPI summaries ("... binaries for mojolearn <version>
# (<arch>, <arch>); ..." written by packaging/linux/pack_wheel.py), the same source mojolearn.org uses.
# MOJOLEARN_DOCS_NVIDIA_ARCHES / MOJOLEARN_DOCS_AMD_ARCHES ("sm_89" or "a,b") override that for an offline build.
ARCH_EXAMPLES = {
    "sm_80": "A100, A30", "sm_86": "RTX 30, A40", "sm_89": "L40S, L40, L4, RTX 4090, RTX 6000 Ada",
    "sm_90": "H100, H200", "sm_90a": "H100, H200", "sm_100": "B200", "sm_120": "RTX 50",
    "gfx90a": "Instinct MI210, MI250X", "gfx942": "Instinct MI300X, MI325X", "gfx950": "Instinct MI350X, MI355X",
    "gfx1100": "Radeon RX 7900",
}


def _shipped_arches(package, env):
    import json
    import os
    import re
    import urllib.request
    if os.environ.get(env):
        return [a.strip() for a in os.environ[env].split(",") if a.strip()]
    url = f"https://pypi.org/pypi/{package}/{release}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            summary = json.load(r)["info"]["summary"]
    except Exception as e:
        raise RuntimeError(f"conf.py: cannot read {url} ({e}); set {env} for an offline build") from e
    m = re.search(r"mojolearn [0-9][^ ]* \(([^)]*)\)", summary)
    if not m:
        raise RuntimeError(f"conf.py: no architecture list in the {package} {release} summary: {summary!r}")
    return [a.strip() for a in m.group(1).split(",") if a.strip()]


def _facts():
    facts = {"release": release}
    for vendor, package in (("nvidia", "mojolearn-nvidia"), ("amd", "mojolearn-amd")):
        arches = _shipped_arches(package, f"MOJOLEARN_DOCS_{vendor.upper()}_ARCHES")
        facts[f"{vendor}_arches"] = ", ".join(f"`{a}`" for a in arches)
        facts[f"{vendor}_examples"] = ", ".join(ARCH_EXAMPLES.get(a, a) for a in arches)
    return facts


FACTS = _facts()


def _fill_facts(app, docname, source):
    # Every page's source, before parsing: works in prose, tables and code blocks alike.
    for key, value in FACTS.items():
        source[0] = source[0].replace("{{" + key + "}}", value)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_site", "site_root", "tools", "README.md"]
templates_path = ["_templates"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = True
# A module page lists only the module's own public names (its __all__), not helpers it imports.
autosummary_ignore_module_all = False
# Names that differ only in case (the hdbscan module and the HDBSCAN class) would share one file on a
# case-insensitive disk; give the lower-case ones a distinct page name.
_lower = {}
for _n in mojolearn.__all__:
    _lower.setdefault(_n.lower(), []).append(_n)
autosummary_filename_map = {f"mojolearn.{n}": f"mojolearn.{n}-module"
                            for group in _lower.values() if len(group) > 1 for n in group if n == n.lower()}
autodoc_default_options = {"members": True, "inherited-members": False, "show-inheritance": False,
                           "exclude-members": "__init__"}
# Docstrings write names in single backticks; render them as code, not as italic title references.
default_role = "code"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
# Types in docstrings are prose ("float32 array of shape (n,)"); render them as text, not cross-references.
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = True
# Keep trailing underscores in attribute names (labels_, coef_); without this rST eats them as link syntax.
strip_signature_backslash = True
# Docstrings cite source paths and references in prose; keep warnings from them non-fatal.
suppress_warnings = ["autosummary.import_cycle"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"mojolearn {release} documentation"
html_static_path = ["_static"]
html_css_files = ["mojolearn.css"]
# Light/dark choice shared with mojolearn.org through a cookie on the parent domain.
html_js_files = ["theme-cookie.js"]
html_favicon = "_static/mojolearn-mark-full.svg"
html_theme_options = {
    "logo": {
        "text": "mojolearn",
        "image_light": "_static/mojolearn-mark.svg",
        "image_dark": "_static/mojolearn-mark-full.svg",
        "link": "https://mojolearn.org",
    },
    "external_links": [
        {"name": "Get started", "url": "https://mojolearn.org/get-started"},
        {"name": "Measurements", "url": "https://mojolearn.org/measurements"},
    ],
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/mojolearn/mojolearn", "icon": "fa-brands fa-github"},
        {"name": "PyPI", "url": "https://pypi.org/project/mojolearn/", "icon": "fa-brands fa-python"},
    ],
    "navbar_align": "left",
    # One published release (the current one), so no version switcher.
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {"default_mode": "auto"}
html_show_sourcelink = False
# Publish no page sources (_sources/); the build tooling also keeps doctrees and .buildinfo out of the output.
html_copy_source = False
# The site root is the current release (older releases are not kept).
html_baseurl = "https://docs.mojolearn.org/"
# Link previews: the same card as every mojolearn.org page (_templates/layout.html writes the og:/twitter: tags).
html_context["share_image"] = "https://mojolearn.org/social-v3.png"
html_context["share_description"] = "mojolearn documentation: install, user guide and API reference for reproducible GPU machine learning."
sitemap_url_scheme = "{link}"


# Members left out of the docs: broken, unverified or build-dependent (api_exclude.txt lines with a dot,
# e.g. "SmallMLPTrainer.train_steps"). Public names without a dot are skipped by tools/gen_api.py.
def _excluded():
    import pathlib
    lines = (pathlib.Path(__file__).parent / "api_exclude.txt").read_text().splitlines()
    return {l.split("\t", 1)[0].strip() for l in lines if l.strip() and not l.startswith("#")}


_EXCLUDED = _excluded()
_EXCLUDED_MEMBERS = {e for e in _EXCLUDED if "." in e}
_EXCLUDED_NAMES = _EXCLUDED - _EXCLUDED_MEMBERS  # also dropped where a public module lists them


def _skip_member(app, what, name, obj, skip, options):
    qualname = getattr(obj, "__qualname__", "")                     # Class.method
    module = (getattr(obj, "__module__", "") or "").rsplit(".", 1)[-1]  # module.function
    if qualname in _EXCLUDED_MEMBERS or f"{module}.{name}" in _EXCLUDED_MEMBERS or name in _EXCLUDED_NAMES \
            or name == "__init__":
        return True
    return None


def _unlink_prose_types(app, doctree):
    # Docstring types are prose ("float32 array of shape (n,)"). A lowercase word there is not a class:
    # left as a cross-reference, "shape" would link to whichever class has a .shape. Keep links only for
    # capitalized names (our classes) and Python builtins (intersphinx).
    import builtins
    from docutils import nodes
    from sphinx import addnodes
    for ref in list(doctree.findall(addnodes.pending_xref)):
        target = ref.get("reftarget", "")
        if ref.get("refdomain") == "py" and target[:1].islower() and "." not in target \
                and not hasattr(builtins, target):
            ref.replace_self(nodes.Text(ref.astext()))


def setup(app):
    app.connect("source-read", _fill_facts)
    app.connect("autodoc-skip-member", _skip_member)
    app.connect("doctree-read", _unlink_prose_types)

