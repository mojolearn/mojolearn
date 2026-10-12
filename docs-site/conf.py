# Sphinx configuration for docs.mojolearn.org. Built against the installed mojolearn
# release (pip install mojolearn==<version>), so the API reference is what users install.
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import mojolearn

project = "mojolearn"
author = "Andrew Hendel"
copyright = "2026, Andrew Hendel"
release = mojolearn.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "tools", "README.md"]
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
    # Version dropdown, like PyTorch's: tools/publish.sh writes /switcher.json listing every published release.
    "switcher": {"json_url": "https://docs.mojolearn.org/switcher.json", "version_match": release},
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "check_switcher": False,
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {"default_mode": "auto"}
html_show_sourcelink = False
# Sitemap for search engines, under the stable/ alias so indexes point at the current release.
html_baseurl = "https://docs.mojolearn.org/stable/"
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
    app.connect("autodoc-skip-member", _skip_member)
    app.connect("doctree-read", _unlink_prose_types)

