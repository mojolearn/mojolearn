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
html_favicon = "_static/mojolearn-mark-on-white.svg"
html_theme_options = {
    "logo": {
        "text": "mojolearn",
        "image_light": "_static/mojolearn-mark.svg",
        "image_dark": "_static/mojolearn-mark-on-white.svg",
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
