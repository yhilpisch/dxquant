# Sphinx configuration for the DXQuant documentation build.
from dxquant import __version__

project = "DXQuant"
author = "DXQuant Contributors"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = "alabaster"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
