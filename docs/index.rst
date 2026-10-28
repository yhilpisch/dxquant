DXQuant Documentation
=====================

DXQuant is an alpha quantitative finance library inspired by DX Analytics.
Its current focus is derivatives, with a limited rates/curves subset.
The distribution and Python package are ``dxquant``. Examples use
``import dxquant as dx``. Install from this checkout until publication.
Read :doc:`numerical_contracts` for numerical domains and limitations.
The tutorial index documents headless and Google Colab validation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   numerical_contracts
   api

Compatibility surface
---------------------

The compatibility policy covers the 61 names re-exported through
``dxquant.__all__``. The API reference also includes module-level helpers outside
that list; their inclusion is not an additional compatibility promise.

Scope and migration
-------------------

The library is a partial rewrite, not a drop-in replacement for ``dx``. The
repository contains 13 tutorials in ``notebooks/`` and ``MIGRATION.md``.
The local legacy reference is not part of ordinary source checkouts. Optional
legacy comparisons skip when that reference is unavailable.

Local numerical findings are addressed within the documented alpha scope.
Read :doc:`numerical_contracts` for model assumptions, supported domains,
sampling errors, scenario/currency semantics, and intentional exclusions.
