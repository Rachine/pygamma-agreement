Welcome to pygamma-agreement's documentation!
=============================================

`pygamma-agreement` is a Python library used to measure the γ inter-annotator
agreement, defined by Mathet et Al. in
`"The Unified and Holistic Method Gamma for Inter-Annotator Agreement Measure and Alignment" <https://www.mitpressjournals.org/toc/coli/41/3>`_ .
It features an easy-to-use API and a a command line interface (CLI) for those
that prefer using regular shell scripts for their data processing tasks.


Installation
------------

The package is available on pip. Just run

.. code-block:: bash

   $ pip install pygamma-agreement

Pygamma-agreement uses the `GNU Linear Programming Kit <https://www.gnu.org/software/glpk/>`_ as its default Mixed Integer
Programming solver (critical step of the gamma-agreement algorithm). Since it is quite slow, you can install the
`CBC <https://projects.coin-or.org/Cbc>`_ solver and its official `python API <https://mpy.github.io/CyLPdoc/>`_.
To use those in `pygamma-agreement`, simply install them:

- **Ubuntu/Debian** :  ``$ sudo apt install coinor-libcbc-dev``
- **Fedora** : ``$ sudo yum install coin-or-Cbc-devel``
- **Arch Linux** : ``$ sudo pacman -S coin-or-cbc``
- **Mac OS X** :
    - ``$ brew tap coin-or-tools/coinor``
    - ``$ brew install coin-or-tools/coinor/cbc pkg-config``

then:

.. code-block:: bash

    $ pip install pygamma-agreement[CBC]

.. warning::

    A bug in GLPK causes the standart ouput to be polluted by non-deactivable messages.
    If you expect to use standart output for informative or parsing purposes, it is strongly
    advised to use the CBC solver.

.. toctree::
   :maxdepth: 2
   :caption: Content:

   quickstart
   principles
   cli
   how-to-s
   issues
   performance
   api_reference
   changelog
