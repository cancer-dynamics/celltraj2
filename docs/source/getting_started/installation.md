# Installation

Clone the repository:

```bash
git clone https://github.com/cancer-dynamics/celltraj2.git
cd celltraj2
```

For analysis workflows that need H5, Zarr, TIFF, and ND2 support, install the
package in editable mode with the analysis and ND2 extras:

```bash
python -m pip install -e ".[analysis,nd2,dev]"
```

For lean environments that only need the typed metadata contracts and path
helpers:

```bash
python -m pip install -e .
```

The optional dependencies are imported only when the corresponding readers or
stores are used. That keeps metadata-only tests and lightweight notebooks from
requiring the full microscopy stack.

## Documentation Dependencies

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
bash docs/make_docs.sh
```

The generated HTML is written to `docs/build/html/`.
