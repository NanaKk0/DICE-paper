# Figure-rendering check environment

The complete 22-object `figures` action was run locally with Python 3.11.15,
Matplotlib 3.11.1, NumPy 2.4.6, pandas 2.3.3 and CairoSVG 2.9.1. Arial was
installed separately by the author. CairoSVG also requires the platform Cairo
library. These versions describe the locally tested rendering environment.

`requirements-tested.txt` separately records the Python 3.10 numerical-check
environment. Install `requirements.txt` and provide Arial before using `figures`.
