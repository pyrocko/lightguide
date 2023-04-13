---
hide-toc: true
---
# Welcome to lightguide's documentation

```{caution}
lighguide is under active development, expect changes to the API.
```

Lightguide is a Python framework for handling distributed acoustic sensing (DAS) data.
It offers containers and objects for signal processing and analyzing the data.

## Example

```py
from lightguide import Blast

blast = Blast.from_miniseed("my-data.mseed")

blast.afk_filter(exponent=0.8)
blast.highpass(cutoff_freq=10.0)
```

Lightguide is powered by the [Pyrocko](https://pyrocko.org) project for data I/O and data management.

```{toctree}
:hidden:
:maxdepth: 2

data_import
data_handling
afk_filter
```

```{toctree}
:caption: Development
:hidden:

reference/index
genindex
```
