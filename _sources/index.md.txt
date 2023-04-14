---
hide-toc: true
---
# Welcome to lightguide's documentation

```{caution}
lighguide is under active development, expect changes to the API.
```

Lightguide is a package for handling, filtering and modelling distributed acoustic sensing (DAS) data. The package interfaces handling and processing routines of DAS data to the [Pyrocko framework](https://pyrocko.org).

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

examples/1-import-data
examples/2-data-processing
examples/3-event-analysis
```

```{toctree}
:caption: Methods
:hidden:

afk_filter
```

```{toctree}
:caption: Development
:hidden:

reference/index
genindex
```
