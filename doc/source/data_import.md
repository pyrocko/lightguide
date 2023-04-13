# DAS Data Import

## Supported Formats

Lightguide currently supports the following interrogators and data formats.

* MiniSEED
* Silixa TDMS
* Silixa HDF5
* ASN OptoDAS HDF5

```{todo}
As more and more data is coming in we will support other data formats as well.
```

## Loading from MiniSEED

```py
from lightguide import Blast

blast = Blast.from_miniseed("my-data.mseed")
```
