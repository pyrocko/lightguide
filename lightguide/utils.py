import time
from functools import wraps
from typing import Any, Callable

from pyrocko.trace import Trace

import numpy as np
import numpy.typing as npt


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def traces_to_numpy_and_meta(traces: list[Trace]) -> tuple[npt.NDArray, AttrDict]:
    ntraces = len(traces)
    nsamples = set(tr.ydata.size for tr in traces)

    if len(nsamples) != 1:
        raise ValueError("Traces nsamples differ")
    nsamples = nsamples.pop()

    data = np.zeros((ntraces, nsamples))
    meta = {}
    for itr, tr in enumerate(traces):
        data[itr, :] = tr.ydata
        meta = tr.meta

    return data, AttrDict(meta)


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t = time.time()
        ret = func(*args, **kwargs)
        print(func.__qualname__, time.time() - t)
        return ret

    return wrapper
