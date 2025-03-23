"""Profile"""

import pstats
import cProfile
from typing import Callable


def profile(
        function: Callable,
        *args,
        profile_filename: str = '',
        **kwargs,
) -> None:
    """Profile a function with cProfile

    If a path is provided, the profile data is saved and can be opend with:

    :code:`pyprof2calltree -k -i path/to/file.profile`

    """
    n_time = kwargs.pop('pstat_n_time', 30)
    n_cumtime = kwargs.pop('pstat_n_cumtime', 30)
    prof = cProfile.Profile()
    result = prof.runcall(function, *args, **kwargs)
    if profile_filename:
        prof.dump_stats(profile_filename)
        pstat = pstats.Stats(profile_filename)
    else:
        pstat = pstats.Stats(prof)
    pstat.sort_stats('time').print_stats(n_time)
    pstat.sort_stats('cumtime').print_stats(n_cumtime)
    return result
