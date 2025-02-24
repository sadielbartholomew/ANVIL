import functools
import logging
import os
import sys

from pprint import pformat
from time import time

import cf
import nvector as nv


# ----------------------------------------------------------------------------
# Set up timing and logging
# ----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def timeit(func):
    """A decorator to measure and report function execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = time()
        output = func(*args, **kwargs)
        endtime = time()
        totaltime = endtime - starttime

        # Note: using a print not log call here, so they always emerge. At
        # release time we can subsume this into the logging system.
        print(
            f"\n_____ Time taken (in s) for {func.__name__!r} to run: "
            f"{round(totaltime, 4)} _____\n"
        )
        return output

    return wrapper


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------

def get_nvectors(construct):
    """TODO
    """
    pass  # TODO


# ----------------------------------------------------------------------------
# Main procedure
# ----------------------------------------------------------------------------

def get_env_and_diagnostics_report():
    """Provide an optional report of environment and diagnostics.

    TODO: DETAILED DOCS
    """
    # Get the cf-python environment
    cf_env = cf.environment(display=False)

    # Append the nvector library details, as the other core dependency
    print(cf_env, type(cf_env))

    logger.info(
        "Using Python and CF environment of:\n"
        f"{cf.environment(display=False)}\n"
    )


@timeit
def main():
    """TODOO."""
    # 0. Print divisions
    div = "\n---------------\n"
    new ="\n"

    # 1. Initiate tool

    print(div, "Running ANVIL", div)
    print("Using environment of:")
    get_env_and_diagnostics_report()

    # 2. Get data to use.
    # Just use an example field for now, we only care about the regular lat-lon
    # grid and its resolution, not the data itself.
    fl = cf.example_fields()
    f = fl[0]
    print(new, "Data is:")
    f.dump()

    # 3. Get all latitudes in the upper hemisphere - we can reflect about the
    # hemisphere to get all the lower hemisphere equivalents and then use
    # rotational symmetry of the Earth to get the information for each
    # longitude for a given latitude.
    print(new, "Processing latitudes")
    lats_key, lats = f.coordinate("latitude", item=True)
    print("IS", lats_key, lats)

    # Subspacing to the upper hemisphere only
    kwargs = {lats_key: cf.ge(0)}  # greater than or equal to 0, i.e. equator
    upper_hemi_lats = f.subspace(**kwargs)
    print(new, "Upper hemisphere latitudes are:", upper_hemi_lats)

    # 4. Get n-vectors for lat (at any, take first lon value) grid points
    nvectors = get_nvectors(construct)



if __name__ == "__main__":
    sys.exit(main())
