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


def get_nvector(lats, lons):
    """TODO."""
    return nv.lat_lon2n_E(lats.data, lons.data)


def get_nvectors_across_coord(field, across_latitude=True):
    """TODO

    If across_latitude is True the nvectors will be returned for every
    latitude value of the field in order, else it will be returned for every
    longitude value in order.
    """
    lats = field.coordinate("latitude")
    lons = field.coordinate("longitude")

    # Always take the first if the other coord (lat/lon) as the one
    # to find n-vector for
    nvectors = []
    if across_latitude:
        for datum in lats:
            nvector = get_nvector(datum, lons[0])
            nvectors.append(nvector)
    else:
        for datum in lons:
            nvector = get_nvector(lats[0], lons)
            nvectors.append(nvector)

    print("n-vector list determined is:", nvectors)
    return nvectors


# ----------------------------------------------------------------------------
# Main procedure
# ----------------------------------------------------------------------------

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
    f = cf.example_field(0)
    print("Data is:")
    f.dump()

    # 3. Get all latitudes in the upper hemisphere - we can reflect about the
    # hemisphere to get all the lower hemisphere equivalents and then use
    # rotational symmetry of the Earth to get the information for each
    # longitude for a given latitude.
    print("Processing latitudes")
    lats_key, lats = f.coordinate("latitude", item=True)

    # Subspacing to the upper hemisphere only
    kwargs = {lats_key: cf.ge(0)}  # greater than or equal to 0, i.e. equator
    upper_hemi_lats_field = f.subspace(**kwargs)
    print("Upper hemisphere latitudes are:", upper_hemi_lats_field)

    # 4. Get n-vectors for lat (at any, take first lon value) grid points
    nvectors = get_nvectors_across_coord(upper_hemi_lats_field)



if __name__ == "__main__":
    sys.exit(main())
