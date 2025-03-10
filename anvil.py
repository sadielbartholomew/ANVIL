import functools
import logging
import os
import sys

import numpy as np

from itertools import pairwise
from pprint import pformat, pprint
from time import time

import cf
import nvector as nv


# TODO eventually match to precise model value, for now always apply
# cf-python default
EARTH_RADIUS = cf.field._earth_radius


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

def get_env_report():
    """Provide an optional report of environment and diagnostics.

    TODO: DETAILED DOCS
    """
    # Get the cf-python environment
    print("Python and cf-python environment is:")
    cf_env = cf.environment(paths=False)

    # Append the nvector library details, as the other core dependency
    print(f"nvector version is: {nv.__version__}\n")


@timeit
def get_u_field(path=None):
    """TODO."""
    # Use example field for now, later read in from path
    f = cf.example_field(0)
    print("Data using to process grid is:")
    f.dump()
    return f


@timeit
def get_nvector(lats, lons):
    """TODO."""
    return nv.lat_lon2n_E(lats.data, lons.data)


@timeit
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


@timeit
def compare_to_earth_circumference(gc_distance):
    """TODO."""
    earths_circumference = 2 * np.pi * EARTH_RADIUS
    # Round as this is intended as a reference figure so precision not req'd.
    return round(gc_distance / earths_circumference, 3)


@timeit
def get_great_circle_distance(
        n_vector_a, n_vector_b, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=True
):
    """TODO.

    For a lone pair of n-vectors. For sequences use plural function
    'get_great_circle_distances'.
    """
    # Based on example at:
    # https://nvector.readthedocs.io/en/latest/tutorials/
    # getting_started_functional.html#example-5-surface-distance
    gc_distance = nv.great_circle_distance(
        n_vector_a, n_vector_b, radius=earth_radius_in_m)[0]
    print("gc_distance is (units of metres):", gc_distance)

    if not ec_comparison:
        return gc_distance

    return gc_distance, compare_to_earth_circumference(gc_distance)


@timeit
def get_great_circle_distances(
        n_vectors_a, n_vectors_b, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=True
):
    """TODO.

    For sequences of n-vectors e.g. across an axis. For singular pairs use
    non-plural function 'get_great_circle_distance'.
    """
    gc_distances = []
    # TODO: vectorise this for efficiency
    for na, nb in zip(n_vectors_a, n_vectors_b):
        # Based on example at:
        # https://nvector.readthedocs.io/en/latest/tutorials/
        # getting_started_functional.html#example-5-surface-distance
        s_AB = nv.great_circle_distance(
            na, nb, radius=earth_radius_in_m)[0]
        gc_distances.append(s_AB)

    print("gc_distances list is (units of metres):", gc_distances)
    if not ec_comparison:
        return gc_distances

    fractions_of_ec = [
        compare_to_earth_circumference(gc_distance) for
        gc_distance in gc_distances
    ]

    return gc_distances, fractions_of_ec


@timeit
def basic_testing(nvectors):
    """TODO.

    TODO move out to testing module eventually.
    """
    nvector_respective_distances = []
    for nv_a, nv_b in pairwise(nvectors):
        distance = get_great_circle_distance(nv_a, nv_b)
        nvector_respective_distances.append(distance)

    print("Vectors are:")
    pprint(nvector_respective_distances)

    # Check the pairwise distances add up to the same as the distance from
    # the first to the last
    full_distance = get_great_circle_distance(nvectors[0], nvectors[-1])
    print("Full distance is:", full_distance)
    all_dist = np.sum([n[0] for n in nvector_respective_distances])
    print("In comparison to summed distance of: ", all_dist)

    # 6. FURTHER TEST demo: iteration of pairwise points, add to check distance
    # is that expected i.e. adds to a quarter of earth's circumference
    # SKIP FOR NOW as general, use for this three-lat case by adding above
    # TODO

    # 7. Test that the two values add up
    # TODO


# ----------------------------------------------------------------------------
# Main procedure
# ----------------------------------------------------------------------------

@timeit
def main():
    """TODO."""
    # 0. Print divisions
    div = "\n---------------\n"
    new ="\n"

    # 1. Initiate tool
    print(div, "Running ANVIL", div)
    print("Using environment of:")
    get_env_report()

    # 2. Get data to use.
    # Just use an example field for now, we only care about the regular lat-lon
    # grid and its resolution, not the data itself.
    f = get_u_field()

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

    # 5. Basic testing
    basic_testing(nvectors)


if __name__ == "__main__":
    sys.exit(main())
