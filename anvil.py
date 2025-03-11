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

# Use this to get a string representation of values that is '1.23' rather
# than 'np.float64(1.23)' which is less readable for dev. & debugging
np.set_printoptions(legacy="1.25")


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
    # Note lat_lon2n_E expects radians so convert from degrees if those are
    # the units. We are assuming the units come in degrees_* form and need
    # converting, for now.
    return nv.lat_lon2n_E(nv.rad(lats), nv.rad(lons))


@timeit
def get_nvectors_across_coord(
        field, across_latitude=True, include_lat_lon_ref=True):
    """TODO

    If across_latitude is True the nvectors will be returned for every
    latitude value of the field in order, else it will be returned for every
    longitude value in order.
    """
    lats = field.coordinate("latitude").data.array
    lons = field.coordinate("longitude").data.array

    # Always take the first if the other coord (lat/lon) as the one
    # to find n-vector for
    nvectors = []
    lat_lon_ref = []
    if across_latitude:
        for lat_datum in lats:
            coords = lat_datum, lons[0]
            nvector = get_nvector(*coords)
            nvectors.append(nvector)
            if include_lat_lon_ref:
                lat_lon_ref.append(coords)
    else:
        for lon_datum in lons:
            coords = lats[0], lon_datum
            nvector = get_nvector(*coords)
            nvectors.append(nvector)
            if include_lat_lon_ref:
                lat_lon_ref.append(coords)

    print("n-vector list determined is:")
    pprint(nvectors)

    if include_lat_lon_ref:
        print("lat-lon ref. for n-vector list is:")
        pprint(lat_lon_ref)
        return nvectors, lat_lon_ref
    else:
        return nvectors


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
def basic_testing(nvectors, ll_ref):
    """TODO.

    TODO move out to testing module eventually.
    """
    nvector_respective_distances = []

    for ll_a, ll_b in pairwise(ll_ref):
        print("LL refs are:", ll_a, ll_b)

    for nv_a, nv_b in pairwise(nvectors):
        print("HAVE", nv_a, nv_b)
        distance = get_great_circle_distance(nv_a, nv_b)
        nvector_respective_distances.append(distance)

    print(
        "Distances are from n-vectors (2-tuple, raw in m plus in terms of "
        "Earth's circumference):"
    )
    pprint(nvector_respective_distances)

    # Check the pairwise distances add up to the same as the distance from
    # the first to the last
    full_distance = get_great_circle_distance(nvectors[0], nvectors[-1])
    print("Full distance is:", full_distance, "for", ll_ref[0], ll_ref[-1])

    all_dist = np.sum([n[0] for n in nvector_respective_distances])
    print(
        "In comparison to summed distance along axis of: ",
        all_dist, compare_to_earth_circumference(all_dist)
    )

    # Asserting that the distance from the first to last n-vector is equal to
    # the sum of the distances from N to N+1 along the line, which it should
    # be given that they are all on one longitude or latitude and therefore
    # arcs along the same great circle.
    assert full_distance[0] == all_dist


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

    # FOR NOW: assuming lats and los are in degrees_* units, so need
    # conversion to radians for n-vector libary calcs. TODO add logic to
    # check on exact units and convert or don't as appropriate.

    # 4. Get n-vectors for lat (at any, take first lon value) grid points
    nvectors, ll_ref = get_nvectors_across_coord(upper_hemi_lats_field)
    print("lat and lon reference is:", ll_ref)

    # 5. Basic testing
    basic_testing(nvectors, ll_ref)


if __name__ == "__main__":
    sys.exit(main())
