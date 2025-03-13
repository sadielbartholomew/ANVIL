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
# Grab float value to use instead of cf-python object which may cause issue
# for optimisation strategies.
EARTH_RADIUS = cf.field._earth_radius.array.item()

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

    # Regular lat-lon grid test cases. In order of low -> high resolution.
    # Test case 1: lat x lon of 5 x 8
    ###f = cf.example_field(0)
    # SLB timings 12/03/25: Time taken (in s) for 'main' to run: 1.1899

    # Test case 2: lat x lon of 30 x 48
    ###f = cf.read("test_data/regrid.nc")[1]
    # SLB timings 12/03/25: ???

    # Test case 2: lat x lon of 73 x 96
    ###f = cf.read("test_data/regrid.nc")[0]
    # SLB timings 12/03/25: ???

    # Test case 4: lat x lon of 160 x 320
    f = cf.read("test_data/160by320griddata.nc")[0].squeeze()
    # SLB timings 12/03/25: ???
    # _____ Time taken (in s) for
    # 'perform_nvector_field_iteration' to run: 180.7028 _____
    # _____ Time taken (in s) for
    # 'perform_nvector_field_iteration' to run: 181.1562 _____
    # To calculate one full GCD field, ~180 i.e. ~3 minutes. Would need to
    # calculate 160/2 = 80 => expect it to take 3m * 80 = 240 mins = 4 hours
    # for all of the GC distance fields.
    # Time taken (in s) for 'perform_nvector_field_iteration' to run: 123.7935
    # To calculate one full azimuth angle field, takes ~7-12 seconds =>
    # expect 80 * 10 = 800 seconds = ~14 minutes for all the azimuth fields.

    print("Data using to process grid is:")
    f.dump()
    return f


def get_nvector(lat, lon):
    """TODO."""
    # Note lat_lon2n_E expects radians so convert from degrees if those are
    # the units. We are assuming the units come in degrees_* form and need
    # converting, for now.
    return nv.lat_lon2n_E(nv.rad(lat), nv.rad(lon))


@timeit
def get_nvectors_across_grid(field):
    """TODO."""

    lats = field.coordinate("latitude").data.array
    lons = field.coordinate("longitude").data.array
    lats_len = lats.size
    lons_len = lons.size

    nv_data_size = (3, lats_len, lons_len)
    output_data_array = np.zeros(nv_data_size)  # 3 for 3 comps to an n-vector
    for lat_i, lat in enumerate(lats):
        for lon_i, lon in enumerate(lons):
            # For a quick check
            if lat_i == 2 and lon_i == 2:  # test on arbitrary case
                print(
                    "All LL", lat_i, lat, lon_i, lon, field[lat_i, lon_i])

            # Squeeze to unpack from (3, 1) default shape to (3,) required
            grid_nvector = get_nvector(lat, lon).squeeze()
            print("nvector is", grid_nvector, grid_nvector.shape)
            output_data_array[:, lat_i, lon_i] = grid_nvector

    output_field = field.copy().squeeze()  # squeeze out time
    # Need to create a new domain axis of size three to hold the n-vector
    # components
    # Using https://ncas-cms.github.io/cf-python/tutorial.html#id253 as guide
    nvector_component_axis = output_field.set_construct(cf.DomainAxis(3))
    # Note: no need for new dimension coordinate but still nice to have named
    dc = cf.DimensionCoordinate()
    dc.set_data(cf.Data([0, 1, 2]))
    dc.set_property("long_name", "nvector_components")
    nv_key = output_field.set_construct(dc, axes=nvector_component_axis)

    print("out field", output_field)
    output_field.dump()
    # TODO re-set standard name.
    # Now set data array all at once, to avoid multiple setting operations
    output_field.set_data(
        output_data_array,
        axes=(nvector_component_axis,) + output_field.get_data_axes()
    )
    # TODO Change names and units to be appropriate

    return output_field


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


###@timeit
def get_great_circle_distance(
        n_vector_a, n_vector_b, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=False
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
    ###print("gc_distance is (units of metres):", gc_distance)

    if not ec_comparison:
        return gc_distance

    return gc_distance, compare_to_earth_circumference(gc_distance)


###@timeit
def get_great_circle_distances(
        n_vectors_a, n_vectors_b, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=False
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


###@timeit
def get_azimuth_angle_between(
        n_vector_a, n_vector_b, earth_radius_in_m=EARTH_RADIUS,
        degrees_output=True
):
    """TODO."""
    # See in the n-vector system docs, 'Example 1: A and B to delta' at:
    # https://www.ffi.no/en/research/n-vector/#example_1
    # where we need to find the azimuth using the Python nv library as per:
    # https://nvector.readthedocs.io/en/latest/tutorials/
    # getting_started_functional.html#example-1-a-and-b-to-delta

    # Note: there are some documented physical limitations to this calc,
    # for example the azimuth is undefinde at the poles. TODO watch out for
    # this, work out how to navigate relative azimuths for calculation!

    # Use the same terminology / variable names as in the Python nv library
    # Example, to help to cross-reference and keep consistent. But we do
    # rename our input args as:
    # n_vector_a = n_EA_E
    # n_vector_b = n_EB_E
    # and we assume no height/z i.e. at Earth surface => z_EA = 0 and z_EB = 0
    p_AB_E = nv.n_EA_E_and_n_EB_E2p_AB_E(n_vector_a, n_vector_b)
    R_EN = nv.n_E2R_EN(n_vector_a)
    p_AB_N = np.dot(R_EN.T, p_AB_E).ravel()

    # Finally, we can bring this information together to find the azimuth
    azimuth = np.arctan2(p_AB_N[1], p_AB_N[0])
    if degrees_output:
        azi_deg = nv.deg(azimuth)
        ### print(
        ###    "Found an azimuth (relative to North) in degrees of:", azi_deg)
        return azi_deg
    else:
        ### print("Found an azimuth (relative to North) in radians of:", azimuth)
        return azimuth


@timeit
def perform_nvector_field_iteration(
        r0_i, r0_nvector, result_data_size, lats, lons,
        operation, long_name_start, origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    output_field_for_r0 = grid_nvectors_field_flattened.copy()
    print("Output field metadata will be", output_field_for_r0)

    # Replace the data in the field with the value of the distance to
    # the point on the grid.
    #
    # TODO use Dask or vectorise etc. to make more efficient once working
    # TODO make robust to domain axes other than lat-lon only

    # Re-set all data at once, so create as full numpy array before
    # re-setting
    #output_data_array = np.zeros(nv_data_size)  # 3 for 3 comps to
    # an n-vector
    grid_nvectors_data = grid_nvectors_field.data.array

    # Use a (partially / most as can) vectorised approach for efficiency!
    # Fow now, this has slowed us down! Compared to ~180s per field, have:
    # Time taken (in s) for 'perform_nvector_field_iteration' to run: 247.9112
    # but when we use Dask or numba this should improve greatly

    # Shape: (lat_size, lon_size)
    output_data_array = np.zeros_like(grid_nvectors_data[0])

    # Reshape to (3, lat_size * lon_size)
    grid_nvectors = grid_nvectors_data.reshape(3, -1)

    def compute_distance(grid_nvector_1D):
        """TODO."""
        grid_nvector = grid_nvector_1D[:, np.newaxis]  # Reshape to (3,1)
        return operation(r0_nvector, grid_nvector)

    # Apply function along axis 0 (across all lat/lon grid points)
    # Shape: (lat_size * lon_size,)
    gc_distances = np.apply_along_axis(compute_distance, 0, grid_nvectors)

    # Reshape back to 2D grid
    output_data_array[:] = gc_distances.reshape(output_data_array.shape)

    # TODO re-set standard name.

    # Now set data array all at once, to avoid multiple setting operations
    output_field_for_r0.set_data(output_data_array)

    # Finally, add to FieldList to store outputs
    print("*** Final field result of:", output_field_for_r0)
    pprint(output_field_for_r0.data.array)

    # Label the field by name so we know which lat-lon the origin r_nvector
    # is for
    r0_lat, r0_lon = origin_ll_ref[r0_i]
    output_field_for_r0.long_name = (
        f"{long_name_start}_from_point_at_lat_{r0_lat}_lon_{r0_lon}"
    )
    print("Name is", output_field_for_r0.long_name)

    return output_field_for_r0


def perform_operation_with_nvectors_on_origin_fl(
        operation, long_name_start, origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    print("Origin LL ref is:", origin_ll_ref)

    # Process input grid_nvectors_field lats and lons ready to iterate over
    lats = grid_nvectors_field.coordinate("latitude").data.array
    lons = grid_nvectors_field.coordinate("longitude").data.array
    result_data_size = (lats.size, lons.size)

    output_fieldlist = cf.FieldList()
    # Iterate over all input origin_nvectors
    for r0_i, r0_nvector in enumerate(origin_nvectors):
        output_field_for_r0 = perform_nvector_field_iteration(
            r0_i, r0_nvector, result_data_size, lats, lons,
            operation, long_name_start, origin_nvectors, origin_ll_ref,
            grid_nvectors_field, grid_nvectors_field_flattened,
        )

        output_fieldlist.append(output_field_for_r0)

    return output_fieldlist


def get_gc_distance_fieldlist(
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    return perform_operation_with_nvectors_on_origin_fl(
        get_great_circle_distance, "great_circle_distance",
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
    )


def get_azimuth_angles_fieldlist(
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    return perform_operation_with_nvectors_on_origin_fl(
        get_azimuth_angle_between, "azimuth_angle",
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
    )


# ----------------------------------------------------------------------------
# Basic testing, to be pulled out & consol'd into testing module eventually
# ----------------------------------------------------------------------------

@timeit
def basic_gc_distance_testing(nvectors, ll_ref=None):
    """TODO."""
    nvector_respective_distances = []

    if ll_ref:
        for ll_a, ll_b in pairwise(ll_ref):
            print("LL refs are:", ll_a, ll_b)

    for nv_a, nv_b in pairwise(nvectors):
        print("Have:", nv_a, nv_b)
        distance = get_great_circle_distance(nv_a, nv_b)
        nvector_respective_distances.append(distance)

    print(
        "Distances are from n-vectors (2-tuple, raw in m plus in terms of "
        "Earth's circumference):"
    )
    pprint(nvector_respective_distances)

    # Check the pairwise distances add up to the same as the distance from
    # the first to the last
    full_distance = get_great_circle_distance(
        nvectors[0], nvectors[-1], ec_comparison=True)
    if ll_ref:
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


def basic_azimuth_angle_testing(field):
    """TODO."""
    nvectors, _ = get_nvectors_across_coord(
        field, across_latitude=False)
    # Since the n-vectors are across the lon axis, in order of increasing
    # lon, they should be at 90 degs azimuth relative to any one before,
    # though note complications of cyclicity meaning it might not be safe
    # to compare to say the first and last which may be at -90 instead.
    assert get_azimuth_angle_between(
        nvectors[0], nvectors[1]) == 90.0
    assert get_azimuth_angle_between(nvectors[1], nvectors[2]) == 90.0
    # etc., but no need to test on more for a basic check/validation


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
    origin_nvectors, origin_ll_ref = get_nvectors_across_coord(
        upper_hemi_lats_field)
    print("Lat-lon reference is:", origin_ll_ref)

    # 5. Basic testing for GC distance calculation - input calc's from 4
    ###basic_gc_distance_testing(origin_nvectors, origin_ll_ref)

    # 6. Basic testing for azimuth angle (bearing) calculation
    ###basic_azimuth_angle_testing(upper_hemi_lats_field)

    # 7. Get grid of n-vectors for every lat-lon grid-point. Must use original
    # f field not upper hemi field since the latter was subspaced down but we
    # need to find the full, original lat-lon grid of n-vectors.
    grid_nvectors_field = get_nvectors_across_grid(f)
    print(f"Full grid of n-vectors for field {f!r} is:")
    grid_nvectors_field.dump()

    # 8. Use inputs of (A) the upper hemisphere lats field, from step 3, and
    # (B) the full lat-lon grid of n-vectors, from step 7. For each point in
    # (A) to cover all lats for a given/set lon, find the distances and
    # azimuths to all grid-points as organised in (B) and store these as one
    # field each.
    #
    # So, for example, if there are 10 latitudes in (A) and 20 lat x 40 lon
    # points in (B), we end up with 10 fields of 20*40=800 values each for the
    # GC distances and a further 10 fields of 800 vaues each for the azimuth
    # angles.
    # 9. Get fields with GC distances
    print("Starting FieldList calculations for GC distance.")
    cc_distance_example_fl = get_gc_distance_fieldlist(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f)
    print(
        "\n*** Done FieldList calculations for GC distance. Have total of "
        f"{len(cc_distance_example_fl)} fields in result."
    )
    for field in cc_distance_example_fl:
        # As a basic test, only one point (coresponding to the r0_nvector grid
        # point) should have a 0.0 distance since it will be a coincident point
        f_data = field.data.array
        assert (np.count_nonzero(f_data) + 1 == f_data.size)
        print(
            f"\nOutput gc distance field with name '{field.long_name}' "
            f"has data of {field.data}."
        )

    cf.write(cc_distance_example_fl, "test_outputs/out_gc_distance.nc")

    # 10. Get fields with bearings (azimuth angles)
    #     TODO once have fast enough approach for the GC distance.
    ###get_azimuth_angles_fieldlist(
    ###    upper_hemi_lats_field, grid_nvectors, ll_ref)
    print("Starting FieldList calculations for azimuth angle.")

    cc_distance_example_fl = get_azimuth_angles_fieldlist(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f)
    print(
        "\n*** Done FieldList calculations for azimuth angle. Have total of "
        f"{len(cc_distance_example_fl)} fields in result."
    )
    for f in cc_distance_example_fl:
        print(
            f"\nOutput azimuth angle field with name '{f.long_name}' "
            f"has data of {f.data}."
        )

    cf.write(cc_distance_example_fl, "test_outputs/out_azimuth_angle.nc")


if __name__ == "__main__":
    sys.exit(main())
