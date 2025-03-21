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
def get_u_and_v_fields(path=None):
    """TODO."""
    # latitude(160), longitude(320) i.e. 160x320
    vector_fields = cf.read("test_data/ggap.nc")
    u_component_field = cf.read("test_data/ggap.nc")[1]
    v_component_field = cf.read("test_data/ggap.nc")[2]
    u_p0_only = u_component_field[0, 0, :, :].squeeze()
    v_p0_only = v_component_field[0, 0, :, :].squeeze()
    print("u and v fields are:", u_p0_only, v_p0_only)

    # Regrid since this is too coarse for now!
    # latitude(5), longitude(8) i.e. 5x8
    grid_field = cf.example_field(0)
    u = u_p0_only  ###.regrids(grid_field, method="linear")
    v = v_p0_only  ###.regrids(grid_field, method="linear")
    print("u and v regridded fields are:", u, v)
    # Regular lat-lon grid test cases. In order of low -> high resolution.

    # Use example field for now, later read in from path
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
    ###f = cf.read("test_data/160by320griddata.nc")[0].squeeze()
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

    print("Data using to process grid is:\n", u)
    return u, v


def clear_selected_properties(field):
    """TODO."""
    remove_props = ["valid_max", "valid_min", "standard_name", "long_name"]
    for prop in remove_props:
        if field.has_property(prop):
            field.del_property(prop)


def set_reference_latlon_properties(field, ref_lat, ref_lon):
    """TODO."""
    ref_props = {
        "reference_origin_latitude": ref_lat,
        "reference_origin_longitude": ref_lon,
    }
    for prop_key, prop_value in ref_props.items():
        field.set_property(prop_key, prop_value)


def get_reference_latlon_properties(field):
    """TODO."""
    return (
        field.get_property("reference_origin_latitude"),
        field.get_property("reference_origin_longitude"),
    )


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
    # TODO consolidate array operations to remove need for 'for' loop
    for lat_i, lat in enumerate(lats):
        for lon_i, lon in enumerate(lons):
            # Squeeze to unpack from (3, 1) default shape to (3,) required
            grid_nvector = get_nvector(lat, lon).squeeze()
            ###print("nvector is", grid_nvector, grid_nvector.shape)
            output_data_array[:, lat_i, lon_i] = grid_nvector

    output_field = field.copy().squeeze()  # squeeze out time
    # Need to remove any restrictions on data e.g. valid min and valid max
    # which could prevent the GC distance and angles data being written without
    # masking out
    clear_selected_properties(output_field)

    # Need to create a new domain axis of size three to hold the n-vector
    # components
    # Using https://ncas-cms.github.io/cf-python/tutorial.html#id253 as guide
    nvector_component_axis = output_field.set_construct(cf.DomainAxis(3))
    # Note: no need for new dimension coordinate but still nice to have named
    dc = cf.DimensionCoordinate()
    dc.set_data(cf.Data([0, 1, 2]))
    dc.set_property("long_name", "nvector_components")
    nv_key = output_field.set_construct(dc, axes=nvector_component_axis)

    print("*** Out field is:\n", output_field)
    ###output_field.dump()
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

    print("*** n-vector list determined is:")
    pprint(nvectors)

    if include_lat_lon_ref:
        print("*** lat-lon ref. for n-vector list is:")
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


def get_chord_distance(
        n_vector_a, n_vector_b, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=False
):
    """TODO."""
    distance = nv.euclidean_distance(
        n_vector_a, n_vector_b, radius=earth_radius_in_m)[0]
    ###print("gc_distance is (units of metres):", gc_distance)

    if not ec_comparison:
        return distance

    return distance, compare_to_earth_circumference(distance)


def get_euclidean_2d_distance(
        latlon1, latlon2, earth_radius_in_m=EARTH_RADIUS,
        ec_comparison=False
):
    """TODO."""
    # TODOMAKE 0 AND 1 NOT 1, 2
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    # Need these in radians for the formula
    # TODO AS FIELD USE, convert convert_degrees_to_radians(azimuth_angles_fl)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    print("RAD LATS FOR PLANAR EXAMPLE:", lat1, lat2)

    # Basic Pythagorean theorem
    # Conversion of flt plane / Mercator projection formula
    # of sqrt(x**2 + y**2) to lat/lon points is approximatley:
    ref_lat = (lat1 + lat2) / 2
    lat_diff = lat2 - lat1  # no wrapping/cyclicty concerns here - I think?
    lon_diff = np.remainder(lon2 - lon1, 2 * np.pi)
    lon_diff_with_wrapping = np.where(
        lon_diff > np.pi, lon_diff - 2 * np.pi, lon_diff)
    planar_2d_distance = np.sqrt(
        lat_diff**2 + (lon_diff_with_wrapping * np.cos(ref_lat))**2
    ) * earth_radius_in_m
    ###print("gc_distance is (units of metres):", gc_distance)

    if not ec_comparison:
        return planar_2d_distance

    return planar_2d_distance, compare_to_earth_circumference(
        planar_2d_distance)


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

    print("*** gc_distances list is (units of metres):", gc_distances)
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
    p_AB_N = np.dot(R_EN.T, p_AB_E)

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


def perform_nvector_field_iteration(
        r0_i, r0_nvector, result_data_size, lats, lons,
        operation, long_name_start, units_string, origin_nvectors,
        origin_ll_ref, grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    output_field_for_r0 = grid_nvectors_field_flattened.copy()
    clear_selected_properties(output_field_for_r0)

    print("*** Output field metadata will be", output_field_for_r0)

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
    # is for. Remove any original standard name first.
    r0_lat, r0_lon = origin_ll_ref[r0_i]
    output_field_for_r0.long_name = (
        f"{long_name_start}_from_point_at_lat_"
        f"{round(r0_lat, 2)}_lon_{round(r0_lon, 2)}"
    )
    print("*** Name is:", output_field_for_r0.long_name)
    output_field_for_r0.override_units(units_string, inplace=True)

    set_reference_latlon_properties(output_field_for_r0, r0_lat, r0_lon)
    rlat, rlon = get_reference_latlon_properties(output_field_for_r0)
    print("Registered reference origin lat and lon of:", rlat, rlon)

    return output_field_for_r0


def perform_operation_with_nvectors_on_origin_fl(
        operation, long_name_start, units, origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    print("*** Origin LL ref is:", origin_ll_ref)

    # Process input grid_nvectors_field lats and lons ready to iterate over
    lats = grid_nvectors_field.coordinate("latitude").data.array
    lons = grid_nvectors_field.coordinate("longitude").data.array
    result_data_size = (lats.size, lons.size)

    output_fieldlist = cf.FieldList()
    # Iterate over all input origin_nvectors
    for r0_i, r0_nvector in enumerate(origin_nvectors):
        output_field_for_r0 = perform_nvector_field_iteration(
            r0_i, r0_nvector, result_data_size, lats, lons,
            operation, long_name_start, units, origin_nvectors, origin_ll_ref,
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
        get_great_circle_distance, "great_circle_distance", "m",
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
    )


def get_chord_distance_fieldlist(
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO.

    Note: not core to library, just for useful comparison to GC distance.
    """
    return perform_operation_with_nvectors_on_origin_fl(
        get_chord_distance, "chord_distance", "m",
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
    )


def get_planar_distance_fieldlist(upper_hemi_lats_field, origin_field):
    """TODO.

    Note: not core to library, just for useful comparison to GC distance.
    """
    output_fieldlist = cf.FieldList()
    lats = origin_field.coordinate("latitude").data.array
    lons = origin_field.coordinate("longitude").data.array

    lats_o = upper_hemi_lats_field.coordinate("latitude").data.array
    lons_o = upper_hemi_lats_field.coordinate("longitude").data.array
    # For each field in upper hemi lats
    for lat_oi, lat_o in enumerate(lats_o):
        print("lat_o is", lat_o)
        # Take first lon only!
        lon_o = lons_o[0]
        print("lon_o is", lon_o)
        # TODO apply simple vectorisation array op instead
        # once not under the cosh!
        distance_data = np.zeros(origin_field.shape)
        for lat_i, lat in enumerate(lats):
            for lon_i, lon in enumerate(lons):
                euc_2d_data = get_euclidean_2d_distance(
                    (lat, lon), (lat_o, lon_o))
                distance_data[lat_i, lon_i] = euc_2d_data

        dist_field = origin_field.copy()
        dist_field.set_data(distance_data)
        clear_selected_properties(dist_field)

        dist_field.long_name = (
            f"euclidean_2d_distance_from_point_at_lat_"
            f"{np.round(lat_o, 2)}_lon_{np.round(lon_o, 2)}"
        )
        dist_field.override_units("m", inplace=True)

        output_fieldlist.append(dist_field)

    print("OUTPUT FIELDLIST is", output_fieldlist)
    return output_fieldlist


def get_azimuth_angles_fieldlist(
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
):
    """TODO."""
    return perform_operation_with_nvectors_on_origin_fl(
        get_azimuth_angle_between, "azimuth_angle", "degree",
        origin_nvectors, origin_ll_ref,
        grid_nvectors_field, grid_nvectors_field_flattened,
    )


def validate_gc_distance_fl(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f
):
    """TODO."""
    print("Starting FieldList calculations for GC distance.")
    gc_distance_fl = get_gc_distance_fieldlist(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f)
    print(
        "\n*** Done FieldList calculations for GC distance. Have total of "
        f"{len(gc_distance_fl)} fields in result."
    )
    for field in gc_distance_fl:
        # As a basic test, only one point (coresponding to the r0_nvector grid
        # point) should have a 0.0 distance since it will be a coincident point
        f_data = field.data.array
        assert (np.count_nonzero(f_data) + 1 == f_data.size)
        print(
            f"\nOutput gc distance field with name '{field.long_name}' "
            f"has data of {field.data}."
        )

    return gc_distance_fl


def validate_azimuth_angles_fl(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f):
    """TODO."""
    print("Starting FieldList calculations for azimuth angle.")

    azimuth_angles_fl = get_azimuth_angles_fieldlist(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f)
    print(
        "\n*** Done FieldList calculations for azimuth angle. Have total of "
        f"{len(azimuth_angles_fl)} fields in result."
    )
    for f in azimuth_angles_fl:
        print(
            f"\nOutput azimuth angle field with name '{f.long_name}' "
            f"has data of {f.data}."
        )
    return azimuth_angles_fl


def compare_euclidean_2d_distance_fl(field_over, f):
    """TODO.

    Note: not core to library, just for useful comparison to GC distance.
    """
    print("Starting FieldList calculations for PLANAR distance.")
    planar_distance_fl = get_planar_distance_fieldlist(field_over, f)
    print(
        "\n*** Done FieldList calculations for PLANAR distance. Have total of "
        f"{len(planar_distance_fl)} fields in result."
    )
    for field in planar_distance_fl:
        # As a basic test, only one point (coresponding to the r0_nvector grid
        # point) should have a 0.0 distance since it will be a coincident point
        f_data = field.data.array
        ###assert (np.count_nonzero(f_data) + 1 == f_data.size)
        print(
            f"\nOutput PLANAR distance field with name '{field.long_name}' "
            f"has data of {field.data}."
        )

    return planar_distance_fl


def compare_chord_distance_fl(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f
):
    """TODO.

    Note: not core to library, just for useful comparison to GC distance.
    """
    print("Starting FieldList calculations for CHORD distance.")
    chord_distance_fl = get_chord_distance_fieldlist(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, f)
    print(
        "\n*** Done FieldList calculations for CHORD distance. Have total of "
        f"{len(chord_distance_fl)} fields in result."
    )
    for field in chord_distance_fl:
        # As a basic test, only one point (coresponding to the r0_nvector grid
        # point) should have a 0.0 distance since it will be a coincident point
        f_data = field.data.array
        assert (np.count_nonzero(f_data) + 1 == f_data.size)
        print(
            f"\nOutput CHORD distance field with name '{field.long_name}' "
            f"has data of {field.data}."
        )

    return chord_distance_fl


def map_fields_to_reference_latlon_points(fieldlist, lats_values):
    """TODO."""
    mapping = {}
    lats_with_no_field_set = []
    for lat in lats_values:
        print("lat is", lat)
        f = fieldlist.select_by_property(reference_origin_latitude=lat)
        if not f:
            lats_with_no_field_set.append(str(lat))
        # String-value converted float is safer for float precision issues
        # TODO is this a safe/good way?
        mapping[str(lat)] = f

        # Convert to array for efficiency now know size (TODO strcitly already
        # know size to expect so refactor to account for this.
    ###lats_with_no_field_set = np.array(lats_with_no_field_set)

    print("Mapping of latitudes to fields is (with first mapping):")
    pprint(mapping)
    return mapping, lats_with_no_field_set


def apply_reg_latlon_grid_reflective_symmetry(
        lats_to_fields_mapping, empty_lats):
    """TODO."""
    # Note these should be the lower hemisphere values!
    print("Latitude values to acquire fields for are:", empty_lats)

    for lat in empty_lats:
        ###print("Pos value is", abs(lat))
        # String-value converted float is safer for float precision issues
        # TODO is this a safe/good way?
        neg_lat_key = lat.lstrip("-")
        positive_equivalent_field = lats_to_fields_mapping[neg_lat_key][0]

        # Perform reflection on the positive equivalnt i.e. upper hemisphere
        # field - and rename and label properties appropriately
        flipped_field = positive_equivalent_field.copy()
        flipped_field.data.flip(0, inplace=True)
        # TODO also update field name
        flipped_field.set_property("reference_origin_latitude", lat)

        print("Flipped field data is", flipped_field.data.array)
        # String-value converted float is safer for float precision issues
        # TODO is this a safe/good way?
        lats_to_fields_mapping[str(lat)] = cf.FieldList(flipped_field)

    print("Mapping of latitudes to fields is (with post-reflection mapping):")
    pprint(lats_to_fields_mapping)
    return lats_to_fields_mapping


def apply_reg_latlon_grid_rotational_symmetry(original_field, lons):
    """TODO."""
    # TODO
    lons_to_fields_mapping = {}
    # 1. Get the field for the corresponding latitude
    lat, original_lon = get_reference_latlon_properties(original_field)
    print("Longitudes are", lons)
    for lon in lons:
    # TODO float precision care, use cf.isclose?
        if lon == original_lon:
            lons_to_fields_mapping[str(lon)] = original_field
        else:
            # Need to roll to apply rotational symmetry of regular lat-lon grid
            # to get the approproate field from the one at the same latitude
            # but different longitude. Apply roll and edit metadata accordingly.
            rolled_field = original_field.copy()

            # TODO this is slow and we can assume contiguous therefore
            # increasing longitudes - so consolidate
            # Find by how many positions we need to shift by!
            lat_difference_in_steps_0 = original_field.indices(
                longitude=original_lon)[1].item()
            lat_difference_in_steps_1 = original_field.indices(
                longitude=lon)[1].item()
            lat_difference_in_steps = (
                lat_difference_in_steps_1 - lat_difference_in_steps_0
            )
            print(
                "rotation index difference values are:",
                lat_difference_in_steps
            )

            rolled_field.data.roll(
                shift=lat_difference_in_steps, axis=1, inplace=True)
            # TODO also update field name
            rolled_field.set_property("reference_origin_longitude", lon)
            lons_to_fields_mapping[str(lon)] = rolled_field

    print("Mapping of longitudes to fields is (with post-rotation mapping):")
    pprint(lons_to_fields_mapping)
    return lons_to_fields_mapping


def mask_outside_annulus(
        gc_distance_field, distance_lower_bound, distance_upper_bound):
    """TODO."""
    wo_query = cf.wo(distance_lower_bound, distance_upper_bound)
    masked_field = gc_distance_field.where(wo_query, cf.masked)

    return masked_field


def convert_degrees_to_radians(azimuth_angles_fl):
    """TODO."""
    ###print("Value before conversion:", azimuth_angles_fl[0].data[0, -1])
    # Intergration goes from 0 to 2*pi so requires radians
    for field in azimuth_angles_fl:
        field.units = "radian"
    ###print("Value after conversion:", azimuth_angles_fl[0].data[0, -1])


@timeit
def annulus_calculation(
        u_data, v_data, u0, v0,
        annulus_lower, annulus_upper,
        gc_final_latlon_field, aa_final_latlon_field,
):
    """TODO."""
    # Get points within the r + dr annulus limits

    # --- #
    # The below is more clean but doesn't seem to work for some reason!
    # Everything gets masked out!
    # Apply masking to *u* field based on *gc distance* field
    # condition.
    # u_masked_for_annulus = u.where(
    #     ((gc_final_latlon_field < annulus_upper)) |
    #     ((gc_final_latlon_field >= annulus_lower)),
    #     cf.masked
    # )
    # --- #

    gc_final_latlon_data = gc_final_latlon_field.data  # .array needed?
    aa_final_latlon_data = aa_final_latlon_field.data
    # GET RESULT FOR THIS STEP
    # TODO do we use open_lower and/or _upper for open intervals?
    masked_gcd_field = mask_outside_annulus(
        gc_final_latlon_field, annulus_lower, annulus_upper
    )
    gcd_mask = masked_gcd_field.data.mask
    print(f"mask is {gcd_mask}")

    # Masking: apply mask from gc distances masking to the u-field
    # Ignore below for now, is cf-field approach
    ###u_masked_for_annulus = u_data.copy()
    ###u_new_mask_data = np.ma.array(u_data, mask=gcd_mask)
    ###u_masked_for_annulus.set_data(new_mask_data)
    ###v_masked_for_annulus = v_data.copy()
    ###v_new_mask_data = np.ma.array(v_data, mask=gcd_mask)
    ###v_masked_for_annulus.set_data(v_new_mask_data)
    u_masked_for_annulus = np.ma.array(u_data, mask=gcd_mask)
    v_masked_for_annulus = np.ma.array(v_data, mask=gcd_mask)

    ###print("MASKED RESULT", u_masked_for_annulus)
    nm_count = u_masked_for_annulus.count()
    print(
        "Non-masked points count is:", nm_count
    )  # would be the same for u field

    # du, velocity increment du(r) = du(r, x) = du(x + r) − u(x)
    u1 = u_masked_for_annulus
    v1 = v_masked_for_annulus
    u_velocity_increment = u1 - u0
    v_velocity_increment = v1 - v0
    angles = aa_final_latlon_data

    # Find the integrand using vector calculations and the formula
    # Formula for unit vector r-hat from polar coordinates is:
    # r = (cos(theta), sin(theta))
    r_unit_vector = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    vector_velocity_increment = np.stack(
        (u_velocity_increment, v_velocity_increment), axis=-1)
    # Vecdot creates correct shape accounting for stacking arrays for vectors
    # in arrays, as above
    dot_product_result = np.vecdot(r_unit_vector, vector_velocity_increment)
    uv_vector_norm = u_velocity_increment**2 + v_velocity_increment**2
    print(
        "SHAPES ARE", r_unit_vector.shape, vector_velocity_increment.shape,
        dot_product_result.shape, uv_vector_norm.shape
    )
    integrand = dot_product_result * uv_vector_norm
    print("*** Integrand result is", integrand)
    # (Note is scalar as is the result of a dot product)

    # Finally, we can perform the actual integral!
    if nm_count == 0:
        print("Warning: empty annulus!")
    elif nm_count < 4:
        # TODO how to account for these cases?
        print(
            # By the pigeonhole principle!
            "Warning: certainly less than one gridpoint "
            "per quadrant. May not be reliable."
        )

    # TODO now apply angles to get dphi then have pre-integral result ready
    # TODO angles

    result_value = integrand  # including angles part too
    print("Result value is", result_value, result_value.shape)

    return result_value, u1, v1


@timeit
def perform_integration(
        u, v, lats, lons, gc_lats_to_fields_mapping, aa_lats_to_fields_mapping
):
    """TODO."""
    # Initialise result field and data array to set on it
    result_field = u.copy()
    clear_selected_properties(result_field)
    result_array = np.zeros(u.shape)
    # Expand result_array shape to hold full shape within each point, as
    # required to store pre-integral result

    # Set limits
    earth_circ = 2 * np.pi * EARTH_RADIUS
    min_r = 0
    max_r = earth_circ * 0.75  # don't get too close to antipode, for now
    # This represents the annulus width
    # TODO make similar size as a mid-lat-like grid cell, for sensible calcs,
    # i.e. determine this value from the inputs.
    # For now a rough estimate is earht's circumference divided by lat points
    # mult by 2
    dr = earth_circ / (lats.size * 2)  ###* 10 for quicker result to check
    print(f"dr intergral discretised increment value in metres is:", dr)

    # Get limits for annuli to use
    lons_data = lons.data.array
    lats_data = lats.data.array
    annuli_limits = np.arange(min_r, max_r, dr)
    for lat_i, lat in enumerate(lats_data):
        print("START calculating iterations for lat", lat)
        # 0 index to unpack from FieldList. TODO always store field only
        # not singular FieldList.
        gc_lat_origin_field = gc_lats_to_fields_mapping[str(lat)][0]
        aa_lat_origin_field = aa_lats_to_fields_mapping[str(lat)][0]
        gc_across_lons = apply_reg_latlon_grid_rotational_symmetry(
            gc_lat_origin_field, lons_data
        )
        aa_across_lons = apply_reg_latlon_grid_rotational_symmetry(
            aa_lat_origin_field, lons_data
        )

        u_data = u.data.array
        v_data = u.data.array
        for lon_i, lon in enumerate(lons_data):
            print("START calculating iterations for lon", lon)
            gc_final_latlon_field = gc_across_lons[str(lon)]
            aa_final_latlon_field = aa_across_lons[str(lon)]
            #print(
            #    "Starting discretised intergation loop with increment "
            #    f"{dr} and total steps of {len(annuli_limits) - 1}.\n"
            #)

            ###print("START annuli calclulations for", lon, lat)
            u0 = u[lat_i, lon_i]  # exact value at gridpoint
            v0 = v[lat_i, lon_i]  # exact value at gridpoint
            for annulus_lower, annulus_upper in pairwise(annuli_limits):
                # print(
                #     "Annuli limits are:", annulus_lower, annulus_upper
                # )
                # Do this all in data array space, for now - may be
                # worthwhile later to go back to cf-field space
                result_value, u1, v1 = annulus_calculation(
                    u_data, v_data, u0, v0,
                    annulus_lower, annulus_upper,
                    gc_final_latlon_field,
                    aa_final_latlon_field,
                )
                result_array[lat_i, lon_i] = result_value
                # Prepare values as origin values for u1 - u0 in next iteration
                u0 = u1
                v0 = v1

    print("Finished discretised intergation loop.")

    # Try aggregating down the results to one field!
    result_field.set_data(result_array)
    result_field.set_property(
        "long_name", "dynamical_inter_scale_energy_transfer_pre_integral")

    # TODO set data onto result field and update metadata accordingly
    return result_field

# ----------------------------------------------------------------------------
# Basic testing, to be pulled out & consol'd into testing module eventually
# ----------------------------------------------------------------------------

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


def reg_latlon_reflection_testing(
        gc_lats_to_fields_mapping, aa_lats_to_fields_mapping, test_lat_val):
    """TODO."""
    to_plot_reflection_test = cf.FieldList(
        [
            gc_lats_to_fields_mapping[test_lat_val],
            gc_lats_to_fields_mapping[f"-{test_lat_val}"],
            aa_lats_to_fields_mapping[test_lat_val],
            aa_lats_to_fields_mapping[f"-{test_lat_val}"],
        ]
    )
    cf.write(to_plot_reflection_test, "all_reflection_test_01.nc")
    # NOTE, for now testing happens in 'test_symmetry_processing.py'
    # script which reads in file written out above, for separation
    # of concerns.


def reg_latlon_rotation_testing(
        gc_lats_to_fields_mapping, lons, example_lat,
        four_test_lon_values_to_plot
):
    """TODO."""
    example_lat_field = gc_lats_to_fields_mapping[example_lat][0]
    example_across_lons = apply_reg_latlon_grid_rotational_symmetry(
        example_lat_field, lons.data.array
    )
    ex_f1 = example_across_lons[four_test_lon_values_to_plot[0]]
    ex_f2 = example_across_lons[four_test_lon_values_to_plot[1]]
    ex_f3 = example_across_lons[four_test_lon_values_to_plot[2]]
    ex_f4 = example_across_lons[four_test_lon_values_to_plot[3]]
    cf.write(cf.FieldList(
        [ex_f1, ex_f2, ex_f3, ex_f4]), "gc_rotation_test_01.nc")
    # NOTE, for now testing happens in 'test_symmetry_processing.py'
    # script which reads in file written out above, for separation
    # of concerns.


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
    # u is the eastward vector component, b is the northward vector component
    u, v = get_u_and_v_fields()
    # Assume u and v have the same dimensional structure and grid, which
    # is only natural if u and v are two components of a consistent vector
    # field, hence only consult on u assuming it is equivalent for v

    # 3. Get all latitudes in the upper hemisphere - we can reflect about the
    # hemisphere to get all the lower hemisphere equivalents and then use
    # rotational symmetry of the Earth to get the information for each
    # longitude for a given latitude.
    print("Processing latitudes (and longitudes for later)")
    lats_key, lats = u.coordinate("latitude", item=True)
    print("Lats values are", lats.data.array)
    lons_key, lons = u.coordinate("longitude", item=True)

    # Subspacing to the upper hemisphere only
    kwargs = {lats_key: cf.ge(0)}  # greater than or equal to 0, i.e. equator
    upper_hemi_lats_field = u.subspace(**kwargs)
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
    grid_nvectors_field = get_nvectors_across_grid(u)
    print(
        f"*** Full grid of n-vectors for field {u!r} is:\n",
        grid_nvectors_field
    )
    ###grid_nvectors_field.dump()

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
    gc_file_name = "test_outputs/out_gc_distance.nc"
    ###try:
    ###    # If already calculated
    ###    gc_distance_fl = cf.read(gc_file_name)
    if True:  ###except:
        gc_distance_fl = validate_gc_distance_fl(
            origin_nvectors, origin_ll_ref, grid_nvectors_field, u)
        cf.write(gc_distance_fl, gc_file_name)

    chord_distance_fl = compare_chord_distance_fl(
        origin_nvectors, origin_ll_ref, grid_nvectors_field, u)
    cf.write(
        chord_distance_fl,
        "test_outputs/chord_distance_for_comparison.nc"
    )

    print("UPTO")
    euclidean_2d_distance_fl = compare_euclidean_2d_distance_fl(
        upper_hemi_lats_field, u)
    cf.write(
       euclidean_2d_distance_fl,
        "test_outputs/euclidean_2d_distance_for_comparison.nc"
    )
    # SADIE TEMP
    exit()

    # 10. Get fields with bearings (azimuth angles)
    aa_file_name = "test_outputs/out_azimuth_angle.nc"
    ###try:
    ###    # If already calculated
    ###    azimuth_angles_fl = cf.read(aa_file_name)
    if True:  ###except:
        azimuth_angles_fl = validate_azimuth_angles_fl(
            origin_nvectors, origin_ll_ref, grid_nvectors_field, u)
        cf.write(azimuth_angles_fl, aa_file_name)


    # Convert degrees to radians for 0 to 2*pi limits
    convert_degrees_to_radians(azimuth_angles_fl)

    # 11. Apply symmetries of grid to get GC distance and azi angles fields
    #     for all grid points
    gc_lats_to_fields_mapping, empty = map_fields_to_reference_latlon_points(
        gc_distance_fl, lats.data.array)
    # Assume same empty as above - can consolidate this later.
    aa_lats_to_fields_mapping, _ = map_fields_to_reference_latlon_points(
        azimuth_angles_fl, lats.data.array)
    # The negative lat values will not yet have fields assigned. We need
    # to use symmetries to find those fields from the existing ones.
    # 11.a) Reflective symmetry about equator to get lower hemisphere.
    apply_reg_latlon_grid_reflective_symmetry(
        gc_lats_to_fields_mapping, empty)
    apply_reg_latlon_grid_reflective_symmetry(
        aa_lats_to_fields_mapping, empty)
    # 11.b) Rotational symmetry is applied during the integration loop
    # for efficiency. But do some basic testing here on it for validation.
    print(
        "*** Writing out files or basic symmetries testing in "
        "separate script"
    )
    reg_latlon_reflection_testing(
        gc_lats_to_fields_mapping, aa_lats_to_fields_mapping,
        test_lat_val="45.0",  ###"75.699844"
    )
    reg_latlon_rotation_testing(
        gc_lats_to_fields_mapping, lons, example_lat="45.0",  ###"39.81285")
        four_test_lon_values_to_plot=[
            "22.5", "112.5", "247.5", "337.5"],
            ###"1.125", "10.125", "100.125", "200.25"],
    )

    print("+++++++++++++++++ STARTING INTEGRATION +++++++++++++++++++++")
    # 12. Perform the integration
    result_field = perform_integration(
        u, v, lats, lons, gc_lats_to_fields_mapping, aa_lats_to_fields_mapping
    )
    print("All done!")
    print("Result is:")
    result_field.dump()
    cf.write(result_field, "test_outputs/final_result_field.nc")


if __name__ == "__main__":
    sys.exit(main())
