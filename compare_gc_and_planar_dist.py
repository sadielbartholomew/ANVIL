import cfplot as cfp
import cf


def mask_outside_annulus(
        gc_distance_field, distance_lower_bound, distance_upper_bound):
    """TODO."""
    wo_query = cf.wo(distance_lower_bound, distance_upper_bound)
    masked_field = gc_distance_field.where(wo_query, cf.masked)

    return masked_field


# Reflection test for lower hemispehre latitudes
gcd_fields = cf.read("test_outputs/out_gc_distance.nc")
planar_2d_fields = cf.read("test_outputs/euclidean_2d_distance_for_comparison.nc")

# Assume both are equivalent size
midway_field = round(len(gcd_fields) / 2)

distance_range_to_mask_a = [1.0e7, 1.1e7]  # a
distance_range_to_mask_b = [1.0e7, 1.1e7]  # b

# Get masked results: 0 is for 0th field, 1 is for midway field
masked_gc_result_0a = mask_outside_annulus(
    gcd_fields[0], *distance_range_to_mask_a)
masked_planar_result_0a = mask_outside_annulus(
    planar_2d_fields[0], *distance_range_to_mask_a)
masked_gc_result_1a = mask_outside_annulus(
    gcd_fields[midway_field], *distance_range_to_mask_a)
masked_planar_result_1a = mask_outside_annulus(
    planar_2d_fields[midway_field], *distance_range_to_mask_a)

masked_gc_result_0b = mask_outside_annulus(
    gcd_fields[0], *distance_range_to_mask_b)
masked_planar_result_0b = mask_outside_annulus(
    planar_2d_fields[0], *distance_range_to_mask_b)
masked_gc_result_1b = mask_outside_annulus(
    gcd_fields[midway_field], *distance_range_to_mask_b)
masked_planar_result_1b = mask_outside_annulus(
    planar_2d_fields[midway_field], *distance_range_to_mask_b)



cfp.gopen(rows=2, columns=1, file="great_circle_vs_planar_dist_1")
cfp.gpos(1)
cfp.con(
    gcd_fields[0], lines=False,
)
cfp.con(
    masked_gc_result_0a, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_0b, lines=False, colorbar=None,
)

cfp.gpos(2)
cfp.con(
    planar_2d_fields[0], lines=False,
)
cfp.con(
    masked_planar_result_0a, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_0b, lines=False, colorbar=None,
)
cfp.gclose()

cfp.gopen(rows=2, columns=1, file="great_circle_vs_planar_dist_2")
cfp.gpos(1)
cfp.con(
    gcd_fields[midway_field], lines=False,
)
cfp.con(
    masked_gc_result_1a, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_1b, lines=False, colorbar=None,
)

cfp.gpos(2)
cfp.con(
    planar_2d_fields[midway_field], lines=False,
)
cfp.con(
    masked_planar_result_1a, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_1b, lines=False, colorbar=None,
)
cfp.gclose()
