import cfplot as cfp
import cf

import matplotlib.pyplot as plt


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
print("0 field is")
gcd_fields[0].dump()
print("mid field is")
gcd_fields[midway_field].dump()
print("0 field is")
planar_2d_fields[0].dump()
print("mid field is")
planar_2d_fields[midway_field].dump()


distance_range_to_mask_a = [1.05e7, 1.10e7]  # a
distance_range_to_mask_b = [0.05e7, 0.10e7]  # a
distance_range_to_mask_c = [0.55e7, 0.60e7]  # b

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

masked_gc_result_0c = mask_outside_annulus(
    gcd_fields[0], *distance_range_to_mask_c)
masked_planar_result_0c = mask_outside_annulus(
    planar_2d_fields[0], *distance_range_to_mask_c)
masked_gc_result_1c = mask_outside_annulus(
    gcd_fields[midway_field], *distance_range_to_mask_c)
masked_planar_result_1c = mask_outside_annulus(
    planar_2d_fields[midway_field], *distance_range_to_mask_c)



cfp.gopen(
    rows=2, columns=1, file="great_circle_vs_planar_dist_1",
    bottom=0.2, top=0.9
)
cfp.mapset(proj="robin")
plt.suptitle(
    (
        "Comparison of great circle distance (top) with Euclidean/planar 2D\n"
        "distance (bottom) for integration calculations on the spherical Earth\n"
        "for the gridpoint at lat 89.14 and lon 0.00 (near North Pole)"
    ),
    fontsize=13,
)
cfp.cscale("WhiteYellowOrangeRed")
cfp.gpos(1)
cfp.con(
    gcd_fields[0], lines=False, colorbar=None,
)
cfp.cscale("GrayWhiteGray", ncols=1)
cfp.con(
    masked_gc_result_0a, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_0b, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_0c, lines=False, colorbar=None,
)

cfp.gpos(2)
cfp.cscale("WhiteYellowOrangeRed")
cfp.con(
    planar_2d_fields[0], lines=False,
    colorbar_title=(
        "Distance calculated assuming given geometry from (89.14, 0.00)\n"
        "in metres x 10^7 where possible r annuli are picked out for\n"
        "illustration at ranges 0.05-0.1, 0.55-0.06 and 1.05-1.1"
    )
)
cfp.cscale("GrayWhiteGray", ncols=1)
cfp.con(
    masked_planar_result_0a, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_0b, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_0c, lines=False, colorbar=None,
)
cfp.gclose()

cfp.gopen(
    rows=2, columns=1, file="great_circle_vs_planar_dist_2",
    bottom=0.2, top=0.9,
)
plt.suptitle(
    (
        "Comparison of great circle distance (top) with Euclidean/planar 2D\n"
        "distance (bottom) for integration calculations on the spherical Earth\n"
        "for the gridpoint at lat 38.69 and lon 0.00 (near Spain)"
    ),
    fontsize=13,
)
cfp.cscale("WhiteYellowOrangeRed")
cfp.gpos(1)
cfp.con(
    gcd_fields[midway_field], lines=False, colorbar=None,
)
cfp.cscale("GrayWhiteGray", ncols=1)
cfp.con(
    masked_gc_result_1a, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_1b, lines=False, colorbar=None,
)
cfp.con(
    masked_gc_result_1c, lines=False, colorbar=None,
)

cfp.gpos(2)
cfp.mapset(proj="robin")
cfp.cscale("WhiteYellowOrangeRed")
cfp.con(
    planar_2d_fields[midway_field], lines=False,
    ###colorbar_title="BOO"
    colorbar_title=(
        "Distance calculated assuming given geometry from (38.69, 0.00)\n"
        "in metres x 10^7 where possible r annuli are picked out for\n"
        "illustration at ranges 0.05-0.1, 0.55-0.06 and 1.05-1.1"
    )
)
cfp.cscale("GrayWhiteGray", ncols=1)
cfp.con(
    masked_planar_result_1a, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_1b, lines=False, colorbar=None,
)
cfp.con(
    masked_planar_result_1c, lines=False, colorbar=None,
)
cfp.gclose()
