import cfplot as cfp
import cf

# Set up
res = "160x320"
res_2 = "5x8"
GC_DATASET = f"test_outputs/out_gc_distance_{res}.nc"
AA_DATASET = f"test_outputs/out_azimuth_angle_{res}.nc"


def plot_fields(gc_distance_fields, azimuth_fields):
    """TODO."""
    # Plotting
    cfp.gopen(rows=2, columns=2, file="gc_distance_with_azi_angles")

    cfp.mapset(proj="robin")

    # For larger datasets - at the North pole
    cfp.gpos(1)
    cfp.con(
        gc_distance_fields[0], lines=False,
        title="Great circle distances (left plots)",
    )

    cfp.gpos(2)
    cfp.con(
        azimuth_fields[0], lines=False,
        title="Azimuth angles relative to North (right plots)",
    )

    # Now a field mid-way through the latitude points
    cfp.gpos(3)
    cfp.con(
        gc_distance_fields[midway_field], lines=False
    )

    cfp.gpos(4)
    cfp.con(
        azimuth_fields[midway_field], lines=False
    )

    cfp.gclose()

    # Bonus plot!
    cfp.gopen(file="bonus_plot_azi_angles_at_pole")
    cfp.mapset(proj="npstere")
    cfp.con(
        azimuth_fields[0], lines=False,
        title="Azimuth angles relative to N for point very close to North Pole",
    )
    cfp.gclose()


def mask_outside_annulus(
        gc_distance_field, distance_lower_bound, distance_upper_bound):
    """TODO."""
    wo_query = cf.wo(distance_lower_bound, distance_upper_bound)
    masked_field = gc_distance_field.where(wo_query, cf.masked)

    return masked_field


def plot_masking_examples(
        gc_distances, distance_range_to_mask, plot_name_append, proj="cyl"):
    """TODO."""
    print("Plotting masking example for two different points")
    result_1 = mask_outside_annulus(gc_distances[0], *distance_range_to_mask)
    result_2 = mask_outside_annulus(
        gc_distances[midway_field], *distance_range_to_mask)
    cfp.gopen(file=f"masking_example_{plot_name_append}")
    cfp.mapset(proj=proj)
    cfp.con(
        result_1, lines=False, colorbar=None,
    )
    cfp.con(
        result_2, lines=False,
        title=(
            "Example: masking of all points outside great circle distance\n"
            f"{distance_range_to_mask[0]} to {distance_range_to_mask[1]} m "
            "for two different lat-lon origin points"
        ),
    )
    cfp.gclose()


if __name__ == "__main__":
    """TODO."""
    # Get ANVIL output fields
    gc_distances = cf.read(GC_DATASET)
    azi_angles = cf.read(AA_DATASET)
    # Assume the two field sets above have the same size
    midway_field = round(len(gc_distances) / 2)

    # Plot the fields
    plot_fields(gc_distances, azi_angles)

    # Illustrative plots of masking
    plot_masking_examples(gc_distances, [1.0e7, 1.1e7], "1")
    plot_masking_examples(gc_distances, [0.2e7, 0.3e7], "2", proj="npstere")
