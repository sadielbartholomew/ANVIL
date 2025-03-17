import cfplot as cfp
import cf

# Set up
res = "160x320"
res_2 = "5x8"
GC_DATASET = f"test_outputs/out_gc_distance_{res}.nc"
AA_DATASET = f"test_outputs/out_azimuth_angle_{res}.nc"

a = cf.read(GC_DATASET)
b = cf.read(AA_DATASET)
midway_field = round(len(a) / 2)


# Plotting
cfp.gopen(rows=2, columns=2, file="gc_distance_with_azi_angles")

cfp.mapset(proj="robin")

# For larger datasets - at the North pole
cfp.gpos(1)
cfp.con(a[0], lines=False)

cfp.gpos(2)
cfp.con(b[0], lines=False)

# Now a field mid-way through the latitude points

cfp.gpos(3)
cfp.con(a[midway_field], lines=False)

cfp.gpos(4)
cfp.con(b[midway_field], lines=False)

cfp.gclose()

# Bonus plot!
cfp.gopen(file="bonus_plot_azi_angles_at_pole")
cfp.mapset(proj="npstere")
cfp.con(b[0], lines=False)
cfp.gclose()



