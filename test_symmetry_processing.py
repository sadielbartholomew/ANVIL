import cfplot as cfp
import cf


# Reflection test for lower hemispehre latitudes
test_fields = cf.read("all_reflection_test_01.nc")
cfp.gopen(rows=2, columns=1, file="pos_vs_neg_lats_gc_fields")
cfp.gpos(1)
cfp.con(
    test_fields[0], lines=False,
)

cfp.gpos(2)
cfp.con(
    test_fields[1], lines=False,
)
cfp.gclose()

cfp.gopen(rows=2, columns=1, file="pos_vs_neg_lats_aa_fields")
cfp.gpos(1)
cfp.con(
    test_fields[2], lines=False,
)

cfp.gpos(2)
cfp.con(
    test_fields[3], lines=False,
)
cfp.gclose()

# Rotation test for longitudes
test_fields = cf.read("gc_rotation_test_01.nc")
cfp.gopen(rows=2, columns=2, file="rotation_test_gc_only")
cfp.gpos(1)
cfp.con(
    test_fields[0], lines=False,
)

cfp.gpos(2)
cfp.con(
    test_fields[1], lines=False,
)
cfp.gpos(3)
cfp.con(
    test_fields[2], lines=False,
)

cfp.gpos(4)
cfp.con(
    test_fields[3], lines=False,
)
cfp.gclose()
