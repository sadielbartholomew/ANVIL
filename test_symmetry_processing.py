import cfplot as cfp
import cf


gc_distances = cf.read("gc_reflection_test_01.nc")
cfp.gopen(rows=2, columns=1, file="pos_vs_neg_lats_gc_fields")
cfp.gpos(1)
cfp.con(
    gc_distances[0], lines=False,
)

cfp.gpos(2)
cfp.con(
    gc_distances[1], lines=False,
)
cfp.gclose()
