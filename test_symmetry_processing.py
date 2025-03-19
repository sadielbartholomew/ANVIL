import cfplot as cfp
import cf


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
