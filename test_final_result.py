import cfplot as cfp
import cf

res_file = "test_outputs/final_result_field.nc"
#origin_file = None
origin_field = cf.example_field(0)
o = origin_field

# Get output field, ensure there is only one and do some printing to eyeball
r = cf.read(res_file)
assert len(r) == 1
r = r[0]
print("Result field in short is", r)
r.dump()
print("Data is:", r.data.array)

# Prep. for representative plots
res_shape = r.shape
print("Result shape is", r.shape)
print("Original shape is", o.shape)

# Plot
cfp.gopen(rows=2, columns=1, file="final_result_field_plots")
cfp.gpos(1)
cfp.con(
    r, lines=False,
)

cfp.gpos(2)
cfp.con(
    o, lines=False,
)
cfp.gclose()
