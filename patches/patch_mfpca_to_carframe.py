import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

CAR_MODS = '["NoseVectorCar.x", "NoseVectorCar.y", "EyeDirCar.x", "EyeDirCar.y", "SteeringInput"]'

def patch_cell(idx, mod_var, nf_var, m_var=None):
    s = "".join(nb["cells"][idx]["source"]) if isinstance(nb["cells"][idx]["source"], list) else nb["cells"][idx]["source"]
    orig = s
    # 1) modalities -> car-frame list (replace `<mod_var> = outlier_features`)
    s = s.replace(f"{mod_var} = outlier_features",
                  f"{mod_var} = {CAR_MODS}   # CAR-FRAME channels")
    # 2) nf based on len(outlier_features) -> len(car mods) = 5 (used only for old path; harmless)
    s = s.replace(f"{nf_var} = len(outlier_features)",
                  f"{nf_var} = len({mod_var})")
    # 3) path -> car_reference folder + filename
    s = s.replace('+ f"segment_around_{evt}.csv")',
                  '+ "car_reference/" + f"car_reference_{evt}.csv")')
    s = s.replace('fpath = (_MF_DATA_DIR\n             + "cleaned_data/data_segment/outliers_removed_{_MF_NF}features/"\n             + f"segment_around_{evt}_nearest_neighbor.csv")',
                  'fpath = (_MF_DATA_DIR + "cleaned_data/data_segment/car_reference/"\n             + f"car_reference_{evt}.csv")')
    # A1 builds path differently:
    s = s.replace('fpath = (_MF_DATA_DIR\n             + "cleaned_data/data_segment/"\n             + f"segment_around_{evt}.csv")',
                  'fpath = (_MF_DATA_DIR + "cleaned_data/data_segment/car_reference/"\n             + f"car_reference_{evt}.csv")')
    # the A1 fpath actually:
    s = s.replace('fpath = (_MF_DATA_DIR\n             + "cleaned_data/data_segment/outliers_removed_{_MF_NF}features/"',
                  'fpath = (_MF_DATA_DIR + "cleaned_data/data_segment/car_reference/"')
    if s == orig:
        return False
    ast.parse(s)
    nb["cells"][idx]["source"] = s
    return True

# Cell A1 (119): _MF_MODALITIES, _MF_NF
print("A1 (119):", patch_cell(119, "_MF_MODALITIES", "_MF_NF", "_MF_M"))
# Group cell (122): _GR_MODALITIES, _GR_NF
print("Group (122):", patch_cell(122, "_GR_MODALITIES", "_GR_NF"))

# Verify final path/mod lines
for i in [119, 122]:
    s = "".join(nb["cells"][i]["source"])
    print(f"--- cell {i} now ---")
    for ln in s.splitlines():
        if any(k in ln for k in ["_MODALITIES =", "car_reference", "segment_around"]):
            print("   ", ln.strip()[:110])

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Saved.")
