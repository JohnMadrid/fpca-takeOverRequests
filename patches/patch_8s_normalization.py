import json, ast, sys
sys.stdout.reconfigure(encoding="utf-8")
NB = r"C:\Users\erene\OneDrive\Desktop\fpca-takeOverRequests\Analysis_ED_PCA.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

def find(substr):
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code" and substr in "".join(c["source"]):
            return i
    return None

def setsrc(i,s):
    ast.parse(s); nb["cells"][i]["source"]=s

patched=[]

# ---------- cell 126: PCA -- CAR REFERENCE FRAME ----------
i=find("PCA -- CAR REFERENCE FRAME")
s="".join(nb["cells"][i]["source"])
old='''    curves = []
    for _, sub in df_event.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        data_scaled = StandardScaler().fit_transform(sub[modalities_car].values)
        curves.append(data_scaled.T)
    W = len(curves)
    if W == 0:
        return None, 0
    arr = np.stack(curves, axis=0)              # (W, M, T)
    arr = np.transpose(arr, (0, 2, 1))          # (W, T, M)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    arr = arr[:, crop, :]
    _, T, _ = arr.shape'''
new='''    # 8s-WINDOW NORMALISATION: crop to [-4,4]s FIRST, then per-driver z over 8s.
    curves = []
    for _, sub in df_event.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        _raw = sub[modalities_car].values.astype(float)         # (501, M) time-ordered
        _secs0 = np.linspace(-5, 5, _raw.shape[0])
        _raw = _raw[(_secs0 >= -4) & (_secs0 <= 4)]              # crop to 8s
        curves.append(StandardScaler().fit_transform(_raw))     # z over 8s window
    W = len(curves)
    if W == 0:
        return None, 0
    arr = np.stack(curves, axis=0)              # (W, T, M)
    secs = np.linspace(-5, 5, points_per_window)
    secs = secs[(secs >= -4) & (secs <= 4)]
    _, T, _ = arr.shape'''
assert old in s, "cell126 loader block not found"
setsrc(i, s.replace(old,new)); patched.append(126)

# ---------- cell 127: PCA CAR-FRAME -- PERMUTATION (_load_arr_c) ----------
i=find("PCA CAR-FRAME -- PERMUTATION")
s="".join(nb["cells"][i]["source"])
old='''    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        curves.append(StandardScaler().fit_transform(sub[_MODS_C].values))   # (T, M)
    if not curves:
        return None, None
    arr  = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    return arr[:, crop, :], secs[crop]'''
new='''    # 8s-WINDOW NORMALISATION: crop first, then per-driver z over 8s.
    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window:
            continue
        _raw = sub[_MODS_C].values.astype(float)
        _s0 = np.linspace(-5, 5, _raw.shape[0])
        _raw = _raw[(_s0 >= -4) & (_s0 <= 4)]
        curves.append(StandardScaler().fit_transform(_raw))                  # (T, M)
    if not curves:
        return None, None
    arr  = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, points_per_window)
    secs = secs[(secs >= -4) & (secs <= 4)]
    return arr, secs'''
assert old in s, "cell127 loader block not found"
setsrc(i, s.replace(old,new)); patched.append(127)

# ---------- cell 129: PCA CAR-FRAME (OUTLIERS REMOVED) (_load_arr_or) ----------
i=find("PCA CAR-FRAME (OUTLIERS REMOVED)")
s="".join(nb["cells"][i]["source"])
old='''    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub[_MODS_OR].values))
    if not curves: return None, None
    arr = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, arr.shape[1])
    crop = (secs >= -4) & (secs <= 4)
    return arr[:, crop, :], secs[crop]'''
new='''    # 8s-WINDOW NORMALISATION: crop first, then per-driver z over 8s.
    curves = []
    for _, sub in df.groupby('uid'):
        if len(sub) != points_per_window: continue
        _raw = sub[_MODS_OR].values.astype(float)
        _s0 = np.linspace(-5, 5, _raw.shape[0])
        _raw = _raw[(_s0 >= -4) & (_s0 <= 4)]
        curves.append(StandardScaler().fit_transform(_raw))
    if not curves: return None, None
    arr = np.stack(curves, axis=0)
    secs = np.linspace(-5, 5, points_per_window)
    secs = secs[(secs >= -4) & (secs <= 4)]
    return arr, secs'''
assert old in s, "cell129 loader block not found"
setsrc(i, s.replace(old,new)); patched.append(129)

# ---------- cell 137: OPTION 1 (_u_load) ----------
i=find("OPTION 1: TASK-ALIGNED")
s="".join(nb["cells"][i]["source"])
old='''    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub.sort_values('time_from_event')[_U_VARS].values))
    arr=np.stack(curves,0)                      # (W,501,M) per-driver standardised
    secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    return arr[:,crop,:], secs[crop]'''
new='''    # 8s-WINDOW NORMALISATION: crop first, then per-driver z over 8s.
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        _raw=sub.sort_values('time_from_event')[_U_VARS].values.astype(float)
        _s0=np.linspace(-5,5,_raw.shape[0]); _raw=_raw[(_s0>=-4)&(_s0<=4)]
        curves.append(StandardScaler().fit_transform(_raw))
    arr=np.stack(curves,0)                      # (W,T,M) per-driver z over 8s
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]
    return arr, secs'''
assert old in s, "cell137 loader block not found"
setsrc(i, s.replace(old,new)); patched.append(137)

# ---------- cell 139: OPTION 3 (_s_edcurve) ----------
i=find("OPTION 3: ED(t) SHAPE")
s="".join(nb["cells"][i]["source"])
old='''    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        curves.append(StandardScaler().fit_transform(sub.sort_values('time_from_event')[_S_VARS].values))
    arr=np.transpose(np.stack(curves,0),(0,2,1))   # (W,T,M)->? keep (W,501,M)
    arr=np.stack(curves,0)
    secs=np.linspace(-5,5,arr.shape[1]); crop=(secs>=-4)&(secs<=4)
    arr=arr[:,crop,:]; secs=secs[crop]; W,T,M=arr.shape'''
new='''    # 8s-WINDOW NORMALISATION: crop first, then per-driver z over 8s.
    curves=[]
    for _,sub in df.groupby('uid'):
        if len(sub)!=points_per_window: continue
        _raw=sub.sort_values('time_from_event')[_S_VARS].values.astype(float)
        _s0=np.linspace(-5,5,_raw.shape[0]); _raw=_raw[(_s0>=-4)&(_s0<=4)]
        curves.append(StandardScaler().fit_transform(_raw))
    arr=np.stack(curves,0)
    secs=np.linspace(-5,5,points_per_window); secs=secs[(secs>=-4)&(secs<=4)]; W,T,M=arr.shape'''
assert old in s, "cell139 loader block not found"
setsrc(i, s.replace(old,new)); patched.append(139)

json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Patched 8s-window normalisation in cells: {patched}. Total cells: {len(nb['cells'])}.")
