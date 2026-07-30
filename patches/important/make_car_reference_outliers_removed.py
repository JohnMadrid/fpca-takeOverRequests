"""Create car-reference files from the OUTLIERS-REMOVED (DBSCAN nearest-neighbor)
segment files, mirroring make_car_reference.py but on the cleaned inputs.

Per driver: own-HMD 0.8Hz Butterworth heading -> de-rotate world (x,z) of
EyeDirWorldCombined and NoseVector into car-frame LATERAL.
New columns:
  NoseVectorCar.x = head lateral (car-frame)
  EyeDirCar.x     = eye  lateral (car-frame)
(.y vertical is unaffected by heading rotation; use world .y columns directly.)

In : data/cleaned_data/data_segment/outliers_removed_5features/
        segment_around_<event>_nearest_neighbor.csv
Out: data/cleaned_data/data_segment/car_reference/outliers_removed_5features/
        car_reference_<event>.csv
Keeps all original columns; appends the 2 new ones.
"""
import pandas as pd, numpy as np, os, sys
sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import butter, filtfilt

base   = "data/cleaned_data/data_segment"
indir  = os.path.join(base, "outliers_removed_5features")
outdir = os.path.join(base, "car_reference", "outliers_removed_5features")
os.makedirs(outdir, exist_ok=True)
events = ["StagEventNew", "FallingRocksEventNew", "MotorcyclistEvent"]
FS, CUT, ORD = 50.0, 0.8, 3

def butter_lp(x):
    b, a = butter(ORD, CUT/(FS/2.0), btype="low")
    return filtfilt(b, a, x)

def add_car_frame(df):
    """Per driver: own-HMD 0.8Hz heading, de-rotate (x,z) -> lateral.
    Identical to make_car_reference.py:add_car_frame."""
    df = df.sort_values(["uid", "time_from_event"]).reset_index(drop=True)
    nose_lat = np.full(len(df), np.nan)
    eye_lat  = np.full(len(df), np.nan)
    for u, idx in df.groupby("uid").groups.items():
        idx = np.array(idx)
        sub = df.loc[idx]
        hx = sub["HmdPosition.x"].values.astype(float)
        hz = sub["HmdPosition.z"].values.astype(float)
        if len(hx) < 9:
            th = np.unwrap(np.arctan2(np.gradient(hz), np.gradient(hx)))
        else:
            th = np.unwrap(np.arctan2(np.gradient(butter_lp(hz)),
                                      np.gradient(butter_lp(hx))))
        c, s = np.cos(th), np.sin(th)
        nose_lat[idx] = -s*sub["NoseVector.x"].values + c*sub["NoseVector.z"].values
        eye_lat[idx]  = -s*sub["EyeDirWorldCombined.x"].values + c*sub["EyeDirWorldCombined.z"].values
    df["NoseVectorCar.x"] = nose_lat
    df["EyeDirCar.x"]     = eye_lat
    return df

for e in events:
    fin = os.path.join(indir, f"segment_around_{e}_nearest_neighbor.csv")
    if not os.path.exists(fin):
        print(f"  {e}: source not found, skip"); continue
    df = pd.read_csv(fin, low_memory=False)
    n0 = len(df.columns)
    df = add_car_frame(df)
    # sanity: no NaN in the new lateral columns
    nan_nose = int(df["NoseVectorCar.x"].isna().sum())
    nan_eye  = int(df["EyeDirCar.x"].isna().sum())
    fout = os.path.join(outdir, f"car_reference_{e}.csv")
    df.to_csv(fout, index=False)
    print(f"  {e}: wrote {fout}")
    print(f"      {len(df)} rows, +{len(df.columns)-n0} cols, {df['uid'].nunique()} uids, "
          f"NaN(NoseCar.x)={nan_nose} NaN(EyeCar.x)={nan_eye}")
print("Done.")
