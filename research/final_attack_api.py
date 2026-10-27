"""G15b-v1: pure lawful-input inference; no filesystem, truth, or diagnostic imports."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict
from pwscup2026.rare.outlier import mahalanobis

VERSION = "G15b-lawful-v1"
NUM = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CAT = ["sex", "smoking", "prefecture"]
STATIC = NUM + CAT
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]
MIA_METHODS = ["kde_ratio", "continuous_nn", "full_static_nn", "relative_static_nn", "relative_static_k5", "conditional_relative_nn", "no_geo_conditional_nn", "no_geo_relative_nn"]
AIA_METHODS = ["official_1nn", "log_qi_1nn", "log_qi_window20", "no_geo_sex_1nn", "no_geo_sex_window20"]

def frame_check(frame, columns):
    if list(frame.columns) != columns:
        raise ValueError("Inference input must contain only the declared columns")
    numeric = [col for col in columns if col != "prefecture"]
    if not len(frame) or frame.isna().any().any() or not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Empty or nonfinite inference input")

def checked_keys(frame, column):
    keys = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(keys).all() or not np.equal(keys, np.floor(keys)).all() or frame[column].duplicated().any():
        raise ValueError("Keys must be finite, integral, unique")

def rank_unit(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite raw score")
    return (rankdata(values, method="average") - 0.5) / len(values)

def detected(frame, detector):
    return mahalanobis(frame, np.asarray(detector["mu"]), np.asarray(detector["sigma"]), detector.get("columns", NUM)) >= detector["band"]["lo"]

def block(query_row, donors, drare=None, qrare=None, geography=True):
    sex = donors.sex.to_numpy() == query_row.sex
    mask = sex.copy()
    if geography:
        mask &= donors.prefecture.to_numpy() == query_row.prefecture
        if drare is not None:
            strict = mask & (drare == qrare)
            if strict.any(): mask = strict
    elif drare is not None:
        mask &= (donors.smoking.to_numpy() == query_row.smoking) & (drare == qrare)
    if not mask.any(): mask = sex
    if not mask.any(): mask = np.ones(len(donors), bool)
    return np.flatnonzero(mask)

def predict_mia(c, query, background, cfg, detector):
    frame_check(c, COLUMNS); frame_check(background, COLUMNS)
    frame_check(query, ["record_id"] + STATIC); checked_keys(query, "record_id")
    q = query[STATIC].reset_index(drop=True)
    scale = np.maximum(background[NUM].std(ddof=0).to_numpy(), 1e-12)
    qr = detected(q, detector)
    ds, minima, kth, cond, geo = [], [], [], [], []
    for donors in (c, background):
        numeric = cdist(q[NUM].to_numpy()/scale, donors[NUM].to_numpy()/scale)
        full = numeric**2
        for col in CAT:
            full += 2*(q[col].to_numpy()[:,None] != donors[col].to_numpy()[None,:])
        full = np.sqrt(full)
        dr = detected(donors, detector)
        ds.append(numeric.min(axis=1)); minima.append(full.min(axis=1))
        kth.append(np.partition(full, min(5,len(donors))-1, axis=1)[:,min(5,len(donors))-1])
        cond.append(np.array([full[i,block(row,donors,dr,qr[i])].min() for i,row in enumerate(q.itertuples(index=False))]))
        geo.append(np.array([numeric[i,block(row,donors,dr,qr[i],False)].min() for i,row in enumerate(q.itertuples(index=False))]))
    relative = lambda vals: np.log(vals[1]+1e-9)-np.log(vals[0]+1e-9)
    raw = {"continuous_nn": -ds[0], "full_static_nn": -minima[0], "relative_static_nn": relative(minima), "relative_static_k5": relative(kth), "conditional_relative_nn": relative(cond), "no_geo_conditional_nn": -geo[0], "no_geo_relative_nn": relative(geo)}
    result = pd.DataFrame({"record_id":query.record_id.to_numpy()})
    result["kde_ratio"] = overlap_mia(c,background,q,cfg)
    for name, values in raw.items(): result[name] = rank_unit(values)
    if not np.isfinite(result[MIA_METHODS].to_numpy()).all(): raise ValueError("Nonfinite MIA")
    return result[["record_id"]+MIA_METHODS]

def modal_window(times):
    ts = np.sort(np.asarray(times,dtype=float))
    ends = np.searchsorted(ts,ts+1,side="right")
    start = int(np.argmax(ends-np.arange(len(ts))))
    return float((ts[start]+ts[ends[start]-1])/2)

def log_transform(frame):
    x = frame[QI[:5]].to_numpy(dtype=float).copy()
    x[:,[2,4]] = np.log1p(x[:,[2,4]])
    return x

def predict_aia(c, query, background, cfg):
    frame_check(c,COLUMNS);frame_check(background,COLUMNS)
    frame_check(query,["challenge_row_id"]+QI);checked_keys(query,"challenge_row_id")
    q = query[QI].reset_index(drop=True)
    result = pd.DataFrame({"challenge_row_id":query.challenge_row_id.to_numpy()})
    result["official_1nn"] = aia_predict(c.reset_index(drop=True),q,background,cfg)[1]
    scale = np.maximum(log_transform(background).std(axis=0,ddof=1),1e-12)
    # Squared distance differs by a positive constant from RMS: neighbours are identical.
    distances = cdist(log_transform(q)/scale,log_transform(c)/scale,"sqeuclidean")
    times = c.time.to_numpy(dtype=float)
    for geography, prefix in [(True,"log_qi"),(False,"no_geo_sex")]:
        nearest,modal=[],[]
        for i,row in enumerate(q.itertuples(index=False)):
            ix=block(row,c,geography=geography);d=distances[i,ix];ts=times[ix]
            nearest.append(float(np.sort(ts[d<=d.min()]).mean()))
            boundary=np.partition(d,min(20,len(d))-1)[min(20,len(d))-1]
            modal.append(modal_window(ts[d<=boundary]))
        result[prefix+"_1nn"]=nearest;result[prefix+"_window20"]=modal
    if not np.isfinite(result[AIA_METHODS].to_numpy()).all() or (result[AIA_METHODS]<0).any().any(): raise ValueError("Invalid AIA")
    return result[["challenge_row_id"]+AIA_METHODS]

def keyed_values(frame, key, expected, method):
    checked_keys(frame,key)
    if set(frame[key]) != set(expected): raise ValueError("Missing or extra prediction key")
    return frame.set_index(key).loc[list(expected),method].to_numpy()
