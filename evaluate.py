"""
XenSense CSV Evaluation Script

Usage:
    python xensense_evaluation.py path/to/telemetry.csv --outdir results

What it does:
- Loads telemetry CSV with columns: timestamp,frame,track_id,source,class,dist_m,speed_kmh,TTC_s,ml_risk,stm_risk,global_risk,crash_side,x,y,w,h
- Detects collision frame(s) using crash_side when available, otherwise uses TTC spikes and global_risk peaks
- Computes:
    * Collision timeline and impact frame
    * False Positive Rate (before impact)
    * False Negative Rate (during impact window)
    * Early warning times for ml_risk thresholds [0.5,0.7,0.9]
    * ROC/PR using TTC-based proxy label (TTC_s < 1.5 => collision)
    * Speed estimator stability per track
    * Depth (dist_m) smoothness and outlier ratio
    * Tracking metrics: track lengths, fragmentation, bbox jitter
- Saves summary CSV, plots, and a human-readable report (report.md)

Notes:
- This script makes conservative assumptions when ground-truth crash labels are missing.
- You can tune thresholds via CLI args.

"""

import argparse
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

plt.rcParams.update({"figure.max_open_warning": 0})


def load_latest_csv(log_dir="logs"):
    """Return path of newest CSV inside logs directory."""
    import os
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV logs found in '{log_dir}'")
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
    latest = os.path.join(log_dir, csv_files[0])
    print(f"[INFO] Using latest telemetry log: {latest}")
    return latest


def load_csv(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure numeric columns exist
    for c in ["dist_m","speed_kmh","TTC_s","ml_risk","stm_risk","global_risk"]:
        if c not in df.columns:
            df[c] = np.nan
    # convert types
    numcols = ["timestamp","frame","track_id","dist_m","speed_kmh","TTC_s","ml_risk","stm_risk","global_risk","x","y","w","h"]
    for c in numcols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def detect_collision_frames(df, use_crash_side=True, ttc_threshold=1.0, risk_spike_window=5, risk_spike_factor=2.0):
    """Return list of candidate collision frames (frame index numbers)
    Strategy:
      1) If crash_side column has non-empty values -> treat their frames as collision frames.
      2) Otherwise, find frames where TTC_s < ttc_threshold (critical) and/or global_risk spikes.
      3) Also detect sustained high risk after a spike.
    """
    collision_frames = []
    if use_crash_side and 'crash_side' in df.columns:
        # crash_side may be non-empty string or numeric; consider non-null values
        crash_mask = df['crash_side'].notnull() & (df['crash_side'].astype(str).str.strip() != "")
        cf = df.loc[crash_mask, 'frame'].unique().tolist()
        collision_frames.extend([int(x) for x in cf])

    # If nothing found, use TTC small or big risk jumps
    if not collision_frames:
        # TTC-based
        ttc_mask = df['TTC_s'].notnull() & (df['TTC_s'] < ttc_threshold)
        if ttc_mask.any():
            collision_frames.extend(df.loc[ttc_mask, 'frame'].astype(int).unique().tolist())

    # detect global_risk spikes (relative to rolling median)
    if 'global_risk' in df.columns:
        gr = df['global_risk'].fillna(0.0)
        roll_med = gr.rolling(window=30, min_periods=1, center=True).median()
        # spike where global_risk > median * risk_spike_factor and absolute > 0.6
        spikes = df.loc[(gr > roll_med * risk_spike_factor) & (gr > 0.6), 'frame'].astype(int).unique().tolist()
        collision_frames.extend(spikes)

    # unique and sorted
    collision_frames = sorted(list(set(collision_frames)))
    return collision_frames


def compute_basic_stats(df):
    stats = {}
    stats['n_frames'] = df['frame'].nunique()
    stats['n_tracks'] = df['track_id'].nunique()
    stats['classes'] = df['class'].value_counts().to_dict()
    stats['global_risk_mean'] = float(df['global_risk'].mean())
    stats['global_risk_std'] = float(df['global_risk'].std())
    stats['ml_risk_mean'] = float(df['ml_risk'].mean())
    return stats


def early_warning_analysis(df, impact_frame, thresholds=(0.5,0.7,0.9), lookback_frames=120):
    """For each threshold, compute first frame when global_risk crosses threshold before impact_frame.
    Return lead times in seconds (approx) using timestamp differences.
    """
    res = {}
    # restrict to frames before impact
    pre = df[df['frame'] <= impact_frame]
    for thr in thresholds:
        # find last index where risk crosses thr before impact
        mask = (pre['global_risk'] >= thr)
        if not mask.any():
            res[f'lead_{thr}'] = None
            continue
        first = pre.loc[mask, 'frame'].min()
        # compute lead time = time difference between impact and first
        t_first = float(pre.loc[pre['frame'] == first, 'timestamp'].iloc[0])
        t_impact = float(df.loc[df['frame'] == impact_frame, 'timestamp'].iloc[0])
        res[f'lead_{thr}'] = float(t_impact - t_first)
        res[f'lead_frame_{thr}'] = int(impact_frame - first)
    return res


def compute_false_pos_neg(df, impact_frame, pre_window_frames=None, post_window_frames=30, risk_thresh=0.7, safe_ttc=2.0):
    """Compute false positives before impact and missed detections during impact window.
    FP: frames where global_risk > risk_thresh but TTC_s > safe_ttc (i.e., no imminent physics danger)
    FN: during post-impact window (impact_frame -> impact_frame+post_window) where TTC_s < 1.0 but global_risk < risk_thresh
    """
    if pre_window_frames is None:
        pre_mask = df['frame'] < impact_frame
    else:
        pre_mask = (df['frame'] >= max(df['frame'].min(), impact_frame - pre_window_frames)) & (df['frame'] < impact_frame)
    pre_df = df.loc[pre_mask]
    fp_mask = (pre_df['global_risk'] > risk_thresh) & (pre_df['TTC_s'].fillna(9999) > safe_ttc)

    fp_count = int(fp_mask.sum())
    fp_rate = float(fp_count / max(1, len(pre_df)))

    # FN during post window
    post_mask = (df['frame'] >= impact_frame) & (df['frame'] <= impact_frame + post_window_frames)
    post_df = df.loc[post_mask]
    fn_mask = (post_df['TTC_s'].fillna(9999) < 1.0) & (post_df['global_risk'] < risk_thresh)
    fn_count = int(fn_mask.sum())
    fn_rate = float(fn_count / max(1, len(post_df)))

    return {'fp_count':fp_count, 'fp_rate':fp_rate, 'pre_frames':len(pre_df), 'fn_count':fn_count, 'fn_rate':fn_rate, 'post_frames':len(post_df)}


def compute_roc_pr(df, ttc_positive_threshold=1.5):
    # create proxy label: collision if TTC_s < ttc_positive_threshold
    mask = df['TTC_s'].notnull()
    y_true = (df.loc[mask, 'TTC_s'] < ttc_positive_threshold).astype(int).values
    y_score = df.loc[mask, 'global_risk'].fillna(0.0).values
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    prec, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, prec)
    return {'roc_auc':roc_auc, 'fpr':fpr, 'tpr':tpr, 'prec':prec, 'recall':recall, 'pr_auc':pr_auc}


def speed_metrics(df):
    # per-track frame-to-frame delta statistics
    tracks = {}
    for tid, g in df.groupby('track_id'):
        s = g.sort_values('frame')
        sp = s['speed_kmh'].fillna(method='ffill').fillna(0.0).values
        if len(sp) < 2:
            continue
        diffs = np.abs(np.diff(sp))
        tracks[int(tid)] = {
            'n': len(sp),
            'speed_mean': float(np.nanmean(sp)),
            'speed_std': float(np.nanstd(sp)),
            'speed_mae_frame': float(np.mean(diffs)),
            'speed_median_frame_diff': float(np.median(diffs))
        }
    return tracks


def depth_metrics(df):
    d = df['dist_m'].dropna().values
    if len(d) == 0:
        return {}
    diffs = np.abs(np.diff(d))
    outlier_thresh = np.median(diffs) * 8.0 + 1e-6
    outliers = np.sum(diffs > outlier_thresh)
    return {
        'n_samples': int(len(d)),
        'dist_mean': float(np.nanmean(d)),
        'dist_median': float(np.nanmedian(d)),
        'dist_std': float(np.nanstd(d)),
        'median_abs_diff': float(np.median(diffs)) if len(diffs)>0 else 0.0,
        'outlier_count': int(outliers),
        'outlier_ratio': float(outliers / max(1, len(diffs)))
    }


def tracking_metrics(df):
    res = {}
    tracks = list(df['track_id'].dropna().unique())
    lengths = []
    fragmentations = []
    jitters = []
    for tid in tracks:
        g = df[df['track_id']==tid].sort_values('frame')
        frames = g['frame'].values
        lengths.append(len(frames))
        # fragmentation: count gaps >1 in frame sequence
        if len(frames) > 1:
            gaps = np.sum(np.diff(frames) > 1)
            fragmentations.append(gaps)
        else:
            fragmentations.append(0)
        # jitter: std of center movement
        centers = np.vstack([(g['x'] + g['w']/2).fillna(0).values, (g['y'] + g['h']/2).fillna(0).values]).T
        if centers.shape[0] > 1:
            cent_diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            jitters.append(float(np.std(cent_diffs)))
    res['n_tracks'] = int(len(tracks))
    res['track_len_mean'] = float(np.mean(lengths)) if lengths else 0
    res['track_len_median'] = float(np.median(lengths)) if lengths else 0
    res['fragmentation_mean'] = float(np.mean(fragmentations)) if fragmentations else 0
    res['bbox_jitter_median'] = float(np.median(jitters)) if jitters else 0
    return res


def plot_time_series(df, outdir, impact_frame=None):
    os.makedirs(outdir, exist_ok=True)
    t = df['timestamp'].values - df['timestamp'].values[0]
    plt.figure(figsize=(12,4))
    plt.plot(t, df['global_risk'].fillna(0.0), label='global_risk')
    plt.plot(t, df['ml_risk'].fillna(0.0), label='ml_risk', alpha=0.7)
    plt.plot(t, df['stm_risk'].fillna(0.0), label='stm_risk', alpha=0.7)
    if 'TTC_s' in df.columns:
        plt.plot(t, (df['TTC_s'].fillna(20)/20.0), label='TTC_s (scaled)', alpha=0.6)
    plt.xlabel('time (s)')
    plt.legend()
    if impact_frame is not None:
        try:
            t_imp = float(df.loc[df['frame']==impact_frame, 'timestamp'].iloc[0] - df['timestamp'].iloc[0])
            plt.axvline(t_imp, color='r', linestyle='--', label='impact')
        except Exception:
            pass
    plt.title('Risk Time Series')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'risk_time_series.png'))
    plt.close()

    # distance and speed
    plt.figure(figsize=(12,4))
    plt.plot(t, df['dist_m'].fillna(method='ffill'), label='dist_m')
    plt.plot(t, df['speed_kmh'].fillna(0), label='speed_kmh')
    plt.xlabel('time (s)')
    plt.legend()
    plt.title('Distance and Speed')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dist_speed_time_series.png'))
    plt.close()


def save_report(outdir, stats, early, fp_fn, roc_res, spd, depthm, trackm):
    rpt = []
    rpt.append('# XenSense Evaluation Report')
    rpt.append('\n## Basic stats')
    for k,v in stats.items():
        rpt.append(f'- **{k}**: {v}')
    rpt.append('\n## Early warning analysis')
    for k,v in early.items():
        rpt.append(f'- **{k}**: {v}')
    rpt.append('\n## False positives / negatives')
    for k,v in fp_fn.items():
        rpt.append(f'- **{k}**: {v}')
    if roc_res:
        rpt.append('\n## ROC / PR')
        rpt.append(f'- **ROC AUC**: {roc_res["roc_auc"]:.4f}')
        rpt.append(f'- **PR AUC**: {roc_res["pr_auc"]:.4f}')
    rpt.append('\n## Speed metrics (sample of tracks)')
    for tid, info in list(spd.items())[:8]:
        rpt.append(f'- Track {tid}: n={info["n"]}, mean={info["speed_mean"]:.2f}, frame_mae={info["speed_mae_frame"]:.3f}')
    rpt.append('\n## Depth metrics')
    for k,v in depthm.items():
        rpt.append(f'- **{k}**: {v}')
    rpt.append('\n## Tracking metrics')
    for k,v in trackm.items():
        rpt.append(f'- **{k}**: {v}')

    with open(os.path.join(outdir, 'report.md'), 'w') as f:
        f.write('\n'.join(rpt))


def main():
    parser = argparse.ArgumentParser(description='XenSense telemetry CSV evaluator')
    parser.add_argument('--outdir', default='xensense_eval', help='output directory')
    parser.add_argument('--ttc_proxy', type=float, default=1.5, help='TTC threshold for proxy collision label')
    parser.add_argument('--risk_thresh', type=float, default=0.7, help='global_risk threshold considered dangerous')
    args = parser.parse_args()

    # Automatically load latest CSV
    csv_path = load_latest_csv("logs")

    df = load_csv(csv_path)
    os.makedirs(args.outdir, exist_ok=True)

    stats = compute_basic_stats(df)

    # Detect collision frames
    coll_frames = detect_collision_frames(df, use_crash_side=True, ttc_threshold=1.0)
    if coll_frames:
        impact_frame = coll_frames[0]
    else:
        # fallback
        if df['TTC_s'].notnull().any():
            try:
                impact_frame = int(df.loc[df['TTC_s'] < 1.0, 'frame'].min())
            except Exception:
                impact_frame = int(df['frame'].median())
        else:
            impact_frame = int(df.loc[df['global_risk'].idxmax(), 'frame'])

    early = early_warning_analysis(df, impact_frame, thresholds=(0.5, 0.7, 0.9))
    fp_fn = compute_false_pos_neg(df, impact_frame, risk_thresh=args.risk_thresh)
    roc_res = compute_roc_pr(df, ttc_positive_threshold=args.ttc_proxy)
    spd = speed_metrics(df)
    depthm = depth_metrics(df)
    trackm = tracking_metrics(df)

    plot_time_series(df, args.outdir, impact_frame=impact_frame)

    # Save summary
    summary = {
        'csv_used': csv_path,
        'impact_frame': impact_frame,
        'collision_candidates': coll_frames,
        'basic_stats': stats,
        'early_warning': early,
        'fp_fn': fp_fn,
        'roc': {'roc_auc': roc_res['roc_auc']} if roc_res else None,
        'speed_sample': {k: spd[k] for k in list(spd)[:8]},
        'depth_metrics': depthm,
        'tracking_metrics': trackm
    }

    import json
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    save_report(args.outdir, stats, early, fp_fn, roc_res, spd, depthm, trackm)

    print("Evaluation complete.")
    print(f"[INFO] CSV evaluated: {csv_path}")
    print(f"[INFO] Results saved in: {args.outdir}")
