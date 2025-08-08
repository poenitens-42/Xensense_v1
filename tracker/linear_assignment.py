import numpy as np

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_w, inter_h = max(0, xb - xa), max(0, yb - ya)
    inter_area = inter_w * inter_h
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def min_cost_matching(tracks, detections, iou_threshold=0.3):
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, track in enumerate(tracks):
        for d, det in enumerate(detections):
            cost_matrix[t, d] = 1 - iou(track.to_tlwh(), det.tlwh)

    matches, unmatched_tracks, unmatched_dets = [], [], []
    while cost_matrix.size > 0:
        t, d = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        if 1 - cost_matrix[t, d] < iou_threshold:
            break
        matches.append((t, d))
        cost_matrix = np.delete(cost_matrix, t, 0)
        cost_matrix = np.delete(cost_matrix, d, 1)

    matched_tracks = [m[0] for m in matches]
    matched_dets = [m[1] for m in matches]
    unmatched_tracks = [t for t in range(len(tracks)) if t not in matched_tracks]
    unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets
