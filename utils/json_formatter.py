# utils/json_formatter.py
import datetime

def frame_to_json(frame_id, tracked_objects):
    """
    Converts tracked objects into JSON format for reasoner
    """
    frame_json = {
        "frame_id": frame_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "objects": tracked_objects
    }
    return frame_json
