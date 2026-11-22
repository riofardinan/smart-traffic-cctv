import cv2
import numpy as np

COLORS = [(0, 200, 0), (0, 150, 255), (0, 220, 220), (80, 80, 255)]
LINES = {"in": "masuk", "out": "keluar"}
LINE_COLORS = {"in": (255, 255, 0), "out": (255, 0, 255)}
TRACK_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def bottom_center(box):
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2), float(y2)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


class TrafficCounter:
    def __init__(self, approaches, class_names):
        self.approaches = approaches
        self.class_names = class_names
        self.zones = {a["name"]: np.array(a["zone"], np.int32) for a in approaches}
        self.counts = {a["name"]: {**dict.fromkeys(class_names.values(), 0), "masuk": 0, "keluar": 0}
                       for a in approaches}
        self.last_points = {}
        self.counted = set()
        self.last_line = {}

    def update(self, tracks):
        current = {a["name"]: dict.fromkeys(self.class_names.values(), 0) for a in self.approaches}
        for box, track_id, class_id in zip(tracks.xyxy, tracks.tracker_id, tracks.class_id):
            vehicle = self.class_names[int(class_id)]
            point = bottom_center(box)
            previous = self.last_points.get(track_id)
            self.last_points[track_id] = point
            for approach in self.approaches:
                name = approach["name"]
                if cv2.pointPolygonTest(self.zones[name], point, False) >= 0:
                    current[name][vehicle] += 1
                for kind, label in LINES.items():
                    line, key = approach[kind], (track_id, name, kind)
                    if line and previous and key not in self.counted and intersect(previous, point, *line):
                        self.counts[name][label] += 1
                        if kind == "out":
                            self.counts[name][vehicle] += 1
                        self.counted.add(key)
                        self.last_line[track_id] = kind
        return current


def draw_approaches(frame, approaches):
    for index, approach in enumerate(approaches):
        name, color = approach["name"], COLORS[index % len(COLORS)]
        zone = np.array(approach["zone"], np.int32).reshape(-1, 2)
        if len(zone) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        if len(zone):
            cv2.polylines(frame, [zone], len(zone) >= 3, color, 3)
            cv2.putText(frame, name, (int(zone[:, 0].min()), int(max(zone[:, 1].min() - 10, 30))), FONT, 1, color, 3)
        for kind, label in LINES.items():
            if len(approach[kind]) == 2:
                a, b = approach[kind]
                cv2.line(frame, a, b, LINE_COLORS[kind], 4)
                cv2.putText(frame, label, (max(a[0], b[0]) + 10, max(a[1], b[1])), FONT, 1, LINE_COLORS[kind], 3)
    return frame


def draw_tracks(frame, tracks, counter):
    for box, track_id, class_id in zip(tracks.xyxy, tracks.tracker_id, tracks.class_id):
        x1, y1, x2, y2 = box.astype(int)
        color = LINE_COLORS.get(counter.last_line.get(track_id), TRACK_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{track_id} {counter.class_names[int(class_id)]}", (x1, y1 - 6), FONT, 0.6, color, 2)
    return frame


def draw_summary(frame, current, counts):
    rows = [f"{name} | di zona: {', '.join(f'{n} {vehicle}' for vehicle, n in vehicles.items())} | "
            f"masuk {counts[name]['masuk']} | keluar {counts[name]['keluar']}" for name, vehicles in current.items()]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 20 + 35 * len(rows)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, row in enumerate(rows):
        cv2.putText(frame, row, (15, 40 + 35 * i), FONT, 0.8, (255, 255, 255), 2)
    return frame
