import tempfile
from pathlib import Path

import cv2
import imageio
import streamlit as st
import supervision as sv
from streamlit_image_coordinates import streamlit_image_coordinates
from trackers import ByteTrackTracker
from ultralytics import YOLO

from counter import LINES, TrafficCounter, draw_approaches, draw_summary, draw_tracks
from detector import CLASS_NAMES, detect, needs_heavy_model
from motion import MotionGate

IMAGE_WIDTH = 1280
TRACK_THRESHOLD = 0.3
MODES = {"zone": "Zona", "in": "Garis masuk", "out": "Garis keluar"}
HINTS = ["Klik minimal 3 titik di sekeliling lajur yang menuju simpang.",
         "Klik 2 titik melintang lajur, di ujung zona yang jauh dari simpang.",
         "Klik 2 titik di garis henti, ujung zona yang dekat simpang."]


def new_draft():
    return {"zone": [], "in": [], "out": []}


def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, round(image.shape[0] * IMAGE_WIDTH / image.shape[1])))


def process(video_path, approaches):
    light_model, heavy_model = YOLO("yolo26n.pt"), YOLO("yolo26m.pt")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion = MotionGate()
    tracker = ByteTrackTracker(frame_rate=fps, track_activation_threshold=TRACK_THRESHOLD,
                               high_conf_det_threshold=TRACK_THRESHOLD)
    counter = TrafficCounter(approaches, CLASS_NAMES)
    lines = [a[kind] for a in approaches for kind in LINES if a[kind]]
    output_path = video_path.with_name(f"hasil_{video_path.stem}.mp4")
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=2)
    progress = st.progress(0.0, "Memproses video...")
    preview = st.empty()
    occupancy = {a["name"]: [] for a in approaches}
    tracks = sv.Detections.empty()
    frame_index = detected_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if motion.should_detect(frame):
            detections = detect(light_model, frame)
            if needs_heavy_model(detections, lines):
                detections = detect(heavy_model, frame)
            tracks = tracker.update(detections, timestamp=frame_index / fps)
            tracks = tracks[tracks.tracker_id >= 0]
            detected_frames += 1
        current = counter.update(tracks)
        for name, vehicles in current.items():
            occupancy[name].append(sum(vehicles.values()))
        image = draw_tracks(draw_approaches(frame, approaches), tracks, counter)
        image = draw_summary(resize(image), current, counter.counts)
        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if frame_index % round(fps) == 0:
            preview.image(image, channels="BGR")
            progress.progress(min(frame_index / total_frames, 1.0),
                              f"Memproses video... {frame_index}/{total_frames} frame")
        frame_index += 1
    writer.close()
    preview.empty()
    progress.progress(1.0, "Selesai")
    summary = [{"jalan masuk": name, "rata-rata di zona": round(sum(values) / len(values), 1),
                "maksimum di zona": max(values), **counter.counts[name]} for name, values in occupancy.items()]
    return {"frames": frame_index, "detected_frames": detected_frames, "summary": summary, "video": output_path}


st.set_page_config(page_title="Pemantau Lalu Lintas", layout="wide")
st.title("Pemantau Lalu Lintas Berbasis Video")
st.caption("Jumlah kendaraan per jenis di setiap jalan masuk simpang: yang sedang berada di zona, yang melewati garis "
           "masuk, dan yang melewati garis keluar. Data seperti ini menjadi dasar pengaturan lampu lalu lintas (ATCS).")

upload = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
if upload is None:
    st.stop()

state = st.session_state
if state.get("video_id") != upload.file_id:
    video_path = Path(tempfile.gettempdir()) / upload.name
    video_path.write_bytes(upload.getbuffer())
    state.update(video_id=upload.file_id, video_path=video_path, frame=cv2.VideoCapture(str(video_path)).read()[1],
                 approaches=[], draft=new_draft(), result=None)

click = state.get("click")
if click and click["unix_time"] != state.get("last_click"):
    state.last_click = click["unix_time"]
    ratio = state.frame.shape[1] / click["width"]
    point = (round(click["x"] * ratio), round(click["y"] * ratio))
    shape = state.draft[state.mode]
    state.draft[state.mode] = [point] if state.mode in LINES and len(shape) == 2 else shape + [point]

st.header("Atur jalan masuk")
left, right = st.columns([3, 1])
with right:
    names = [a["name"] for a in state.approaches]
    name = st.text_input("Nama jalan masuk", f"jalan_{len(names) + 1}")
    mode = st.radio("Gambar", MODES, format_func=MODES.get, captions=HINTS, key="mode")
    if st.button("Hapus titik terakhir", disabled=not state.draft[mode]):
        state.draft[mode].pop()
        st.rerun()
    incomplete_line = any(len(state.draft[kind]) == 1 for kind in LINES)
    if st.button("Simpan jalan masuk",
                 disabled=len(state.draft["zone"]) < 3 or incomplete_line or not name or name in names):
        state.approaches.append({"name": name, **state.draft})
        state.draft = new_draft()
        st.rerun()
    st.write(f"Tersimpan: {', '.join(names) or 'belum ada'}")
    if st.button("Hapus jalan masuk terakhir", disabled=not names):
        state.approaches.pop()
        st.rerun()

image = draw_approaches(state.frame.copy(), state.approaches + [{"name": name, **state.draft}])
for shape in state.draft.values():
    for point in shape:
        cv2.circle(image, point, 8, (255, 255, 255), -1)
with left:
    streamlit_image_coordinates(cv2.cvtColor(resize(image), cv2.COLOR_BGR2RGB), key="click", width="stretch",
                                image_format="JPEG", cursor="crosshair")

st.header("Proses")
left, right = st.columns([3, 1])
if right.button("Mulai proses", disabled=not state.approaches):
    with left:
        state.result = process(state.video_path, state.approaches)

if state.result:
    result = state.result
    st.header("Hasil")
    left, right = st.columns([3, 1])
    left.video(str(result["video"]))
    right.metric("Frame diproses", result["frames"])
    right.metric("Frame dideteksi", f"{100 * result['detected_frames'] / result['frames']:.0f}%")
    st.dataframe(result["summary"], hide_index=True)
