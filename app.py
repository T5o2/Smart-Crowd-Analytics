
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import yt_dlp

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# App configuration
# ============================================================
APP_TITLE = "أرصاد الحشود | Crowd Analytics V2"
MODEL_PATH = os.getenv("CROWD_MODEL", "yolo26n.pt")
TRACKER = "bytetrack.yaml"
PERSON_CLASS_ID = 0  # COCO person class
CONFIDENCE = float(os.getenv("CROWD_CONF", "0.35"))
IOU = float(os.getenv("CROWD_IOU", "0.50"))

# Sampling for video/live. 1 = every frame, 3 = one frame out of three, etc.
VIDEO_FRAME_STRIDE = int(os.getenv("VIDEO_FRAME_STRIDE", "3"))
LIVE_FRAME_STRIDE = int(os.getenv("LIVE_FRAME_STRIDE", "6"))

# Optional camera calibration. Set this for a fixed Haram camera.
# Format: "x1,y1;x2,y2;x3,y3;x4,y4" using normalized coordinates 0..1.
DEFAULT_ROI = os.getenv("CROWD_ROI", "")
DEFAULT_COUNTING_LINE = os.getenv("CROWD_COUNT_LINE", "")
DEFAULT_ROI_AREA_M2 = float(os.getenv("CROWD_ROI_AREA_M2", "0"))
DEFAULT_PEOPLE_PER_M2 = float(os.getenv("CROWD_PEOPLE_PER_M2", "3.0"))
DEFAULT_CAPACITY = int(os.getenv("CROWD_CAPACITY", "500"))

# Gender classification is deliberately disabled unless a dedicated,
# validated classifier is supplied. Pixel color must never be used as a proxy.
GENDER_ENABLED = False


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Data structures
# ============================================================
@dataclass
class TrackState:
    last_center: Tuple[float, float]
    last_time: int
    ema_speed: float = 0.0
    last_side: Optional[float] = None


@dataclass
class AnalysisResult:
    frame_bgr: np.ndarray
    count: int
    occupancy: float
    crowd_level: str
    avg_speed: float
    in_count: int
    out_count: int
    flow_direction: str
    density_score: float
    male_pct: Optional[float] = None
    female_pct: Optional[float] = None
    heatmap: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None
    detections: int = 0


# ============================================================
# Utility helpers
# ============================================================
def parse_normalized_points(value: str, expected: Optional[int] = None) -> Optional[List[Tuple[float, float]]]:
    """Parse x,y;x,y format with normalized coordinates."""
    if not value or not value.strip():
        return None

    points: List[Tuple[float, float]] = []
    try:
        for item in value.split(";"):
            x_str, y_str = item.strip().split(",")
            x, y = float(x_str), float(y_str)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Coordinates must be between 0 and 1")
            points.append((x, y))
    except Exception as exc:
        raise ValueError(
            "Invalid points. Use x,y;x,y;x,y format with values from 0 to 1."
        ) from exc

    if expected is not None and len(points) != expected:
        raise ValueError(f"Expected exactly {expected} points.")
    return points


def points_to_pixels(points: Optional[List[Tuple[float, float]]], width: int, height: int):
    if not points:
        return None
    return np.array([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def make_roi_mask(frame_shape, roi_points: Optional[List[Tuple[float, float]]]):
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not roi_points:
        mask[:, :] = 255
        return mask
    polygon = points_to_pixels(roi_points, width, height)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def draw_roi(frame: np.ndarray, roi_mask: np.ndarray, roi_points):
    out = frame.copy()
    if roi_points:
        polygon = points_to_pixels(roi_points, frame.shape[1], frame.shape[0])
        cv2.polylines(out, [polygon], isClosed=True, color=(0, 220, 255), thickness=2)
    # Lightly shade outside the ROI to make the analysis region obvious.
    outside = roi_mask == 0
    if np.any(outside):
        shade = out.copy()
        shade[outside] = (35, 35, 35)
        out = cv2.addWeighted(out, 0.72, shade, 0.28, 0)
    return out


def point_in_roi(x: float, y: float, roi_mask: np.ndarray) -> bool:
    xi = int(np.clip(round(x), 0, roi_mask.shape[1] - 1))
    yi = int(np.clip(round(y), 0, roi_mask.shape[0] - 1))
    return roi_mask[yi, xi] > 0


def line_side(point: Tuple[float, float], line: Optional[List[Tuple[float, float]]]) -> Optional[float]:
    if not line:
        return None
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def format_level(score: float) -> str:
    if score < 20:
        return "منخفض"
    if score < 45:
        return "متوسط"
    if score < 70:
        return "مرتفع"
    return "مرتفع جدًا"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# Crowd analysis engine
# ============================================================
class CrowdAnalyzer:
    """Shared engine used by image, recorded video and live stream tabs."""

    def __init__(
        self,
        model,
        roi_points=None,
        counting_line=None,
        roi_area_m2=0.0,
        people_per_m2=3.0,
        capacity=500,
    ):
        self.model = model
        self.roi_points = roi_points
        self.counting_line = counting_line
        self.roi_area_m2 = max(0.0, float(roi_area_m2))
        self.people_per_m2 = max(0.1, float(people_per_m2))
        self.capacity = max(1, int(capacity))
        self.track_states: Dict[int, TrackState] = {}
        self.in_count = 0
        self.out_count = 0
        self.frame_index = 0
        self.heatmap_accum = None
        self.count_ema = 0.0
        self.occupancy_ema = 0.0

    def reset(self):
        self.track_states.clear()
        self.in_count = 0
        self.out_count = 0
        self.frame_index = 0
        self.heatmap_accum = None
        self.count_ema = 0.0
        self.occupancy_ema = 0.0

    def _prepare_heatmap(self, shape):
        if self.heatmap_accum is None or self.heatmap_accum.shape != shape[:2]:
            self.heatmap_accum = np.zeros(shape[:2], dtype=np.float32)

    def _perspective_weight(self, foot_y: float, height: int) -> float:
        """Heuristic perspective compensation for a fixed overhead/long-range view.

        A real calibrated camera should use a camera-specific perspective map.
        This weighting is intentionally bounded and is not a substitute for calibration.
        """
        y = float(np.clip(foot_y / max(1, height - 1), 0.0, 1.0))
        # Objects near the top of the image are usually farther away and smaller.
        # Increase their contribution modestly, while keeping the factor bounded.
        weight = 1.0 / max(0.35, (0.35 + y) ** 1.35)
        return float(np.clip(weight, 0.8, 4.0))

    def _estimate_capacity(self) -> float:
        if self.roi_area_m2 > 0:
            return max(1.0, self.roi_area_m2 * self.people_per_m2)
        return float(self.capacity)

    def _compute_occupancy(self, boxes: np.ndarray, roi_mask: np.ndarray) -> Tuple[float, float]:
        if boxes.size == 0:
            return 0.0, 0.0

        height, width = roi_mask.shape[:2]
        effective_people = 0.0
        occupied_pixels = 0

        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2.0
            foot_y = y2
            if not point_in_roi(cx, foot_y, roi_mask):
                continue
            weight = self._perspective_weight(foot_y, height)
            effective_people += weight

            ix1 = int(np.clip(x1, 0, width - 1))
            iy1 = int(np.clip(y1, 0, height - 1))
            ix2 = int(np.clip(x2, 0, width - 1))
            iy2 = int(np.clip(y2, 0, height - 1))
            if ix2 > ix1 and iy2 > iy1:
                occupied_pixels += (ix2 - ix1) * (iy2 - iy1)

        capacity = self._estimate_capacity()
        score = np.clip((effective_people / capacity) * 100.0, 0.0, 100.0)

        roi_pixels = max(1, int(cv2.countNonZero(roi_mask)))
        pixel_occupancy = np.clip((occupied_pixels / roi_pixels) * 100.0, 0.0, 100.0)

        # Blend count-based and pixel-occupancy indicators into a stable UI score.
        # Count is dominant because the detector is the primary signal.
        final_occupancy = float(np.clip(score * 0.75 + pixel_occupancy * 0.25, 0, 100))
        return final_occupancy, float(score)

    def _update_flow(self, track_ids, centers, fps: float):
        speeds = []
        directions = {"يمين": 0, "يسار": 0, "أعلى": 0, "أسفل": 0}

        for track_id, center in zip(track_ids, centers):
            x, y = center
            old = self.track_states.get(track_id)
            if old is not None:
                dx = x - old.last_center[0]
                dy = y - old.last_center[1]
                dist = float(np.hypot(dx, dy))
                speed = dist * max(1.0, fps)
                old.ema_speed = 0.75 * old.ema_speed + 0.25 * speed
                speeds.append(old.ema_speed)

                if abs(dx) >= abs(dy):
                    if dx > 0.5:
                        directions["يمين"] += 1
                    elif dx < -0.5:
                        directions["يسار"] += 1
                else:
                    if dy > 0.5:
                        directions["أسفل"] += 1
                    elif dy < -0.5:
                        directions["أعلى"] += 1

                if self.counting_line:
                    new_side = line_side(center, self.counting_line)
                    if old.last_side is not None and new_side is not None:
                        if old.last_side < 0 <= new_side:
                            self.in_count += 1
                        elif old.last_side > 0 >= new_side:
                            self.out_count += 1
                    old.last_side = new_side

                old.last_center = center
                old.last_time = self.frame_index
            else:
                self.track_states[track_id] = TrackState(
                    last_center=center,
                    last_time=self.frame_index,
                    last_side=line_side(center, self.counting_line),
                )

        # Remove stale tracks to avoid unbounded memory use.
        stale = [tid for tid, state in self.track_states.items() if self.frame_index - state.last_time > 150]
        for tid in stale:
            del self.track_states[tid]

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        flow_direction = "غير واضح"
        if directions:
            best = max(directions, key=directions.get)
            if directions[best] > 0:
                flow_direction = best
        return avg_speed, flow_direction

    def _update_heatmap(self, centers, shape):
        self._prepare_heatmap(shape)
        for x, y in centers:
            xi = int(np.clip(round(x), 0, shape[1] - 1))
            yi = int(np.clip(round(y), 0, shape[0] - 1))
            cv2.circle(self.heatmap_accum, (xi, yi), radius=18, color=1.0, thickness=-1)
        self.heatmap_accum = cv2.GaussianBlur(self.heatmap_accum, (0, 0), 12)

    def _render_heatmap(self, frame, roi_mask):
        if self.heatmap_accum is None or float(self.heatmap_accum.max()) <= 0:
            return frame.copy()

        hm = self.heatmap_accum / max(float(self.heatmap_accum.max()), 1e-6)
        hm = (hm * 255).astype(np.uint8)
        color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        masked = np.zeros_like(color)
        masked[roi_mask > 0] = color[roi_mask > 0]
        out = cv2.addWeighted(frame, 0.75, masked, 0.25, 0)
        return out

    def _draw_counting_line(self, frame):
        if not self.counting_line:
            return frame
        points = points_to_pixels(self.counting_line, frame.shape[1], frame.shape[0])
        cv2.line(frame, tuple(points[0]), tuple(points[1]), (255, 200, 0), 3)
        return frame

    def _draw_detections(self, frame, result, roi_mask):
        annotated = frame.copy()
        boxes = []
        centers = []
        ids = []

        if result.boxes is None or len(result.boxes) == 0:
            return annotated, np.empty((0, 4), dtype=np.float32), centers, ids

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(xyxy))
        track_ids = result.boxes.id.cpu().numpy().astype(int).tolist() if result.boxes.id is not None else [-(i + 1) for i in range(len(xyxy))]

        for box, conf, track_id in zip(xyxy, confs, track_ids):
            x1, y1, x2, y2 = map(float, box)
            cx = (x1 + x2) / 2.0
            foot_y = y2
            if not point_in_roi(cx, foot_y, roi_mask):
                continue
            boxes.append([x1, y1, x2, y2])
            centers.append((cx, foot_y))
            ids.append(int(track_id))

            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 220, 120), 2)
            label = f"Person {conf:.2f}"
            if track_id >= 0:
                label = f"ID {track_id}  {conf:.2f}"
            cv2.putText(annotated, label, (int(x1), max(18, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 2, cv2.LINE_AA)
            cv2.circle(annotated, (int(cx), int(foot_y)), 3, (0, 220, 255), -1)

        return annotated, np.array(boxes, dtype=np.float32), centers, ids

    def process(self, frame_bgr: np.ndarray, mode: str = "image", fps: float = 25.0) -> AnalysisResult:
        self.frame_index += 1
        roi_mask = make_roi_mask(frame_bgr.shape, self.roi_points)

        if mode == "image":
            result = self.model.predict(
                source=frame_bgr,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE,
                iou=IOU,
                verbose=False,
            )[0]
        else:
            result = self.model.track(
                source=frame_bgr,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE,
                iou=IOU,
                tracker=TRACKER,
                persist=True,
                verbose=False,
            )[0]

        annotated, boxes, centers, track_ids = self._draw_detections(frame_bgr, result, roi_mask)
        count = len(boxes)

        occupancy, density_score = self._compute_occupancy(boxes, roi_mask)
        self.count_ema = 0.80 * self.count_ema + 0.20 * count
        self.occupancy_ema = 0.80 * self.occupancy_ema + 0.20 * occupancy

        if mode != "image":
            avg_speed, flow_direction = self._update_flow(track_ids, centers, fps)
            self._update_heatmap(centers, frame_bgr.shape)
        else:
            avg_speed, flow_direction = 0.0, "غير متاح"
            self._prepare_heatmap(frame_bgr.shape)
            self._update_heatmap(centers, frame_bgr.shape)

        annotated = self._render_heatmap(annotated, roi_mask)
        annotated = draw_roi(annotated, roi_mask, self.roi_points)
        annotated = self._draw_counting_line(annotated)

        # Top banner
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (15, 22, 30), -1)
        banner = (
            f"People: {count} | Occupancy: {self.occupancy_ema:.1f}% | "
            f"Level: {format_level(self.occupancy_ema)}"
        )
        cv2.putText(annotated, banner, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.63,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return AnalysisResult(
            frame_bgr=annotated,
            count=count,
            occupancy=float(self.occupancy_ema),
            crowd_level=format_level(self.occupancy_ema),
            avg_speed=float(avg_speed),
            in_count=self.in_count,
            out_count=self.out_count,
            flow_direction=flow_direction,
            density_score=float(density_score),
            male_pct=None,
            female_pct=None,
            heatmap=self.heatmap_accum.copy() if self.heatmap_accum is not None else None,
            roi_mask=roi_mask,
            detections=count,
        )


# ============================================================
# Model loading
# ============================================================
def load_model():
    if YOLO is None:
        st.error("Ultralytics is not installed. Run: pip install ultralytics")
        st.stop()
    try:
        return YOLO(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load model '{MODEL_PATH}': {exc}")
        st.stop()


# ============================================================
# UI helpers
# ============================================================
def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            padding: 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(18,24,32,0.90);
            text-align:center;
            min-height: 105px;
        ">
            <div style="font-size:0.85rem;opacity:0.72">{title}</div>
            <div style="font-size:1.75rem;font-weight:700;margin-top:6px">{value}</div>
            <div style="font-size:0.72rem;opacity:0.55;margin-top:3px">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard(result: AnalysisResult):
    cols = st.columns(6)
    metrics = [
        ("الأشخاص المرصودون", f"{result.count}", "Person detections"),
        ("مؤشر الإشغال", f"{result.occupancy:.1f}%", result.crowd_level),
        ("سرعة الحركة", f"{result.avg_speed:.1f}", "relative px/s"),
        ("الدخول", f"{result.in_count}", "line crossings"),
        ("الخروج", f"{result.out_count}", "line crossings"),
        ("اتجاه التدفق", result.flow_direction, "dominant direction"),
    ]
    for col, item in zip(cols, metrics):
        with col:
            metric_card(*item)

    st.markdown("### 👥 تصنيف الرجال والنساء")
    st.info(
        "غير مفعّل في هذه النسخة: لا يتم استنتاج الجنس من لون الملابس أو سطوع البكسلات. "
        "يمكن إضافة مصنّف مستقل ومدرّب ومختبر لاحقًا إذا كان مطلوبًا أكاديميًا."
    )


def show_setup_status():
    st.markdown("### إعداد الكاميرا")
    if st.session_state.get("roi_points_text"):
        st.success("ROI مفعّل: التحليل يقتصر على المنطقة المحددة.")
    else:
        st.warning(
            "لم تحدد ROI. المحرك يحلل كامل الصورة، لذلك لإبعاد المباني والبلاط غير المرغوب فيه "
            "حدد منطقة أرضية الحشد من الشريط الجانبي."
        )
    if st.session_state.get("count_line_text"):
        st.success("خط الدخول/الخروج مفعّل.")
    else:
        st.info("خط الدخول/الخروج اختياري. بدونه ستبقى إحصاءات التدفق الأساسية متاحة دون عد عبور البوابة.")


def parse_sidebar_config():
    with st.sidebar:
        st.header("⚙️ إعدادات التحليل")
        st.caption("نفس الإعدادات تُستخدم في الصورة والفيديو والبث المباشر.")

        model_path = st.text_input("Model path", value=MODEL_PATH)
        conf = st.slider("Confidence", 0.10, 0.90, CONFIDENCE, 0.05)
        iou = st.slider("IoU", 0.20, 0.90, IOU, 0.05)
        frame_stride_video = st.number_input("Video frame stride", min_value=1, max_value=30, value=VIDEO_FRAME_STRIDE)
        frame_stride_live = st.number_input("Live frame stride", min_value=1, max_value=60, value=LIVE_FRAME_STRIDE)

        roi_text = st.text_input(
            "ROI normalized polygon",
            value=st.session_state.get("roi_points_text", DEFAULT_ROI),
            help="Example: 0.05,0.25;0.95,0.25;0.98,0.95;0.02,0.95",
        )
        st.session_state["roi_points_text"] = roi_text.strip()

        count_line_text = st.text_input(
            "Counting line (optional)",
            value=st.session_state.get("count_line_text", DEFAULT_COUNTING_LINE),
            help="Example: 0.50,0.20;0.50,0.90",
        )
        st.session_state["count_line_text"] = count_line_text.strip()

        roi_area = st.number_input(
            "ROI area in m² (optional)",
            min_value=0.0,
            value=float(st.session_state.get("roi_area", DEFAULT_ROI_AREA_M2)),
            step=1.0,
        )
        people_per_m2 = st.number_input(
            "People per m² capacity",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.get("people_per_m2", DEFAULT_PEOPLE_PER_M2)),
            step=0.5,
        )
        capacity = st.number_input(
            "Fallback capacity if area is unknown",
            min_value=1,
            value=int(st.session_state.get("capacity", DEFAULT_CAPACITY)),
            step=10,
        )

        st.session_state["model_path"] = model_path.strip() or MODEL_PATH
        st.session_state["confidence"] = float(conf)
        st.session_state["iou"] = float(iou)
        st.session_state["frame_stride_video"] = int(frame_stride_video)
        st.session_state["frame_stride_live"] = int(frame_stride_live)
        st.session_state["roi_area"] = float(roi_area)
        st.session_state["people_per_m2"] = float(people_per_m2)
        st.session_state["capacity"] = int(capacity)

        st.markdown("---")
        st.markdown("**ملاحظات المعايرة**")
        st.caption(
            "ROI هو أهم إعداد لتجاهل المباني والأسطح غير المرغوبة. "
            "ومساحة ROI بالمتر المربع تجعل مؤشر الإشغال أقرب لمفهوم الكثافة الفعلية."
        )

    try:
        roi_points = parse_normalized_points(roi_text)
        count_line = parse_normalized_points(count_line_text, expected=2) if count_line_text else None
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    return {
        "model_path": st.session_state["model_path"],
        "conf": st.session_state["confidence"],
        "iou": st.session_state["iou"],
        "video_stride": st.session_state["frame_stride_video"],
        "live_stride": st.session_state["frame_stride_live"],
        "roi_points": roi_points,
        "counting_line": count_line,
        "roi_area_m2": st.session_state["roi_area"],
        "people_per_m2": st.session_state["people_per_m2"],
        "capacity": st.session_state["capacity"],
    }


def create_analyzer(config):
    model = load_model()
    analyzer = CrowdAnalyzer(
        model=model,
        roi_points=config["roi_points"],
        counting_line=config["counting_line"],
        roi_area_m2=config["roi_area_m2"],
        people_per_m2=config["people_per_m2"],
        capacity=config["capacity"],
    )
    return analyzer


def show_result_image(result: AnalysisResult):
    st.image(cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)


# ============================================================
# App
# ============================================================
config = parse_sidebar_config()

st.title("🕋 المنصة الذكية لأرصاد الحشود")
st.caption("نسخة V2: Person Detection + ROI + Tracking + Flow + Heatmap + Occupancy")
st.markdown("---")

show_setup_status()

# Keep the user's original three-tab architecture.
tab1, tab2, tab3 = st.tabs(["🖼️ صورة ثابتة", "🎞️ فيديو مسجل", "🔴 بث مباشر (يوتيوب)"])


with tab1:
    uploaded_file = st.file_uploader(
        "📂 ارفع صورة",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_upload",
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### الصورة الأصلية")
            st.image(image, use_container_width=True)
        with c2:
            st.markdown("#### ناتج التحليل")
            result_ph = st.empty()

        if st.button("🚀 بدء تحليل الصورة", key="btn_img"):
            with st.spinner("جاري تحميل النموذج وتحليل الصورة..."):
                analyzer = create_analyzer(config)
                result = analyzer.process(img_array, mode="image", fps=1.0)
            result_ph.image(cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            show_dashboard(result)


with tab2:
    uploaded_video = st.file_uploader(
        "📂 ارفع مقطع فيديو",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_upload",
    )

    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🚀 تشغيل الأرصاد على الفيديو", key="btn_vid"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
            if fps <= 0:
                fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            analyzer = create_analyzer(config)
            image_ph = st.empty()
            dash_ph = st.empty()
            progress = st.progress(0)
            status = st.empty()

            frame_no = 0
            processed = 0
            last_result = None
            try:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_no += 1
                    if frame_no % config["video_stride"] != 0:
                        continue

                    processed += 1
                    last_result = analyzer.process(frame, mode="video", fps=fps / config["video_stride"])
                    image_ph.image(
                        cv2.cvtColor(last_result.frame_bgr, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    with dash_ph.container():
                        show_dashboard(last_result)

                    if total_frames > 0:
                        progress.progress(min(1.0, frame_no / total_frames))
                    status.info(f"Frames processed: {processed}")
            finally:
                cap.release()
                try:
                    os.unlink(video_path)
                except OSError:
                    pass

            if last_result:
                st.success("اكتمل تحليل الفيديو.")


with tab3:
    youtube_url = st.text_input(
        "🔗 أدخل رابط بث الحرم من يوتيوب",
        key="yt_url",
        placeholder="https://www.youtube.com/...",
    )

    st.caption("سيتم محاولة استخراج رابط البث المباشر من YouTube بواسطة yt-dlp ثم تمريره إلى OpenCV.")

    if youtube_url and st.button("🔴 بدء الاستشعار الحي", key="btn_live"):
        img_ph = st.empty()
        dash_ph = st.empty()
        status = st.empty()

        analyzer = create_analyzer(config)
        cap = None
        try:
            status.info("جاري استخراج رابط البث...")
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "quiet": True,
                "noplaylist": True,
                "live_from_start": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                stream_url = info.get("url")
                if not stream_url:
                    formats = info.get("formats") or []
                    candidates = [f for f in formats if f.get("url")]
                    if not candidates:
                        raise RuntimeError("No playable stream URL was found.")
                    candidates.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
                    stream_url = candidates[-1]["url"]

            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise RuntimeError(
                    "OpenCV could not open the YouTube stream. "
                    "Try a stream URL/format that OpenCV/FFmpeg can decode on your machine."
                )

            frame_no = 0
            status.success("البث يعمل. اترك الصفحة مفتوحة أثناء التحليل.")
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    status.warning("توقف البث أو تعذر قراءة الإطار التالي.")
                    break

                frame_no += 1
                if frame_no % config["live_stride"] != 0:
                    continue

                result = analyzer.process(frame, mode="live", fps=25.0 / config["live_stride"])
                img_ph.image(
                    cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                with dash_ph.container():
                    show_dashboard(result)
        except Exception as exc:
            st.error(f"تعذر تشغيل البث: {exc}")
        finally:
            if cap is not None:
                cap.release()


st.markdown("---")
st.caption(
    "ملاحظة منهجية: هذا الإصدار يتجنب تصنيف الرجال/النساء من لون البكسلات. "
    "ولتحويل مؤشر الإشغال إلى كثافة مكانية فعلية بالمتر المربع، يجب ضبط مساحة ROI ومُعايرة الكاميرا."
)
