import cv2
import requests
import time
import json
import os
import argparse
import threading
import queue
from datetime import datetime


DEFAULT_API_KEY = os.getenv("RAPIDAPI_KEY", "").strip()

API_URL = "https://mask-detection2.p.rapidapi.com/detect-mask"
API_HOST = "mask-detection2.p.rapidapi.com"


def get_value(data, *keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data:
            return data[key]
    lower_map = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if str(key).lower() in lower_map:
            return lower_map[str(key).lower()]
    return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, min_value, max_value):
    return max(min_value, min(int(value), max_value))


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def status_from_text(value):
    text = str(value).lower().strip()

    no_mask_words = [
        "without_mask", "without mask", "no_mask", "no mask", "nomask",
        "not wearing", "unmasked", "false", "0"
    ]
    mask_words = [
        "with_mask", "with mask", "mask", "masked", "wearing mask", "true", "1"
    ]

    for word in no_mask_words:
        if word in text:
            return "NO_MASK"

    for word in mask_words:
        if word in text:
            return "MASK"

    return "UNKNOWN"



def load_local_detectors():
    detectors = {
        "dnn_face": None,
        "haar_face": None,
        "haar_eye": None,
    }

    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"

    if os.path.exists(model_file) and os.path.exists(config_file):
        try:
            detectors["dnn_face"] = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("The DNN face detection model has been successfully loaded.")
        except Exception as e:
            print("DNN face detection model loading failed:", e)
    else:
        print("The DNN model file was not found. Therefore, OpenCV Haar will be used as the local pre-detection.")
        print("需要 DNN 时，请把 deploy.prototxt 和 res10_300x300_ssd_iter_140000.caffemodel 放到当前目录。")

    # 2. Haar face fallback
    haar_face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_face = cv2.CascadeClassifier(haar_face_path)
    if not haar_face.empty():
        detectors["haar_face"] = haar_face
        print("Haar The face detector has been successfully loaded.")
    else:
        print("Haar Face detector loading failed")

    haar_eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    haar_eye = cv2.CascadeClassifier(haar_eye_path)
    if not haar_eye.empty():
        detectors["haar_eye"] = haar_eye
        print("Haar Eye detector loaded successfully")
    else:
        print("Haar Eye detector loading failed")

    return detectors


def detect_faces_dnn(frame, net, conf_threshold=0.35):
    if net is None:
        return []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            x1 = clamp(x1, 0, w - 1)
            y1 = clamp(y1, 0, h - 1)
            x2 = clamp(x2, 0, w - 1)
            y2 = clamp(y2, 0, h - 1)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1, float(confidence), "DNN_FACE"))
    return faces


def detect_faces_haar(frame, detector):
    if detector is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    found = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(50, 50)
    )

    return [(int(x), int(y), int(w), int(h), 0.0, "HAAR_FACE") for (x, y, w, h) in found]


def detect_eyes_haar(frame, detector):
    if detector is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    found = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(20, 20)
    )

    return [(int(x), int(y), int(w), int(h), 0.0, "EYE") for (x, y, w, h) in found]


def detect_person_local(frame, detectors):

    faces = detect_faces_dnn(frame, detectors.get("dnn_face"), conf_threshold=0.35)

    if len(faces) == 0:
        faces = detect_faces_haar(frame, detectors.get("haar_face"))

    eyes = []
    if len(faces) == 0:
        eyes = detect_eyes_haar(frame, detectors.get("haar_eye"))

    has_person = len(faces) > 0 or len(eyes) >= 1
    return faces, eyes, has_person


def draw_local_boxes(frame, faces, eyes):
    for (x, y, w, h, conf, source) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 2)
        label = source if conf == 0 else f"{source} {conf:.2f}"
        cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)

    for (x, y, w, h, conf, source) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, "EYE TRIGGER", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def parse_bbox(box, img_w, img_h, roboflow_center=False):
    if box is None:
        return None

    if isinstance(box, dict):
        # RapidAPI / AI Engine: Left Top Width Height
        left = get_value(box, "Left", "left", default=None)
        top = get_value(box, "Top", "top", default=None)
        width = get_value(box, "Width", "width", "w", default=None)
        height = get_value(box, "Height", "height", "h", default=None)
        right = get_value(box, "Right", "right", default=None)
        bottom = get_value(box, "Bottom", "bottom", default=None)

        # Roboflow: x y width height，通常 x/y 是中心点
        x = get_value(box, "x", "X", default=None)
        y = get_value(box, "y", "Y", default=None)

        if right is not None and bottom is not None and left is not None and top is not None:
            left = to_float(left)
            top = to_float(top)
            right = to_float(right)
            bottom = to_float(bottom)
            if left <= 1 and top <= 1 and right <= 1 and bottom <= 1:
                x1, y1, x2, y2 = left * img_w, top * img_h, right * img_w, bottom * img_h
            else:
                x1, y1, x2, y2 = left, top, right, bottom

        elif left is not None and top is not None and width is not None and height is not None:
            left = to_float(left)
            top = to_float(top)
            width = to_float(width)
            height = to_float(height)
            if left <= 1 and top <= 1 and width <= 1 and height <= 1:
                x1, y1 = left * img_w, top * img_h
                x2, y2 = (left + width) * img_w, (top + height) * img_h
            else:
                x1, y1 = left, top
                x2, y2 = left + width, top + height

        elif x is not None and y is not None and width is not None and height is not None:
            x = to_float(x)
            y = to_float(y)
            width = to_float(width)
            height = to_float(height)

            if x <= 1 and y <= 1 and width <= 1 and height <= 1:
                x *= img_w
                y *= img_h
                width *= img_w
                height *= img_h

            if roboflow_center:
                x1, y1 = x - width / 2, y - height / 2
                x2, y2 = x + width / 2, y + height / 2
            else:
                x1, y1 = x, y
                x2, y2 = x + width, y + height
        else:
            return None

    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        x = to_float(box[0])
        y = to_float(box[1])
        w = to_float(box[2])
        h = to_float(box[3])
        if x <= 1 and y <= 1 and w <= 1 and h <= 1:
            x1, y1 = x * img_w, y * img_h
            x2, y2 = (x + w) * img_w, (y + h) * img_h
        else:
            x1, y1 = x, y
            x2, y2 = x + w, y + h
    else:
        return None

    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 0, img_w - 1)
    y2 = clamp(y2, 0, img_h - 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2



def normalize_result(raw_json, img_w, img_h):
    detections = []

    if not isinstance(raw_json, dict):
        return detections

    data = raw_json

    body = get_value(data, "body", "Body", default=None)
    if body is not None:
        if isinstance(body, str):
            try:
                data = json.loads(body)
            except Exception:
                data = raw_json
        elif isinstance(body, dict):
            data = body

    try:
        with open("last_api_response.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    predictions = get_value(data, "predictions", "Predictions", default=None)
    if isinstance(predictions, list):
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            label_text = get_value(pred, "class", "label", "name", "Class", default="")
            status = status_from_text(label_text)
            confidence = to_float(get_value(pred, "confidence", "score", "Confidence", default=0.0))
            box = parse_bbox(pred, img_w, img_h, roboflow_center=True)
            detections.append({"status": status, "confidence": confidence, "box": box})
        return detections

    persons = get_value(data, "Persons", "persons", "detections", "results", default=[])
    summary = get_value(data, "Summary", "summary", default={})

    persons_with_mask = set(str(x) for x in get_value(summary, "PersonsWithMask", "personsWithMask", default=[]))
    persons_without_mask = set(str(x) for x in get_value(summary, "PersonsWithoutMask", "personsWithoutMask", default=[]))
    persons_unknown = set(str(x) for x in get_value(summary, "PersonsIndeterminate", "personsIndeterminate", default=[]))

    if not isinstance(persons, list):
        return detections

    for i, person in enumerate(persons):
        if not isinstance(person, dict):
            continue

        person_id = str(get_value(person, "Id", "id", default=i))

        box = get_value(person, "BoundingBox", "boundingBox", "box", "bbox", default=None)
        pixel_box = parse_bbox(box, img_w, img_h)

        mask_obj = get_value(person, "Mask", "mask", default={})
        confidence = 0.0
        status = "UNKNOWN"

        if person_id in persons_with_mask:
            status = "MASK"
        elif person_id in persons_without_mask:
            status = "NO_MASK"
        elif person_id in persons_unknown:
            status = "UNKNOWN"

        if isinstance(mask_obj, dict):
            confidence = to_float(get_value(mask_obj, "Confidence", "confidence", "score", default=0.0))
            value = get_value(mask_obj, "Value", "value", "label", "class", default="")
            if status == "UNKNOWN":
                status = status_from_text(value)
        else:
            if status == "UNKNOWN":
                status = status_from_text(mask_obj)

        direct_mask = get_value(person, "mask_detected", "maskDetected", "has_mask", "hasMask", default=None)
        if status == "UNKNOWN" and isinstance(direct_mask, bool):
            status = "MASK" if direct_mask else "NO_MASK"

        if confidence == 0:
            confidence = to_float(get_value(person, "confidence", "Confidence", "score", default=0.0))

        detections.append({"status": status, "confidence": confidence, "box": pixel_box})

    return detections



class APIWorker(threading.Thread):
    def __init__(self, api_key, request_queue, shared_data, lock, timeout):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.request_queue = request_queue
        self.shared_data = shared_data
        self.lock = lock
        self.timeout = timeout
        self.running = True
        self.session = requests.Session()
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": API_HOST
        }

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                item = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            frame_id, jpg_bytes, img_w, img_h = item
            start_time = time.time()

            try:
                files = {"image": ("frame.jpg", jpg_bytes, "image/jpeg")}
                response = self.session.post(API_URL, headers=self.headers, files=files, timeout=self.timeout)
                api_time = time.time() - start_time

                print("\nAPI STATUS =", response.status_code)

                if response.status_code != 200:
                    err = response.text[:200]
                    if "quota" in err.lower() or "exceed" in err.lower():
                        err = "API quota exceeded. Monthly limit has been used up."
                    with self.lock:
                        self.shared_data["error"] = f"API Error {response.status_code}: {err}"
                    print("API ERROR =", err)
                    continue

                raw_json = response.json()
                detections = normalize_result(raw_json, img_w, img_h)

                with self.lock:
                    self.shared_data["detections"] = detections
                    self.shared_data["error"] = None
                    self.shared_data["api_time"] = api_time
                    self.shared_data["last_update"] = now()

                print(f"解析结果: {detections}")

            except Exception as e:
                with self.lock:
                    self.shared_data["error"] = str(e)
                print("API ERROR =", str(e))


def draw_results(frame, detections):
    mask_count = 0
    no_mask_count = 0
    unknown_count = 0

    for det in detections:
        status = det.get("status", "UNKNOWN")
        confidence = to_float(det.get("confidence", 0.0))
        box = det.get("box")

        if status == "MASK":
            label = f"MASK {confidence:.2f}"
            color = (0, 255, 0)
            mask_count += 1
        elif status == "NO_MASK":
            label = f"NO MASK {confidence:.2f}"
            color = (0, 0, 255)
            no_mask_count += 1
        else:
            label = "UNKNOWN"
            color = (0, 255, 255)
            unknown_count += 1

        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return mask_count, no_mask_count, unknown_count


def draw_big_status(frame, has_person, waiting_api, mask_count, no_mask_count, unknown_count):
    h, w = frame.shape[:2]

    if not has_person:
        text = "NO FACE"
        color = (0, 255, 255)
        sub = "No API call"
    elif waiting_api:
        text = "FACE FOUND"
        color = (255, 120, 0)
        sub = "Waiting for API result"
    elif no_mask_count > 0:
        text = "NO MASK!"
        color = (0, 0, 255)
        sub = "Please wear a mask"
    elif mask_count > 0:
        text = "MASK OK"
        color = (0, 255, 0)
        sub = "Mask detected"
    else:
        text = "UNKNOWN"
        color = (0, 255, 255)
        sub = "Check last_api_response.json"

    cv2.rectangle(frame, (0, 0), (w, 95), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 1.55
    thick = 4
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = int((w - size[0]) / 2)
    cv2.putText(frame, text, (x, 55), font, scale, color, thick)

    sub_scale = 0.62
    sub_size, _ = cv2.getTextSize(sub, font, sub_scale, 2)
    sub_x = int((w - sub_size[0]) / 2)
    cv2.putText(frame, sub, (sub_x, 85), font, sub_scale, (255, 255, 255), 2)


def draw_panel(frame, fps, api_call_count, api_time, last_update, error):
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"API Calls: {api_call_count}",
        f"API Time: {api_time:.2f}s" if api_time else "API Time: waiting",
        f"Last Update: {last_update}" if last_update else "Last Update: none",
        "Press Q to quit"
    ]

    y = 125
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 25

    if error:
        cv2.putText(frame, error[:100], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-interval", type=float, default=5.0, help="有人脸时，几秒调用一次 API，建议 3-5 秒")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--api-key", type=str, default="")
    args = parser.parse_args()

    api_key = args.api_key or DEFAULT_API_KEY
    if not api_key:
        print("Error: No API Key. Please run python mask_fixed.py --api-key YourKey or set the environment variable RAPIDAPI_KEY.")
        return

    detectors = load_local_detectors()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("The camera failed to turn on. You can try again：python mask_fixed.py --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    request_queue = queue.Queue(maxsize=1)
    lock = threading.Lock()
    shared_data = {
        "detections": [],
        "error": None,
        "api_time": None,
        "last_update": None
    }

    worker = APIWorker(api_key, request_queue, shared_data, lock, args.timeout)
    worker.start()

    frame_id = 0
    last_api_submit = 0
    api_call_count = 0
    fps = 0.0
    fps_counter = 0
    last_fps_time = time.time()

    print("The program has started successfully.")
    print("Recovery point：1) API sends the clean original image, but does not send the image with a frame; 2) Supports Summary to determine MASK/NO_MASK; 3) Eye detection triggers the API.")
    print(f"API Call interval：{args.min_interval} Seconds")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            fps_counter += 1
            current_time = time.time()

            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time

            # 关键修复：保存一份干净原图给 API，后面再画框
            api_frame = frame.copy()
            img_h, img_w = api_frame.shape[:2]

            faces, eyes, has_person = detect_person_local(api_frame, detectors)

            if not has_person:
                with lock:
                    shared_data["detections"] = []
                    shared_data["error"] = None
                    shared_data["api_time"] = None
                    shared_data["last_update"] = None

            should_call_api = (
                has_person
                and frame_id % args.frame_skip == 0
                and current_time - last_api_submit >= args.min_interval
            )

            if should_call_api:
                ok, encoded = cv2.imencode(".jpg", api_frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if ok:
                    jpg_bytes = encoded.tobytes()
                    if request_queue.full():
                        try:
                            request_queue.get_nowait()
                        except queue.Empty:
                            pass
                    request_queue.put((frame_id, jpg_bytes, img_w, img_h))
                    last_api_submit = current_time
                    api_call_count += 1
                    print(f"[{now()}] Detecting a person/eye, invoking the API， {api_call_count} times")

            draw_local_boxes(frame, faces, eyes)

            with lock:
                detections = list(shared_data["detections"])
                error = shared_data["error"]
                api_time = shared_data["api_time"]
                last_update = shared_data["last_update"]

            mask_count, no_mask_count, unknown_count = draw_results(frame, detections)
            waiting_api = has_person and len(detections) == 0 and error is None

            draw_big_status(frame, has_person, waiting_api, mask_count, no_mask_count, unknown_count)
            draw_panel(frame, fps, api_call_count, api_time, last_update, error)

            cv2.imshow("Mask Detection Fixed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Manual stop")

    finally:
        worker.stop()
        try:
            request_queue.put(None)
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("Program exit")
        print(f"The number of API calls made this time：{api_call_count}")


if __name__ == "__main__":
    main()
