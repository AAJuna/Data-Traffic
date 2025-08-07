import sys
import time
import os
import getpass
import platform
from datetime import datetime
import socket
import argparse

def show_ascii_header():
    header = r"""
┏━━━┓╋╋╋╋╋╋╋╋┏┓
┃┏━┓┃╋╋╋╋╋╋╋┏┛┗┓
┃┃╋┗╋━━┳┓┏┳━╋┓┏╋┳━┓┏━━┓
┃┃╋┏┫┏┓┃┃┃┃┏┓┫┃┣┫┏┓┫┏┓┃
┃┗━┛┃┗┛┃┗┛┃┃┃┃┗┫┃┃┃┃┗┛┃
┗━━━┻━━┻━━┻┛┗┻━┻┻┛┗┻━┓┃
╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋┏━┛┃
╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋┗━━┛
┏━┓┏━┓╋╋╋╋┏━━┓┏┓
┃┃┗┛┃┃╋╋╋╋┃┏┓┃┃┃
┃┏┓┏┓┣┓╋┏┓┃┗┛┗┫┃┏━━┳━━┳━━┳┳━┓┏━━┳━━┓
┃┃┃┃┃┃┃╋┃┃┃┏━┓┃┃┃┃━┫━━┫━━╋┫┏┓┫┏┓┃━━┫
┃┃┃┃┃┃┗━┛┃┃┗━┛┃┗┫┃━╋━━┣━━┃┃┃┃┃┗┛┣━━┃
┗┛┗┛┗┻━┓┏┛┗━━━┻━┻━━┻━━┻━━┻┻┛┗┻━┓┣━━┛
╋╋╋╋╋┏━┛┃╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋┏━┛┃
╋╋╋╋╋┗━━┛╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋┗━━┛
        """
    byline = "\n      Billboard Baliho Counting Impression By Iklan Yoda\n"
    print('\033[95m' + header + '\033[0m')  # Neon Magenta
    print('\033[96m' + byline + '\033[0m')   # Cyan

def play_loading_animation(duration=3.5):
    """Animasi loading bar dengan gradasi warna cyan ke magenta"""
    bar_length = 32
    loading_str = " [ LOADING TRACKERKARTA ] "
    start_time = time.time()
    i = 0
    while (time.time() - start_time) < duration:
        pos = i % (bar_length * 2)
        if pos < bar_length:
            bar = "█" * pos + " " * (bar_length - pos)
        else:
            p2 = bar_length * 2 - pos
            bar = "█" * p2 + " " * (bar_length - p2)
        
        # Efek gradasi cyan ke magenta
        colored_bar = ""
        for j, char in enumerate(bar):
            if char == "█":
                ratio = j / bar_length
                r = int(0 * (1-ratio) + 255 * ratio)
                g = int(255 * (1-ratio) + 0 * ratio)
                b = 255
                colored_bar += f'\033[38;2;{r};{g};{b}m{char}'
            else:
                colored_bar += char

        sys.stdout.write(f"\r\033[95m{loading_str}\033[0m\033[96m[{colored_bar}\033[0m\033[96m]\033[0m")
        sys.stdout.flush()
        time.sleep(0.04)
        i += 1
    sys.stdout.write("\r" + " " * (len(loading_str) + bar_length + 10) + "\r")
    sys.stdout.flush()

try:
    import torch
    torch.backends.cudnn.benchmark = True
    TORCH_GPU = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if TORCH_GPU else "N/A"
except Exception:
    TORCH_GPU = False
    GPU_NAME = "N/A"

def get_system_info():
    user = getpass.getuser()
    sysinfo = platform.system() + " " + platform.release()
    now = datetime.now()
    tanggal = now.strftime('%Y-%m-%d %H:%M:%S')
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except:
        ip = "N/A"
    loc = ""
    return user, sysinfo, tanggal, ip, loc

def print_env_info():
    user, sysinfo, tanggal, ip, loc = get_system_info()
    print('\033[92m')
    print(f" [*] User         : {user}")
    print(f" [*] System       : {sysinfo}")
    print(f" [*] Date & Time  : {tanggal}")
    print(f" [*] GPU Support  : {'Yes' if TORCH_GPU else 'No'}")
    print(f" [*] GPU Device   : {GPU_NAME}")
    print('\033[0m')
    print('\033[95m' + "-"*55 + '\033[0m')
    print('\033[96m' + "    Console Live Counting Impression    " + '\033[0m')
    print('\033[95m' + "-"*55 + '\033[0m')

show_ascii_header()
play_loading_animation()
print_env_info()
time.sleep(1)

parser = argparse.ArgumentParser(description="Vehicle counting with YOLOv8 tracking")
parser.add_argument("--video", type=str, default="Sampel-Jateng.mp4", help="path to input video")
parser.add_argument("--weights", type=str, default="yolov8m.pt", help="YOLO model path")
parser.add_argument("--output", type=str, default="hasil_cyberpunk_trail.mp4", help="output video path")
parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="tracker config (e.g., bytetrack.yaml or botsort.yaml)")
parser.add_argument("--imgsz", type=int, default=960, help="inference image size")
parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
parser.add_argument("--device", type=str, default="cuda" if TORCH_GPU else "cpu", help="compute device")
parser.add_argument("--half", action="store_true", help="use FP16 for faster inference on GPU")
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np

CYBER_COLORS = [
    (255, 0, 255),   # Neon pink
    (0, 255, 255),   # Neon cyan
    (255, 255, 0),   # Neon yellow
    (0, 255, 128),   # Neon green
    (128, 0, 255),   # Neon purple
    (0, 128, 255),   # Neon blue
]

def cyberpunk_box(img, x1, y1, x2, y2, color, thickness=3, corner_len=25, glow=True):
    for i in range(thickness, 0, -1):
        c = tuple(int(x) for x in np.array(color) * (0.4 + 0.6 * i / thickness))
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), c, i)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), c, i)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), c, i)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), c, i)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), c, i)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), c, i)
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), c, i)  # horizontal
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), c, i)  # vertical
    if glow:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=thickness * 4)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

def cyberpunk_text(img, text, org, color, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)

MAX_TRAIL_LENGTH = 40
trails = defaultdict(lambda: deque(maxlen=MAX_TRAIL_LENGTH))

model = YOLO(args.weights)
model.to(args.device)
if args.half and TORCH_GPU:
    model.model.half()
class_list = model.names

cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

output_path = args.output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

window_name = "Sabar Bos Baru Di Itung..."
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

line_y_red = frame_height // 2
class_counts = defaultdict(int)
crossed_ids = set()
object_tracker = {}
rasio_meter_per_pixel = 0.05

allowed_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

last_console_refresh = time.time()
refresh_interval = 0.5  # detik, refresh console setiap 0.5 detik

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    trail_layer = np.zeros_like(frame)
    glow_layer = np.zeros_like(frame)

    timestamp = time.time()
    results = model.track(
        frame,
        persist=True,
        tracker=args.tracker,
        classes=allowed_classes,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False
    )

    cv2.line(frame, (0, line_y_red), (frame_width, line_y_red), (255, 0, 255), 3)
    cyberpunk_text(frame, 'Garis Hitung', (10, line_y_red - 10), (255, 0, 255), 1, 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        debug_counts = defaultdict(int)

        for idx, (box, track_id, class_idx) in enumerate(zip(boxes, track_ids, class_indices)):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = class_list[class_idx]
            debug_counts[class_name] += 1

            color = CYBER_COLORS[class_idx % len(CYBER_COLORS)]

            speed_kmph = 0
            if track_id in object_tracker:
                prev_cx, prev_cy, prev_time = object_tracker[track_id]
                dt = timestamp - prev_time
                if dt > 0:
                    distance_pixels = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                    distance_meters = distance_pixels * rasio_meter_per_pixel
                    speed_kmph = (distance_meters / dt) * 3.6
            object_tracker[track_id] = (cx, cy, timestamp)

            trails[track_id].append((cx, cy))

            trail_pts = trails[track_id]
            n_trail = len(trail_pts)
            for i in range(1, n_trail):
                pt1, pt2 = trail_pts[i-1], trail_pts[i]
                fade = int(255 * (i / n_trail))
                seg_color = (
                    int(color[0] * (i / n_trail)),
                    int(color[1] * (i / n_trail)),
                    int(color[2] * (i / n_trail))
                )
                cv2.line(trail_layer, pt1, pt2, seg_color, 5)
                cv2.line(glow_layer, pt1, pt2, seg_color, 18)

            cyberpunk_box(frame, x1, y1, x2, y2, color, thickness=3, corner_len=28, glow=True)
            cv2.circle(frame, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)

            label = f"ID:{track_id} {class_name}"
            cyberpunk_text(frame, label, (x1, y1 - 16), color, scale=1.1, thickness=2)
            cyberpunk_text(frame, f"{speed_kmph:.1f} km/h", (x1, y1 - 2), (0, 255, 255), scale=1, thickness=2)

            if cy > line_y_red and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

        y_dbg = frame_height - 110
        for cname, dcount in debug_counts.items():
            cyberpunk_text(frame, f"Deteksi {cname}: {dcount}", (10, y_dbg), (0, 255, 255), 1, 2)
            y_dbg += 24

    y_offset = 30
    for class_name, count in class_counts.items():
        cyberpunk_text(frame, f"{class_name}: {count}", (50, y_offset), (0, 255, 255), 1.2, 2)
        y_offset += 28

    # ----------- LIVE CONSOLE COUNTING (Rapi di Windows) ------------
    if time.time() - last_console_refresh > refresh_interval:
        os.system('cls' if os.name == 'nt' else 'clear')
        show_ascii_header()
        print_env_info()
        print('\033[92m' + '[COUNTING] ' + datetime.now().strftime("%H:%M:%S") + '\033[0m')
        total_count = sum(class_counts.values())
        for class_name, count in class_counts.items():
            print(f"  {class_name:<10}: {count:>5}")
        print(f"  {'TOTAL':<10}: {total_count:>5}")
        print('\033[95m' + '-'*32 + '\033[0m')
        last_console_refresh = time.time()

    glow_blur = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=12, sigmaY=12)
    cv2.addWeighted(glow_blur, 0.45, frame, 1, 0, frame)
    cv2.addWeighted(trail_layer, 0.9, frame, 1, 0, frame)

    out.write(frame)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_path}")
