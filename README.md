# 🚦 Data Traffic - Vehicle Volume Counter from Video

**Data Traffic** is a Python-based application that analyzes traffic footage to detect, track, and count vehicles. It helps calculate traffic volume statistics in real-time or from pre-recorded videos. Ideal for smart traffic monitoring, urban planning, or transportation studies.

---

## 🎯 Features

- 🚗 Vehicle detection using object detection (YOLOv8 or compatible models)
- 🧠 Multi-object tracking (e.g., DeepSORT) to avoid double counting
- 📈 Real-time and batch processing support
- 📁 Export results to CSV for further analysis
- 🎥 Supports various video input formats (MP4, AVI, etc.)
- ⚙️ Command-line options for choosing model weights, tracker, and image size
- ⚡ Optional FP16 inference for faster GPU performance

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8 (or YOLOv5)
- DeepSORT (for tracking)
- Pandas
- Optional trackers: ByteTrack, BoT-SORT, or others via `--tracker`

---

## 📊 Output Example
| Timestamp           | Cars | Trucks | Motorcycles | Buses |
| ------------------- | ---- | ------ | ----------- | ----- |
| 2025-08-07 07:00:00 | 45   | 8      | 12          | 1     |


- Console log
- CSV file (`traffic_log.csv`)
- (Optional) Annotated video output with bounding boxes and counts

---

## 🚀 Getting Started

1. **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

2. **Download or place YOLOv8 model** in the `models/` folder.

3. **Run the main script:**

    ```
    python hitung.py --video your_video.mp4 --weights yolov8m.pt
    ```

4. **Check output** in the `output/` directory.

### 🔧 CLI Options

`hitung.py` now supports several flags to tune accuracy and speed:

| Option | Description |
| ------ | ----------- |
| `--weights` | Path to YOLO model (e.g., `yolov8x.pt` for higher accuracy) |
| `--tracker` | Tracking config such as `bytetrack.yaml` or `botsort.yaml` |
| `--imgsz` | Inference size, larger values (e.g., 1280) improve accuracy |
| `--conf` | Confidence threshold |
| `--device` | Select `cpu`, `cuda`, or specific GPU id |
| `--half` | Use FP16 for faster inference on supported GPUs |

---

## 🚀 Improving Accuracy & Speed

- **Use newer models**: try `yolov8x.pt`, `RT-DETR`, or `YOLO-NAS` for higher accuracy.
- **Hardware acceleration**: export models to ONNX/TensorRT or run with NVIDIA DeepStream for real-time throughput.
- **Advanced trackers**: experiment with `botsort.yaml` or `ocsort.yaml` for better ID management.
- **Parallel processing**: leverage multi-threaded video reading or GPU batch inference when analyzing multiple streams.

---

## 🧠 Use Cases

- Smart city traffic monitoring
- Traffic volume and congestion analysis
- Urban infrastructure planning
- Transportation research
- Traffic law enforcement analytics

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, modify, and contribute.

---

## 🙌 Contributions

Pull requests, feature suggestions, and issue reports are welcome.
Let’s build smarter traffic systems together 🚗📈
