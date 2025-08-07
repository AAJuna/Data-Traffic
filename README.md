# 🚦 Data Traffic - Vehicle Volume Counter from Video

**Data Traffic** is a Python-based application that analyzes traffic footage to detect, track, and count vehicles. It helps calculate traffic volume statistics in real-time or from pre-recorded videos. Ideal for smart traffic monitoring, urban planning, or transportation studies.

---

## 🎯 Features

- 🚗 Vehicle detection using object detection (YOLOv8 or compatible models)
- 🧠 Multi-object tracking (e.g., DeepSORT) to avoid double counting
- 📈 Real-time and batch processing support
- 📁 Export results to CSV for further analysis
- 🎥 Supports various video input formats (MP4, AVI, etc.)

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8 (or YOLOv5)
- DeepSORT (for tracking)
- Pandas

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
    python vehicle_counter.py --input videos/your_video.mp4
    ```

4. **Check output** in the `output/` directory.

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
