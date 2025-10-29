# AI Vehicle Counter

## Overview

**AI Vehicle Counter** is a web-based application built with Streamlit that automatically detects and counts vehicles in traffic videos. It supports real-time detection, directional tracking, and generates class-wise counts for cars, trucks, buses, and motorbikes.

Users can upload MP4, AVI, or MOV videos, view a processed video with bounding boxes and counts, and download the output. The system is optimized for performance using the YOLOv12 ObjectCounter model with GPU acceleration when available.

## Approach

1. Video Upload & Preprocessing
    - Users upload traffic videos in supported formats.
    - Uploaded videos are temporarily stored for processing.
    - Video properties (resolution, FPS) are retrieved to configure output.

2. Vehicle Detection & Counting
    - The YOLOv12 ObjectCounter detects vehicles frame by frame.
    - Vehicles are tracked and counted when crossing a defined virtual line or region.
    - Class-wise counts are generated, showing vehicles moving IN or OUT across the defined region.

3. Output Generation
    - Processed video displays bounding boxes, tracking markers, and directional counts.
    - A textual summary of class-wise and total vehicle counts is generated.
    - Users can download the annotated video for further use.

4. Caching & Resource Management
    - The YOLOv12 model is cached using `@st.cache_resource` to avoid repeated loading.
    - GPU is used automatically if available for faster inference.
    - Temporary files are cleaned up after processing to save disk space.

## Technologies & Tools Used

- **Python 3.12**
- **Streamlit:** Web interface for video upload, display, and download
- **Ultralytics YOLOv12 ObjectCounter:** High-performance model for vehicle detection and counting
- **PyTorch:** Backend for model inference (CPU/GPU)
- **OpenCV (cv2):** Video processing and annotation
- **Tempfile & OS:** Temporary file management

## Steps to Reproduce

1. Clone the repository

    ```bash
    git clone https://github.com/dixon2004/vehicle-counter.git
    cd vehicle-counter
    ```

2.	Create and activate a virtual environment

    ```bash
    python -m venv .venv
    source .venv/bin/activate      # macOS / Linux
    .venv\Scripts\activate         # Windows
    ```

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app

    ```bash
    streamlit run app.py
    ```

5. Upload a video
	- Supported formats: MP4, AVI, MOV
	- Monitor processing progress via the on-screen progress bar.
	- View the processed video with annotated counts.
	- Download the annotated output.

## Notes & Recommendations

- GPU usage is automatic if available, significantly improving processing speed.
- Processing time depends on video resolution, duration, and system resources.
- Vehicle classes can be customized in load_counter() for different applications.
- Temporary files are automatically removed after processing.
- The system is suitable for real-time traffic monitoring and analytics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.