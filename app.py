from ultralytics import solutions
import streamlit as st
import tempfile
import torch
import cv2
import os


def load_counter() -> solutions.ObjectCounter:
    """
    Load the pre-trained YOLOv12n model for vehicle counting.

    Returns:
        solutions.ObjectCounter: Configured object counter for vehicles.
    """
    return solutions.ObjectCounter(
        model="yolo12n.pt",
        classes=[2, 3, 5, 7],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def process_video(
        video_path: str,
        counter: solutions.ObjectCounter,
        output_name: str,
    ) -> tuple[str, dict]:
    """
    Process the input video to count vehicles and generate an output video with visualizations.

    Args:
        video_path (str): Path to the input video file.
        counter (solutions.ObjectCounter): Pre-configured object counter.
        output_name (str): Name for the output video file.

    Returns:
        tuple[str, dict]: Path to the output video and class-wise count results.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error reading video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    counter.region = [(w // 2, 0), (w // 2, h)]

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", output_name)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    progress_bar = st.progress(0, text="Processing video...")
    progress_text = st.empty()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = counter(frame)
        writer.write(result.plot_im)

        frame_idx += 1
        progress = min(frame_idx / total_frames, 1.0)
        progress_bar.progress(progress, text=f"Processing video... {int(progress * 100)}%")

    results = counter.classwise_count

    writer.release()
    cap.release()
    progress_bar.empty()
    progress_text.empty()

    if os.path.getsize(output_path) == 0:
        raise RuntimeError("Output video is empty. Codec may not be supported.")

    return output_path, results


def get_vehicle_summary_text(classwise_count: dict) -> str:
    """
    Generate a summary text of vehicle counts.

    Args:
        classwise_count (dict): Dictionary with vehicle class counts.

    Returns:
        str: Formatted summary text.
    """
    total = sum(v['IN'] + v['OUT'] for v in classwise_count.values())
    total_in = sum(v['IN'] for v in classwise_count.values())
    total_out = sum(v['OUT'] for v in classwise_count.values())

    text = f"**Total Vehicles:** {total} (IN: {total_in}, OUT: {total_out})\n\n"

    for cls, counts in classwise_count.items():
        cls_total = counts['IN'] + counts['OUT']
        text += f"- **{cls.capitalize()}**: {cls_total} (IN: {counts['IN']}, OUT: {counts['OUT']})\n"

    return text


def main() -> None:
    """
    Main function to run the Streamlit app for AI Vehicle Counter.
    """
    st.set_page_config(page_title="AI Vehicle Counter")
    st.title("🚗 AI Vehicle Counter")
    st.caption("An AI-powered system that automatically detects and counts vehicles crossing a virtual line in uploaded traffic videos.")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if not uploaded:
        st.info("Please upload a video file to begin")
        return

    filename = uploaded.name
    base_name, ext = os.path.splitext(filename)
    output_name = f"{base_name}_output{ext}"

    if "output_path" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        counter = load_counter()

        output_path, results = process_video(tmp_path, counter, output_name)
        st.session_state["output_path"] = output_path
        st.session_state["results"] = results

        os.unlink(tmp_path)

    st.success("All done! Your processed video is ready below.")
    st.video(st.session_state["output_path"])
    st.info(get_vehicle_summary_text(st.session_state["results"]))

    with open(st.session_state["output_path"], "rb") as f:
        st.download_button(
            "Download",
            f,
            file_name=os.path.basename(st.session_state["output_path"]),
        )


if __name__ == "__main__":
    main()