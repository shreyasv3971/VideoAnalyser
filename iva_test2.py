import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Moondream2 model
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def extract_key_frames(video_path, interval=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0 or frame_count == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def analyze_frame(frame, model, tokenizer):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enc_image = model.encode_image(image).to(device)
    output = model.answer_question(enc_image, "Describe this image.", tokenizer)
    return output

def main():
    st.title("Video Analysis with Moondream2")
    st.write("Upload a video to extract key frames and analyze them using Moondream2 model")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video("temp_video.mp4")
        
        st.write("Extracting key frames...")
        key_frames = extract_key_frames("temp_video.mp4")
        st.write(f"Extracted {len(key_frames)} key frames")

        st.write("Analyzing frames with Moondream2...")
        summaries = []
        for idx, frame in enumerate(key_frames):
            summary = analyze_frame(frame, model, tokenizer)
            summaries.append(summary)
            st.image(frame, caption=f"Frame {idx+1} - Summary: {summary}")

        st.write("Video Summary:")
        st.write("\n".join(summaries))

if __name__ == "__main__":
    main()
