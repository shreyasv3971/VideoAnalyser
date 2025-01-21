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

def extract_key_frames(cap, interval=300, max_frames=10):
    frames = []
    frame_count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0 or frame_count == 0:
            resized_frame = cv2.resize(frame, (620, 620))
            frames.append(resized_frame)
        frame_count += 1
    return frames

def analyze_frame(frame, model, tokenizer):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enc_image = model.encode_image(image).to(device)
    output = model.answer_question(enc_image, "Describe this image.", tokenizer)
    return output

def main():
    st.title("Webcam Video Analysis with Moondream2")
    st.write("Press 'Start' to capture and analyze frames from your webcam")

    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
            return

        st.write("Extracting key frames...")
        key_frames = extract_key_frames(cap, interval=30, max_frames=5)
        cap.release()
        st.write(f"Extracted {len(key_frames)} key frames")

        st.write("Analyzing frames with Moondream2...")
        summaries = []
        for idx, frame in enumerate(key_frames):
            summary = analyze_frame(frame, model, tokenizer)
            summaries.append(summary)
            st.image(frame, caption=f"Frame {idx+1} - Summary: {summary}")

        st.write("Video Summary:")
        st.write("\n".join(summaries))

if __name__ == "_main_":
    main()