import scenedetect
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import write_scene_list

# Initialize Moondream2 model
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_key_frames(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_hist = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and compute histogram
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # If this is the first frame, just save it
        if prev_hist is None:
            prev_hist = hist
            frames.append(frame)
        else:
            # Compute the correlation between histograms of the current and previous frames
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            # If correlation is below the threshold, consider it a key frame
            if correlation < threshold:
                frames.append(frame)
                prev_hist = hist

        frame_count += 1

    cap.release()
    return frames

def analyze_frame(frame, model, tokenizer):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image
    enc_image = model.encode_image(image)
    output = model.answer_question(enc_image, "Describe this image.", tokenizer)
    return output

def summarize_video_descriptions(descriptions):
    combined_text = " ".join(descriptions)
    
    # Using the summarizer pipeline to summarize the combined text
    summary = summarizer(combined_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    
    return summary

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
        overall_summary = summarize_video_descriptions(summaries)
        st.write(overall_summary)

if __name__ == "__main__":
    main()
