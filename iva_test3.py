import streamlit as st
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import torchvision
import torchvision.transforms as transforms
# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Moondream2 model
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def extract_key_frames(video_path, interval=300, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

from PIL import ImageOps

def summarize_video(frames, model, tokenizer):
    all_descriptions = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enc_image = model.encode_image(image)      
        description = model.answer_question(enc_image, "Describe this image.", tokenizer)
        all_descriptions.append(description)
    
    # Combine all descriptions into one summary
    combined_descriptions = " ".join(all_descriptions)
    final_summary = model(combined_descriptions).logits.argmax(-1)
    return tokenizer.decode(final_summary, skip_special_tokens=True)



def main():
    st.title("Video Summary with Moondream2")
    st.write("Upload a video to generate a summary using the Moondream2 model")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video("temp_video.mp4")
        
        st.write("Extracting key frames...")
        key_frames = extract_key_frames("temp_video.mp4")
        st.write(f"Extracted {len(key_frames)} key frames")
        
        st.write("Generating video summary...")
        video_summary = summarize_video(key_frames, model, tokenizer)
        st.write("Video Summary:")
        st.write(video_summary)
        
        # Optionally, save the summary to a text file
        with open("video_summary.txt", "w") as summary_file:
            summary_file.write(video_summary)
        st.write("Summary saved to video_summary.txt")

if __name__ == "__main__":
    main()
