import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Moondream2 model
model_id = "checkpoints/moondream-ft"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def extract_key_frames(video_path, threshold=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_hist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold:
                frames.append(frame)
        else:
            frames.append(frame)

        prev_hist = hist

    cap.release()
    return frames

def analyze_frame(frame, model, tokenizer):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Encode the image using the model
    enc_image = model.encode_image(image)
    
    # Generate a description for the image
    output = model.answer_question(enc_image, "Describe this image.", tokenizer)
    
    return output

def summarize_video_with_llama(summaries):
    # Combine all summaries into one text block
    combined_summary = "\n".join(summaries)
    prompt = (
        "You have been provided with descriptions of key frames from a video. "
        "Please provide a coherent and concise summary of these descriptions in a structured paragraph format. "
        f"Here are the descriptions: {combined_summary}"
    )
    
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    # Debugging: Print the response to understand its structure
    st.write("Llama3 Response:", response)

    # Extract the summary from the response
    if isinstance(response, dict) and 'message' in response:
        final_summary = response['message']['content']
    else:
        final_summary = "Summary not available. Please check the response format."

    return final_summary

def main():
    st.title("Video Analysis with Scene Change Detection, Moondream2, and Llama3")
    st.write("Upload a video to extract key frames based on scene changes, summarize them with Moondream2, and then generate a final summary using Llama3.")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video("temp_video.mp4")
        
        st.write("Extracting key frames based on scene changes...")
        key_frames = extract_key_frames("temp_video.mp4")
        st.write(f"Extracted {len(key_frames)} key frames")

        st.write("Analyzing frames with Moondream2...")
        summaries = []
        for idx, frame in enumerate(key_frames):
            summary = analyze_frame(frame, model, tokenizer)
            summaries.append(summary)
            st.image(frame, caption=f"Frame {idx+1} - Summary: {summary}")

        st.write("Generating final video summary with Llama3...")
        final_summary = summarize_video_with_llama(summaries)
        st.write("Final Summary:")
        st.write(final_summary)

if __name__ == "__main__":
    main()
