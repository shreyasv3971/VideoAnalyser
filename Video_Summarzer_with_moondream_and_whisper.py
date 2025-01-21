import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
import moviepy.editor as mp
import whisper
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Ensure GPU is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Moondream2 model
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize Whisper model for audio transcription
whisper_model = whisper.load_model("base")  # Replace 'base' with the appropriate model size if needed

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
    description = model.answer_question(enc_image, "Describe this image.", tokenizer)
    
    return description

def transcribe_audio(video_path):
    # Extract audio from video
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    # Use Whisper model to transcribe audio
    result = whisper_model.transcribe(audio_path)
    
    # Clean up temporary files
    os.remove(audio_path)

    return result['text']
def summarize_video_with_llama(summaries, transcript):
    # Combine all summaries into one text block
    combined_summary = "\n".join(summaries)
    
    # Create a detailed prompt for Llama3
    prompt = (
        "You have been provided with descriptions of key frames from a video and a transcript of the video's audio. "
        "Please provide a coherent and concise summary of these descriptions and the transcript in a structured paragraph format. "
        f"Here are the descriptions: {combined_summary}\n\n"
        f"Here is the audio transcript: {transcript}"
    )
    
    # Make a request to Llama3
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
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(video_path)
        
        st.write("Extracting key frames based on scene changes...")
        key_frames = extract_key_frames(video_path)
        st.write(f"Extracted {len(key_frames)} key frames")

        st.write("Analyzing frames with Moondream2...")
        summaries = []
        for idx, frame in enumerate(key_frames):
            summary = analyze_frame(frame, model, tokenizer)
            summaries.append(summary)
            st.image(frame, caption=f"Frame {idx+1} - Summary: {summary}")

        st.write("Transcribing audio from video...")
        transcript = transcribe_audio(video_path)

        st.write("Generating final video summary with Llama3...")
        final_summary = summarize_video_with_llama(summaries, transcript)
        st.write("Final Summary:")
        st.write(final_summary)

if __name__ == "__main__":
    main()
