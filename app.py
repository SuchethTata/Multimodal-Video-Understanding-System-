import streamlit as st
import os
import tempfile
import re
import cv2
import ffmpeg
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# MODEL LOADING
# -----------------------------

EMBED_MODEL = "all-MiniLM-L6-v2"

similarity_model = SentenceTransformer(EMBED_MODEL)

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

whisper_model = whisper.load_model("base")


# -----------------------------
# SCENE DETECTION
# -----------------------------

def detect_scenes(video_path):

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    scene_manager.add_detector(ContentDetector())

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    scenes = []

    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append((start, end))

    return scenes


# -----------------------------
# FRAME EXTRACTION
# -----------------------------

def extract_scene_frames(video_path, scenes, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    for i, (start, end) in enumerate(scenes):

        frame_no = int(start * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()

        if ret:

            frame_path = os.path.join(output_folder, f"scene_{i}.jpg")

            cv2.imwrite(frame_path, frame)

            frames.append((frame_path, start))

    cap.release()

    return frames


# -----------------------------
# AUDIO EXTRACTION
# -----------------------------

def extract_audio(video_path, audio_path):

    (
        ffmpeg
        .input(video_path)
        .output(audio_path)
        .run(overwrite_output=True)
    )


# -----------------------------
# SPEECH TRANSCRIPTION
# -----------------------------

def transcribe_audio(audio_path):

    result = whisper_model.transcribe(audio_path)

    segments = []

    for seg in result["segments"]:
        segments.append((seg["start"], seg["end"], seg["text"].strip()))

    return segments


# -----------------------------
# IMAGE CAPTIONING
# -----------------------------

def caption_frames(frames):

    captions = []

    for img_path, timestamp in frames:

        image = Image.open(img_path).convert("RGB")

        inputs = caption_processor(image, return_tensors="pt")

        output = caption_model.generate(**inputs)

        caption = caption_processor.decode(output[0], skip_special_tokens=True)

        captions.append((timestamp, caption))

    return captions


# -----------------------------
# TIMELINE + PARAGRAPH
# -----------------------------

def generate_timeline(scenes, captions, speech):

    timeline = "Audio-Visual Event Description\n\n"

    paragraph_parts = []

    current_visual = None

    speech_buffer = []

    for i in range(len(scenes)):

        start, end = scenes[i]

        _, caption = captions[i]

        speech_parts = []

        for s_start, s_end, text in speech:

            if s_start >= start and s_start < end:
                speech_parts.append(text)

        speech_combined = " ".join(speech_parts).strip()

        # Timeline output

        if speech_combined:

            timeline += (
                f"[{round(start,2)}–{round(end,2)}s]\n"
                f"{caption} while saying:\n"
                f"\"{speech_combined}\"\n\n"
            )

        else:

            timeline += (
                f"[{round(start,2)}–{round(end,2)}s]\n"
                f"{caption}\n\n"
            )

        # Paragraph generation (internal use only)

        if caption != current_visual:

            if speech_buffer:

                paragraph_parts.append(" ".join(speech_buffer))

                speech_buffer = []

            current_visual = caption

            visual_sentence = caption.capitalize() + "."

            speech_buffer.append(visual_sentence)

        if speech_combined:

            speech_buffer.append(speech_combined)

    if speech_buffer:

        paragraph_parts.append(" ".join(speech_buffer))

    paragraph = "\n\n".join(paragraph_parts)

    paragraph = " ".join(paragraph.split())

    paragraph = re.sub(r'([.!?])([A-Z])', r'\1 \2', paragraph)

    return timeline, paragraph


# -----------------------------
# SEMANTIC SIMILARITY
# -----------------------------

def compute_similarity(text1, text2):

    emb1 = similarity_model.encode(text1, convert_to_tensor=True)

    emb2 = similarity_model.encode(text2, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)

    return float(score)


# -----------------------------
# BRAILLE CONVERSION
# -----------------------------

def text_to_braille(text):

    braille_map = {
        "a":"⠁","b":"⠃","c":"⠉","d":"⠙","e":"⠑","f":"⠋",
        "g":"⠛","h":"⠓","i":"⠊","j":"⠚","k":"⠅","l":"⠇",
        "m":"⠍","n":"⠝","o":"⠕","p":"⠏","q":"⠟","r":"⠗",
        "s":"⠎","t":"⠞","u":"⠥","v":"⠧","w":"⠺","x":"⠭",
        "y":"⠽","z":"⠵"," ":" ",
        ".":"⠲",",":"⠂","?":"⠦","!":"⠖"
    }

    braille_text = ""

    for char in text.lower():
        braille_text += braille_map.get(char, "")

    return braille_text


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Video Understanding System", layout="wide")

st.title("Multimodal Video Understanding System")

st.write(
    "Upload a video and optionally provide a reference description "
    "to evaluate semantic similarity and generate Braille output."
)

uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

reference_text = st.text_area("Reference Description (optional)")


if st.button("Run Analysis"):

    if uploaded_video is None:

        st.error("Please upload a video")

    else:

        with st.spinner("Processing video..."):

            temp_dir = tempfile.mkdtemp()

            video_path = os.path.join(temp_dir, uploaded_video.name)

            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            frames_dir = os.path.join(temp_dir, "frames")

            audio_path = os.path.join(temp_dir, "audio.wav")

            scenes = detect_scenes(video_path)

            frames = extract_scene_frames(video_path, scenes, frames_dir)

            extract_audio(video_path, audio_path)

            speech = transcribe_audio(audio_path)

            captions = caption_frames(frames)

            timeline, paragraph = generate_timeline(
                scenes,
                captions,
                speech
            )

        st.success("Processing Complete")

        # Timeline Output
        st.subheader("Timeline Output")
        st.text(timeline)

        # Semantic Similarity
        if reference_text.strip():

            score = compute_similarity(paragraph, reference_text)

            st.subheader("Semantic Similarity Score")

            st.metric("Similarity", round(score,3))

        # Braille Output
        braille_output = text_to_braille(paragraph)

        st.subheader("Braille Translation")

        st.code(braille_output)

        st.download_button(
            label="Download Braille Output",
            data=braille_output,
            file_name="braille_translation.txt",
            mime="text/plain"
        )