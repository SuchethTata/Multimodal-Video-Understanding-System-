import os
import re
import cv2
import ffmpeg
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from sentence_transformers import SentenceTransformer, util


# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------

VIDEO_PATH = r"C:\Users\suche\Desktop\operators\data\input_videos\videoplayback.mp4"

FRAMES_DIR = r"C:\Users\suche\Desktop\operators\data\frames"
AUDIO_PATH = r"C:\Users\suche\Desktop\operators\data\audio\audio.wav"

TIMELINE_OUTPUT = r"C:\Users\suche\Desktop\operators\outputs\narrations\timeline.txt"
PARAGRAPH_OUTPUT = r"C:\Users\suche\Desktop\operators\outputs\narrations\paragraph.txt"

EMBED_MODEL = "all-MiniLM-L6-v2"


# ------------------------------------------------
# SCENE DETECTION
# ------------------------------------------------

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


# ------------------------------------------------
# FRAME EXTRACTION
# ------------------------------------------------

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


# ------------------------------------------------
# AUDIO EXTRACTION
# ------------------------------------------------

def extract_audio(video_path, output_audio):
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(output_audio)
        .run(overwrite_output=True)
    )


# ------------------------------------------------
# WHISPER TRANSCRIPTION
# ------------------------------------------------

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = []
    for seg in result["segments"]:
        segments.append((seg["start"], seg["end"], seg["text"].strip()))
    return segments


# ------------------------------------------------
# IMAGE CAPTIONING
# ------------------------------------------------

def caption_frames(frames):
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    captions = []
    for img_path, timestamp in frames:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append((timestamp, caption))
    return captions


# ------------------------------------------------
# TIMELINE GENERATION (FIXED)
# ------------------------------------------------

def generate_timeline(scenes, captions, speech):
    timeline = "Audio-Visual Event Description\n\n"
    
    # For paragraph generation
    narrative_parts = []
    current_visual = None
    current_speech_segments = []
    
    for i in range(len(scenes)):
        start, end = scenes[i]
        _, caption = captions[i]
        
        # Collect speech for this scene
        speech_parts = []
        for s_start, s_end, text in speech:
            if s_start >= start and s_start < end:
                speech_parts.append(text)
        speech_combined = " ".join(speech_parts).strip()
        
        # -------- TIMELINE --------
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
        
        # -------- PARAGRAPH CREATION --------
        
        # Check if visual context changed
        is_new_visual = (current_visual is None or caption != current_visual)
        
        if is_new_visual:
            # Save previous visual's speech if exists
            if current_visual is not None and current_speech_segments:
                narrative_parts.append(" ".join(current_speech_segments))
            
            # Start new visual context
            current_visual = caption
            
            # Clean and format visual description
            visual_desc = caption.strip()
            
            # Create natural visual introduction
            if "photo" in visual_desc.lower() or "picture" in visual_desc.lower():
                intro = f"{visual_desc} appears"
            else:
                # Remove leading 'a ' or 'an ' for flow
                if visual_desc.lower().startswith("a "):
                    visual_desc = visual_desc[2:]
                elif visual_desc.lower().startswith("an "):
                    visual_desc = visual_desc[3:]
                # Capitalize first letter
                if visual_desc:
                    visual_desc = visual_desc[0].upper() + visual_desc[1:]
                intro = f"{visual_desc} is shown"
            
            # Start new speech segment with visual intro
            if speech_combined:
                current_speech_segments = [f"{intro}. {speech_combined}"]
            else:
                current_speech_segments = [f"{intro}."]
        else:
            # Same visual - just add speech
            if speech_combined:
                current_speech_segments.append(speech_combined)
    
    # Don't forget last segment
    if current_visual is not None and current_speech_segments:
        narrative_parts.append(" ".join(current_speech_segments))
    
    # Join with paragraph breaks between different visuals
    paragraph = "\n\n".join(narrative_parts)
    
    # Final cleanup
    if paragraph:
        # Fix spacing
        paragraph = " ".join(paragraph.split())
        # Ensure proper sentence spacing
        paragraph = re.sub(r'([.!?])([A-Z])', r'\1 \2', paragraph)
        # Capitalize first letter
        paragraph = paragraph[0].upper() + paragraph[1:]
    
    return timeline, paragraph


# ------------------------------------------------
# SEMANTIC SIMILARITY
# ------------------------------------------------

def compute_similarity(text1, text2):
    model = SentenceTransformer(EMBED_MODEL)
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score)


# ------------------------------------------------
# BRAILLE CONVERSION
# ------------------------------------------------

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


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def main():
    print("\nRunning Video Understanding Pipeline\n")
    
    scenes = detect_scenes(VIDEO_PATH)
    print(f"Detected {len(scenes)} scenes")
    
    frames = extract_scene_frames(VIDEO_PATH, scenes, FRAMES_DIR)
    print(f"Extracted {len(frames)} frames")
    
    extract_audio(VIDEO_PATH, AUDIO_PATH)
    print("Audio extracted")
    
    speech = transcribe_audio(AUDIO_PATH)
    print(f"Transcribed {len(speech)} speech segments")
    
    captions = caption_frames(frames)
    print(f"Generated {len(captions)} captions")
    
    timeline, paragraph = generate_timeline(scenes, captions, speech)
    
    print("\n" + "="*50)
    print("TIMELINE OUTPUT:")
    print("="*50)
    print(timeline)
    
    print("\n" + "="*50)
    print("PARAGRAPH SUMMARY:")
    print("="*50)
    print(paragraph)
    print("="*50)
    
    # Save outputs
    os.makedirs(os.path.dirname(TIMELINE_OUTPUT), exist_ok=True)
    with open(TIMELINE_OUTPUT, "w", encoding="utf-8") as f:
        f.write(timeline)
    
    with open(PARAGRAPH_OUTPUT, "w", encoding="utf-8") as f:
        f.write(paragraph)
    
    print(f"\nSaved to:\n{TIMELINE_OUTPUT}\n{PARAGRAPH_OUTPUT}")
    
    # Optional similarity check
    reference = input("\nEnter reference description (optional): ")
    if reference.strip():
        score = compute_similarity(paragraph, reference)
        print(f"\nSemantic Similarity Score: {round(score, 3)}")
    
    # Braille output
    braille = text_to_braille(paragraph)
    print("\nBraille Translation:\n")
    print(braille)
    
    print("\nPipeline Finished")


if __name__ == "__main__":
    main()