```markdown
# Multimodal Video Understanding System

A **Multimodal AI system** that analyzes videos using **visual frames and audio speech** to generate structured understanding of video content.

The system performs:

- Scene detection
- Frame extraction
- Image captioning
- Speech transcription
- Audio-visual timeline generation
- Paragraph narration generation
- Semantic similarity evaluation
- Braille translation for accessibility

It also includes a **Streamlit web application** for interactive video analysis.

---

# Features

## 1. Scene Detection
Detects scene boundaries in videos using **PySceneDetect**.

## 2. Frame Extraction
Extracts representative frames from each detected scene using **OpenCV**.

## 3. Image Captioning
Uses **BLIP (Bootstrapped Language Image Pretraining)** to generate captions describing visual content.

Model used:

```

Salesforce/blip-image-captioning-base

```

---

## 4. Speech Recognition
Extracts audio from video and converts speech to text using **OpenAI Whisper**.

Model used:

```

Whisper Base

```

---

## 5. Audio-Visual Timeline Generation
Combines visual captions and speech segments to generate a **timeline description of the video**.

Example:

```

[0.0–4.5s]
A man standing in a kitchen while saying:
"Today I will show you how to cook pasta"

```

---

## 6. Narrative Paragraph Generation
Creates a **continuous paragraph summary** describing the video.

Example:

```

A man standing in a kitchen is shown. Today I will show you how to cook pasta.
He picks up a pan and places it on the stove.

```

---

## 7. Semantic Similarity Evaluation
Compares the generated paragraph with a **reference description** using sentence embeddings.

Model used:

```

all-MiniLM-L6-v2

```

---

## 8. Braille Translation
Converts the final narration into **Braille characters** for visually impaired users.

Example:

```

⠁ ⠍⠁⠝ ⠊⠎ ⠎⠞⠁⠝⠙⠊⠝⠛

```

---

# Project Structure

```

project/
│
├── app.py                # Streamlit web application
├── infer.py              # Command-line video understanding pipeline
│
├── data/
│   ├── input_videos/     # Input videos
│   ├── frames/           # Extracted frames
│   └── audio/            # Extracted audio
│
├── outputs/
│   └── narrations/
│       ├── timeline.txt
│       └── paragraph.txt
│
└── README.md

````

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-understanding-system.git
cd video-understanding-system
````

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```
venv\Scripts\activate
```

### Linux / Mac

```
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install streamlit
pip install opencv-python
pip install pillow
pip install ffmpeg-python
pip install scenedetect
pip install transformers
pip install sentence-transformers
pip install openai-whisper
pip install torch
```

---

## 4. Install FFmpeg

FFmpeg is required for audio extraction.

Download from:

```
https://ffmpeg.org/download.html
```

Verify installation:

```bash
ffmpeg -version
```

---

# Usage

## 1. Command Line Execution

Run the video understanding pipeline:

```bash
python infer.py
```

The pipeline will:

1. Detect scenes
2. Extract frames
3. Extract audio
4. Transcribe speech
5. Generate image captions
6. Create audio-visual timeline
7. Generate paragraph narration
8. Produce Braille translation

Outputs will be saved in:

```
outputs/narrations/
```

Generated files:

```
timeline.txt
paragraph.txt
```

---

## 2. Run the Streamlit Web Application

Launch the UI:

```bash
streamlit run app.py
```

Open the browser at:

```
http://localhost:8501
```

Steps:

1. Upload a video
2. Optionally enter a reference description
3. Click **Run Analysis**

The system will display:

* Timeline description
* Semantic similarity score
* Braille translation

You can also **download the Braille output**.

---

# Models Used

| Task               | Model             |
| ------------------ | ----------------- |
| Image Captioning   | BLIP (Salesforce) |
| Speech Recognition | Whisper Base      |
| Text Embeddings    | all-MiniLM-L6-v2  |

---

# Technologies Used

* Python
* OpenCV
* Whisper
* Transformers
* BLIP
* Sentence Transformers
* PySceneDetect
* FFmpeg
* Streamlit

---

# Accessibility Goal

This system helps make **video content accessible for visually impaired users** by:

* Converting video into descriptive narration
* Translating descriptions into **Braille**

---

# Future Improvements

Possible enhancements include:

* Multi-frame captioning per scene
* Real-time video understanding
* GPU acceleration
* Multi-language support
* Advanced Braille formatting
* LLM-based scene summarization

---


```
