import gradio as gr
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os

# --- Setup and Model Loading ---

# Download NLTK stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load pre-trained AI models from Hugging Face
print("Loading AI models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
print("Models loaded successfully.")

# --- Dictionaries and Mappings ---

# Supported languages for translation
TRANSLATION_LANGUAGES = {
    "English": "en", "French": "fr", "Spanish": "es", "German": "de",
    "Telugu": "te", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
    "Chinese": "zh", "Japanese": "ja", "Arabic": "ar"
}

# Emojis corresponding to detected emotions
EMOJI_MAPPING = {
    "joy": "😃", "anger": "😡", "sadness": "😢", "fear": "😨", "love": "😍",
    "surprise": "😲", "disgust": "🤢", "neutral": "😐"
}

# A small dictionary of common words and their ASL GIF representations
ASL_DICTIONARY = {
    "hello": "https://www.lifeprint.com/asl101/gifs/h/hello.gif",
    "thank": "https://www.lifeprint.com/asl101/gifs/t/thank-you.gif",
    "good": "https://www.lifeprint.com/asl101/gifs/g/good.gif",
    "morning": "https://www.lifeprint.com/asl101/gifs/m/morning.gif",
    "sorry": "https://www.lifeprint.com/asl101/gifs/s/sorry.gif",
    "help": "https://www.lifeprint.com/asl101/gifs/h/help.gif",
    "love": "https://www.lifeprint.com/asl101/gifs/l/love.gif",
    "happy": "https://www.lifeprint.com/asl101/gifs/h/happy.gif",
    "sad": "https://www.lifeprint.com/asl101/gifs/s/sad.gif"
}

# ASL finger-spelling GIFs for each letter of the alphabet
ASL_ALPHABET_IMAGES = {letter: f"https://www.lifeprint.com/asl101/fingerspelling/{letter}.gif" for letter in "abcdefghijklmnopqrstuvwxyz"}


# --- Core Functions ---

def extract_keywords(text, num_keywords=5):
    """Extracts the most common words from text, excluding stopwords."""
    words = [word.lower() for word in text.split() if word.lower() not in stopwords.words('english') and word.isalpha()]
    common_words = Counter(words).most_common(num_keywords)
    return [word[0] for word in common_words]

def get_asl_representation(text):
    """Generates an HTML string of ASL GIFs for the given text."""
    words = text.lower().split()
    images_html = "<div style='display: flex; overflow-x: auto; white-space: nowrap; padding: 10px; border: 1px solid #ddd; border-radius: 8px;'>"
    for word in words:
        clean_word = ''.join(filter(str.isalpha, word))
        if clean_word in ASL_DICTIONARY:
            images_html += f"<img src='{ASL_DICTIONARY[clean_word]}' alt='{clean_word}' style='height: 60px; margin-right: 5px; border-radius: 4px;'>"
        else:
            for char in clean_word:
                if char in ASL_ALPHABET_IMAGES:
                    images_html += f"<img src='{ASL_ALPHABET_IMAGES[char]}' alt='{char}' style='height: 60px; margin-right: 2px;'>"
    images_html += "</div>"
    return images_html

def translate_text(text, target_language):
    """Translates text to the specified target language."""
    try:
        lang_code = TRANSLATION_LANGUAGES.get(target_language)
        if lang_code:
            return GoogleTranslator(source="auto", target=lang_code).translate(text)
        return "Translation not available for this language."
    except Exception as e:
        print(f"Translation Error: {e}")
        return "Translation failed."

def text_to_speech(text, target_language):
    """Converts text to an MP3 audio file."""
    lang_code = TRANSLATION_LANGUAGES.get(target_language, "en")
    try:
        tts = gTTS(text, lang=lang_code, slow=False)
        audio_file = "output_audio.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def process_text_and_stream_outputs(text, target_language, min_words, max_words):
    """
    A generator function that processes text and yields results as they become available,
    allowing the Gradio UI to update incrementally.
    """
    if not text.strip():
        yield {
            summary_output: "Please enter some text to analyze.",
            audio_output: None,
            translated_output: "",
            hashtags_output: "",
            asl_output: ""
        }
        return

    # --- Step 1: Summarization and Emotion Detection (CPU-intensive) ---
    print("Step 1: Summarizing text...")
    summary = summarizer(text, max_length=int(max_words), min_length=int(min_words), do_sample=False)[0]['summary_text']
    emotion_result = emotion_detector(summary)[0]
    emotion = emotion_result['label']
    emoji_display = EMOJI_MAPPING.get(emotion, "😐")
    summary_with_emotion = f"{summary} \n\nEmotion: {emotion} {emoji_display}"
    
    # Yield the first result
    yield {summary_output: summary_with_emotion}

    # --- Step 2: Translation (Network I/O) ---
    print("Step 2: Translating summary...")
    translated_summary = translate_text(summary, target_language)
    yield {translated_output: translated_summary}

    # --- Step 3: Text-to-Speech (Network I/O) ---
    print("Step 3: Generating audio...")
    audio_path = text_to_speech(translated_summary, target_language)
    yield {audio_output: audio_path}
    
    # --- Step 4: Hashtags (Fast, local processing) ---
    print("Step 4: Generating hashtags...")
    keywords = extract_keywords(summary)
    hashtags = "#" + " #".join(keywords)
    yield {hashtags_output: hashtags}

    # --- Step 5: ASL Representation (Multiple Network I/O) ---
    print("Step 5: Generating ASL representation...")
    asl_html = get_asl_representation(summary)
    yield {asl_output: asl_html}
    
    print("All steps complete.")


# --- Gradio Interface using Blocks for Streaming ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Universal Text Analyzer 📝🔊🤟
        Summarize text, detect emotions, translate, generate audio, create hashtags, and see it in American Sign Language.
        Results will appear as they are generated.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(lines=10, placeholder="Enter a long text here...", label="Your Text")
            target_language_input = gr.Dropdown(choices=list(TRANSLATION_LANGUAGES.keys()), value="English", label="Translate to")
            
            with gr.Row():
                min_words_slider = gr.Slider(minimum=20, maximum=100, value=30, step=1, label="Min Summary Words")
                max_words_slider = gr.Slider(minimum=50, maximum=500, value=150, step=1, label="Max Summary Words")
            
            submit_button = gr.Button("Analyze Text", variant="primary")

        with gr.Column(scale=3):
            summary_output = gr.Textbox(label="Summary & Emotion", interactive=False)
            translated_output = gr.Textbox(label="Translated Summary", interactive=False)
            hashtags_output = gr.Textbox(label="Generated Hashtags", interactive=False)
            audio_output = gr.Audio(label="Audio of Translated Summary")
            asl_output = gr.HTML(label="ASL Representation")

    # Connect the inputs and outputs to the streaming function
    submit_button.click(
        fn=process_text_and_stream_outputs,
        inputs=[text_input, target_language_input, min_words_slider, max_words_slider],
        outputs=[summary_output, audio_output, translated_output, hashtags_output, asl_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)