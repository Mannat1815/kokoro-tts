import logging
import os
import time
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import generate_speech
import torch
from huggingface_hub import hf_hub_download
from retry import retry
from pathlib import Path
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")

logger = logging.getLogger(__name__)

# Global pipeline and voice cache
kokoro_pipeline = None
voice_cache = {}  # Maps voice_name to local voice file path
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_PATH = "kokoro-v1_0.pth"
VOICE_DIR = os.path.join(os.path.dirname(__file__), '..', 'Kokoro-82M', 'voices')

def initialize_kokoro_pipeline():
    """Initialize and cache the Kokoro pipeline."""
    global kokoro_pipeline
    if kokoro_pipeline is not None:
        logger.debug("Using cached Kokoro pipeline")
        return

    start_time = time.time()
    try:
        # Set espeak-ng data path
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng-data"
        os.environ["ESPEAK_DATA_PATH"] = espeak_path
        logger.debug(f"Set espeak-ng data path: {espeak_path}")

        # Download and cache model file
        local_model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_PATH, local_dir="models")
        logger.debug(f"Cached model at {local_model_path}")

        # Initialize pipeline
        from kokoro import KPipeline
        kokoro_pipeline = KPipeline(lang_code='a')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Initialized KPipeline (device managed internally, detected: {device})")

        logger.info(f"Kokoro pipeline initialized in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro pipeline: {str(e)}")
        kokoro_pipeline = None
        raise

@retry(tries=3, delay=1, backoff=2, logger=logger)
def generate_audio_for_sentence(text, voice, output_path):
    """Generate audio for a single sentence with retry logic."""
    global kokoro_pipeline
    start_time = time.time()
    if not kokoro_pipeline:
        initialize_kokoro_pipeline()

    if voice not in voice_cache:
        voice_file = f"{voice}.pt"
        local_voice_path = os.path.join(VOICE_DIR, voice_file)
        if not os.path.exists(local_voice_path):
            logger.debug(f"Downloading voice {voice}")
            hf_hub_download(repo_id=MODEL_REPO, filename=f"voices/{voice_file}", local_dir=VOICE_DIR)
        voice_cache[voice] = local_voice_path
        logger.debug(f"Cached voice {voice} at {local_voice_path}")

    try:
        audio_segments, chunk_timings, duration = generate_speech(text, voice, output_path, kokoro_pipeline)
        audio_url = f"/static/audio/{os.path.basename(output_path)}"
        logger.debug(f"Generated audio for '{text[:50]}...' with voice {voice} in {time.time() - start_time:.2f}s")
        return {"audio_url": audio_url, "duration": duration, "chunk_timings": chunk_timings, "voice": voice}
    except Exception as e:
        logger.error(f"Error generating audio for '{text[:50]}...' with voice {voice}: {str(e)}")
        raise

@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def generate(request):
    """Handle batch audio generation for multiple sentences with different voices."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    start_time = time.time()
    try:
        # Parse POST data
        texts = request.POST.getlist("text[]") or [request.POST.get("text")]
        voices = request.POST.getlist("voice[]") or [request.POST.get("voice", "af_heart")]
        logger.debug(f"Received POST data: texts={len(texts)}, voices={len(voices)}")

        if not texts or not any(text.strip() for text in texts):
            logger.warning("No valid text provided")
            return JsonResponse({"error": "No text provided"}, status=400)

        if len(voices) != len(texts):
            voices = voices + [voices[-1] if voices else "af_heart"] * (len(texts) - len(voices))
            logger.debug(f"Adjusted voices list to match texts: {voices}")

        # Initialize pipeline if not already done
        if kokoro_pipeline is None:
            initialize_kokoro_pipeline()

        # Generate audio for all sentences
        results = []
        output_dir = os.path.join(os.path.dirname(__file__), 'static', 'audio')
        os.makedirs(output_dir, exist_ok=True)
        for i, (text, voice) in enumerate(zip(texts, voices)):
            text = text.strip()
            if not text:
                continue
            output_path = os.path.join(output_dir, f"{voice}_{hash(text + str(time.time()))}.wav")
            logger.debug(f"Processing sentence {i+1}/{len(texts)}: '{text[:50]}...' with voice {voice}")
            result = generate_audio_for_sentence(text, voice, output_path)
            results.append(result)

        logger.info(f"Generated {len(results)} audio files in {time.time() - start_time:.2f}s")
        return JsonResponse({"audio_urls": results})
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def cleanup_audio(request):
    """Clean up audio files."""
    audio_dir = os.path.join(os.path.dirname(__file__), 'static', 'audio')
    try:
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                if file.endswith(".wav"):
                    os.remove(os.path.join(audio_dir, file))
        logger.debug("Audio folder cleaned up")
        return JsonResponse({"status": "success", "message": "Audio folder cleaned up"})
    except Exception as e:
        logger.error(f"Error cleaning up audio: {str(e)}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)