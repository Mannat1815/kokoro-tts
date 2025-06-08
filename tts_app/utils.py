import torch
import soundfile as sf
import os
import misaki.espeak as espeak
import logging
from huggingface_hub import hf_hub_download
import re

logger = logging.getLogger(__name__)

def split_sentences(text):
    """Split text into sentences, matching index.html logic."""
    # Split on: spaces after [.!?], double spaces after word, or newlines
    pattern = r'(?<=[.!?])\s+|(?<=\w)\s{2,}|\n+'
    sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
    return sentences if sentences else [text.strip()]

def generate_speech(text, voice, output_path, pipeline):
    """Generate speech for a single sentence using a cached pipeline."""
    # Set espeak-ng data path
    espeak_data_path = r"C:\Program Files\eSpeak NG\espeak-ng-data"
    if not os.path.exists(espeak_data_path):
        logger.error(f"espeak-ng data path not found: {espeak_data_path}")
        raise FileNotFoundError(f"espeak-ng data path not found at {espeak_data_path}")

    try:
        espeak.EspeakWrapper.data_path = espeak_data_path
        logger.debug(f"Successfully set espeak-ng data path: {espeak_data_path}")
    except Exception as e:
        logger.error(f"Failed to set espeak-ng data path: {str(e)}")
        raise

    try:
        local_voice_path = os.path.join(os.path.dirname(__file__), '..', 'Kokoro-82M', 'voices', f"{voice}.pt")
        if not os.path.exists(local_voice_path):
            logger.warning(f"Voice file {voice}.pt not found locally, downloading")
            hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice}.pt", local_dir=os.path.dirname(local_voice_path))

        # Split into sentences
        sentences = split_sentences(text)
        audio_segments = []
        chunk_timings = []
        current_time = 0.0

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                generator = pipeline(sentence, voice=voice)
                for _, _, audio in generator:
                    logger.debug(f"Generated audio tensor for sentence {i+1}/{len(sentences)} on device: {audio.device}")
                    audio_segments.append(audio)
                    chunk_duration = audio.shape[0] / 24000  # 24kHz sample rate
                    chunk_timings.append({
                        'start': current_time,
                        'end': current_time + chunk_duration
                    })
                    current_time += chunk_duration
                    logger.debug(f"Processed sentence {i+1}/{len(sentences)}: '{sentence}'")
            except Exception as e:
                logger.warning(f"Skipping sentence due to error: '{sentence}' — {e}")
                continue  # Skip to next sentence instead of raising

        # Save full audio
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=0).cpu().numpy()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, full_audio, 24000)
            logger.info(f"Full audio written to {output_path}")
        else:
            raise RuntimeError("No audio segments generated")

        # Confirm actual audio duration
        with sf.SoundFile(output_path) as f:
            actual_duration = f.frames / f.samplerate
        logger.debug(f"Final audio duration: {actual_duration:.2f}s")

        return audio_segments, chunk_timings, actual_duration

    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        raise