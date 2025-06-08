FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for espeak-ng (required for TTS phoneme generation)
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including Kokoro-82M and media)
COPY . .

# Install gunicorn for Django
RUN pip install gunicorn

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Run Django with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "tts_project.wsgi:application"]