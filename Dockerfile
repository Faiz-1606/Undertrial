FROM python:3.11-slim

WORKDIR /app

# Install server dependencies first (layer caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full package source
COPY . .

# Install the undertrial_ai package itself (editable)
RUN pip install --no-cache-dir -e .

# Episode data directory (populated at build time or via HF dataset mount)
ENV UNDERTRIAL_EPISODES_DIR=/app/data/episodes

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "undertrial_ai.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
