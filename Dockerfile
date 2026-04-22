FROM python:3.11-slim

# HuggingFace Spaces recommends running as non-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install server dependencies first (layer caching)
COPY --chown=user server/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy full package source
COPY --chown=user . /app

# Install the undertrial_ai package itself
RUN pip install --no-cache-dir -e .

# Episode data directory (falls back to built-in demo episodes if empty)
ENV UNDERTRIAL_EPISODES_DIR=/app/data/episodes

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "undertrial_ai.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
