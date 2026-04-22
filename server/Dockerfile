FROM python:3.11-slim

WORKDIR /app

# Copy server requirements
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full package
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Copy episodes data if present
ENV UNDERTRIAL_EPISODES_DIR=/app/data/episodes

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "undertrial_ai.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
