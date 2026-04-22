FROM python:3.11-slim

# HuggingFace Spaces recommends running as non-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# /workspace is the pip build root
# /workspace/undertrial_ai is the actual Python package
WORKDIR /workspace

# Install server dependencies first (layer caching)
COPY --chown=user server/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy pyproject.toml to workspace root (pip needs it here)
COPY --chown=user pyproject.toml pyproject.toml

# Copy entire project into undertrial_ai/ subdirectory
# This creates: /workspace/undertrial_ai/__init__.py etc.
# which is the correct importable structure
COPY --chown=user . /workspace/undertrial_ai/

# pip install from /workspace — setuptools finds undertrial_ai/ package dir
RUN pip install --no-cache-dir --upgrade .

# Episode data directory (built-in demo episodes used as fallback)
ENV UNDERTRIAL_EPISODES_DIR=/workspace/undertrial_ai/data/episodes

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "undertrial_ai.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
