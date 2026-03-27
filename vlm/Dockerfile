FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install VLM dependencies (LLaVA + FastAPI)
# GPU PyTorch (CUDA 12.1) installed first for fast inference
COPY VLM_damage_recognition/requirements.txt /tmp/vlm_requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1

# Install transformers and other dependencies
RUN pip install --no-cache-dir -r /tmp/vlm_requirements.txt

# Install API server and utilities
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart accelerate bitsandbytes

# Copy VLM damage recognition module
COPY VLM_damage_recognition/ /app/VLM_damage_recognition/

# Copy shared supabase client (needed by SupabaseReporter)
COPY supabase_client.py /app/supabase_client.py

# Copy VLM server + comparison module
COPY vlm/server.py vlm/compare.py ./

EXPOSE 5060

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5060"]
