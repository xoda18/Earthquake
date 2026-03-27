# GPU Setup Guide for VLM Inference

## Current Status

✅ **Code**: Ready for GPU (PyTorch with CUDA 12.1)
✅ **Docker image**: Builds successfully with GPU libraries
❌ **Runtime**: nvidia-container-runtime not installed

The VLM system is configured to use GPU (RTX 4060 Max-Q) for fast inference (~2-3 sec/image instead of CPU minutes), but the Docker daemon needs nvidia-container-runtime to access the GPU.

---

## Quick Test (CPU-only for now)

To test the VLM is working correctly on CPU:

```bash
# Start VLM analyzer on CPU
docker compose -f docker-compose.cpu.yml up vlm-analyzer

# In another terminal, test inference
curl -X POST http://localhost:5060/analyze \
  -F "file=@test_vlm_with_damage_images.py" \
  -F "lat=34.765" \
  -F "lon=32.42" \
  -F "building=Test"
```

This tests the image processing, LLaVA model loading, damage schema, and Supabase integration without needing GPU. It will be slower on CPU but confirms the entire pipeline works.

---

## Enable GPU (Requires sudo)

### Prerequisites
- NVIDIA GPU present (confirmed: RTX 4060 Max-Q)
- NVIDIA drivers installed (`nvidia-smi` should work)
- Docker 19.03+

### Step 1: Install nvidia-container-runtime

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-runtime
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Verify installation
which nvidia-container-runtime
```

### Step 2: Configure Docker daemon

Create/edit `/etc/docker/daemon.json` (requires sudo):

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

### Step 3: Verify GPU access

```bash
# Test GPU from Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# Should show your RTX 4060 Max-Q GPU
```

### Step 4: Start VLM with GPU

```bash
# Build and start with GPU enabled
docker compose up vlm-analyzer --build

# Monitor GPU usage in another terminal
docker exec -it earthquake-vlm-analyzer nvidia-smi -l 1
```

---

## Switching Between CPU and GPU

**CPU-only** (for testing or systems without nvidia-container-runtime):
```bash
docker compose -f docker-compose.cpu.yml up vlm-analyzer
```

**GPU-enabled** (after installing nvidia-container-runtime):
```bash
docker compose up vlm-analyzer
```

Both use the same image (`earthquake-vlm:gpu` with GPU PyTorch). The difference is:
- **CPU**: Uses PyTorch GPU libraries on CPU (slower but works)
- **GPU**: Uses PyTorch GPU libraries on actual NVIDIA GPU (2-3x faster)

---

## Performance Expectations

| Mode | Time/Image | Model | Notes |
|------|-----------|-------|-------|
| CPU | 60-180 sec | LLaVA 1.5 7B | PyTorch on CPU, very slow |
| GPU (RTX 4060 Max-Q) | 2-5 sec | LLaVA 1.5 7B | Fast inference, GPU-accelerated |

---

## Troubleshooting

### "could not select device driver"
- nvidia-container-runtime not installed → Run Step 1 above
- Docker daemon not restarted → Run `sudo systemctl restart docker`
- Check with: `docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi`

### nvidia-smi not found
- NVIDIA drivers not installed
- Run: `sudo apt-get install nvidia-driver-550` (or latest version)
- Reboot after installation: `sudo reboot`

### GPU shows up in nvidia-smi but Docker can't see it
- Docker was started before GPU drivers → Restart Docker: `sudo systemctl restart docker`
- Or restart the entire system: `sudo reboot`

### Model downloads keep timing out on slow internet
- Set model cache: `mkdir -p ~/.cache/huggingface/hub && export HF_HOME=~/.cache/huggingface`
- Or use smaller model: `MODEL_ID=llava-hf/llava-1.5-7b-hf` (7B instead of 13B)

---

## Files Modified for GPU Support

- **vlm/Dockerfile**: Changed from CPU to GPU PyTorch (`--index-url https://download.pytorch.org/whl/cu121`)
- **VLM_damage_recognition/requirements.txt**: Added `accelerate` and `bitsandbytes` (GPU support libs)
- **docker-compose.yml**: Added GPU device allocation in `vlm-analyzer` service
- **docker-compose.cpu.yml**: New file for CPU-only testing

---

## References

- [NVIDIA Container Runtime Installation](https://github.com/NVIDIA/nvidia-container-runtime)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
