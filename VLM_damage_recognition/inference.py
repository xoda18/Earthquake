"""
LLaVA Model Inference Wrapper

Handles loading and inference of the LLaVA vision-language model with GPU/CPU auto-detection
and optional quantization for CPU inference.
"""

import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class LLaVAInference:
    """Wrapper for LLaVA model inference with GPU/CPU support."""

    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", quantize: bool = False):
        """
        Initialize LLaVA model.

        Args:
            model_id: HuggingFace model identifier
            quantize: Apply int8 quantization for CPU inference
        """
        self.model_id = model_id
        self.device = self._detect_device()
        self.quantize = quantize and self.device == "cpu"

        print(f"[LLaVA] Device: {self.device}, Quantize: {self.quantize}")
        print(f"[LLaVA] Loading model: {model_id}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load model (quantization via load_in_8bit is deprecated, use normal loading)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        self.model.eval()
        print("[LLaVA] Model loaded successfully")

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect available device (cuda, mps, or cpu)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def analyze_image(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Analyze image with LLaVA.

        Args:
            image: PIL Image
            prompt: Text prompt for analysis
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        # Prepare inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode response
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the response part (after prompt)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response


def create_inference_engine(quantize: bool = False) -> LLaVAInference:
    """Factory function to create inference engine."""
    return LLaVAInference(quantize=quantize)
