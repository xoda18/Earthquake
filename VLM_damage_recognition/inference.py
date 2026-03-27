"""
LLaVA Model Inference Wrapper

Handles loading and inference of the LLaVA vision-language model with GPU/CPU auto-detection
and optional quantization for CPU inference.
"""

import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
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
        # bitsandbytes quantization only works on CUDA
        self.quantize = quantize and self.device == "cuda"

        print(f"[LLaVA] Device: {self.device}, Quantize: {self.quantize}")
        print(f"[LLaVA] Loading model: {model_id}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load model — strategy depends on device
        if self.quantize:
            # CUDA only: 4-bit NF4 quantization — uses ~4GB VRAM, ideal for 8GB cards
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        elif self.device == "cuda":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif self.device == "mps":
            # MPS: no device_map="auto", manually move to MPS
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("mps")
        else:
            # CPU fallback
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        self.model.eval()
        print(f"[LLaVA] Model loaded successfully on {self.device}")

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
        # Build conversation for the processor's chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Let the processor handle image token placement
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Free CUDA cache after inference to avoid OOM on next call
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Decode only the NEW tokens (skip the input prompt tokens)
        input_len = inputs["input_ids"].shape[-1]
        response = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        return response


def create_inference_engine(quantize: bool = False) -> LLaVAInference:
    """Factory function to create inference engine."""
    return LLaVAInference(quantize=quantize)
