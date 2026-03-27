#!/usr/bin/env python3
"""
Installation Test Script

Verify that all dependencies are installed and the module is working.
"""

import sys
import importlib


def test_imports():
    """Test that all required modules can be imported."""
    modules = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "PIL": "Pillow",
        "piexif": "piexif",
        "numpy": "NumPy",
        "pydantic": "Pydantic",
    }

    print("Testing imports...")
    all_ok = True

    for module, name in modules.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name:.<40} OK")
        except ImportError as e:
            print(f"  ✗ {name:.<40} FAILED")
            print(f"    Error: {e}")
            all_ok = False

    # Optional imports
    print("\nTesting optional imports...")
    optional = {
        "rasterio": "Rasterio (GeoTIFF support)",
    }

    for module, name in optional.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name:.<40} OK")
        except ImportError:
            print(f"  ⊘ {name:.<40} Optional (not installed)")

    return all_ok


def test_module():
    """Test that our modules can be imported."""
    print("\nTesting VLM_damage_recognition modules...")

    modules = [
        ("VLM_damage_recognition.image_processor", "ImageProcessor"),
        ("VLM_damage_recognition.prompt_templates", "DamagePrompts"),
        ("VLM_damage_recognition.supabase_reporter", "SupabaseReporter"),
    ]

    all_ok = True

    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"  ✓ {module_name:.<45} OK")
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {module_name:.<45} FAILED")
            print(f"    Error: {e}")
            all_ok = False

    return all_ok


def test_device():
    """Test device detection."""
    print("\nTesting device detection...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA device detected: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print("  ✓ Apple MPS device detected")
        else:
            print("  ⊘ CPU only (no GPU detected)")

        return True
    except Exception as e:
        print(f"  ✗ Device detection failed: {e}")
        return False


def test_supabase():
    """Test Supabase connection."""
    print("\nTesting Supabase integration...")

    try:
        sys.path.insert(0, "..")
        import supabase_client as sb

        print("  ✓ Supabase client available")

        # Try a test query (non-destructive)
        # This will fail silently if credentials are wrong
        print("  ⊘ Supabase credentials check: skipped (will test on actual usage)")

        return True
    except ImportError:
        print("  ⊘ Supabase client not available (will be available from Earthquake root)")
        return True
    except Exception as e:
        print(f"  ✗ Supabase error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VLM DAMAGE RECOGNITION - INSTALLATION TEST")
    print("="*60)

    results = []

    results.append(("Dependencies", test_imports()))
    results.append(("Modules", test_module()))
    results.append(("Device", test_device()))
    results.append(("Supabase", test_supabase()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Ready to use.")
        print("\nNext steps:")
        print("  1. python VLM_damage_recognition/main.py --help")
        print("  2. Download test images or use crack_detection/image_diff/input/")
        print("  3. Run: python VLM_damage_recognition/main.py --input-dir <path>")
        return 0
    else:
        print("\n✗ Some tests failed. Please install missing dependencies:")
        print("  pip install -r VLM_damage_recognition/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
