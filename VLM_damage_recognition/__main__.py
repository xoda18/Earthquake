"""
Package entry point - allows running as: python -m VLM_damage_recognition
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
