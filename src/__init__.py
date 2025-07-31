"""
Medical Reasoning Dataset Generator

A professional synthetic medical dataset generation system that creates high-quality
medical reasoning datasets through simulated doctor-patient interactions.

This system produces dual datasets:
- Raw dataset: Unedited generated medical cases
- Evaluated dataset: LLM-reviewed and refined medical cases

Built for commercial licensing and production use.
"""

__version__ = "1.0.0"
__author__ = "Medical Dataset Generator Team"
__license__ = "Commercial"

from .core import *
from .models import *
from .utils import *