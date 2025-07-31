#!/usr/bin/env python3
"""
Launcher script for Medical Reasoning Dataset Generator.

This script properly sets up the Python path and launches the medical dataset generator
to avoid relative import issues.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import and run the main module
if __name__ == "__main__":
    # Import main after setting up the path
    from main import main
    
    # Run the main function
    main()