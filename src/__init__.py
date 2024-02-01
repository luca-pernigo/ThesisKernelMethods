import sys
import os

# Get the directory of the current file (__init__.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory (src)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to the Python path
sys.path.append(current_dir)
sys.path.append(parent_dir)
