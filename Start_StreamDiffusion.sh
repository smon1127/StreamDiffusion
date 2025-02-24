#!/bin/sh
# Unset PYTHONPATH to avoid TD Python interference
unset PYTHONPATH

# Change to the script's directory
cd "$(dirname "$0")"

# Debug info (disabled by default)
if false; then
    echo "Current directory: $(pwd)"
    echo "PATH: $PATH"
    echo "Available Python versions:"
    which -a python python3
fi

# Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    if false; then
        echo "Using venv at: $(which python)"
        echo "Python version: $(python --version)"
        echo "Python packages installed:"
        pip list
    fi
    # Run the main script with the correct config path
    python main_sdtd.py -c stream_config.json
else
    source .venv/bin/activate
    if false; then
        echo "Using venv at: $(which python)"
        echo "Python version: $(python --version)" 
        echo "Python packages installed:"
        pip list
    fi
    # Run the main script with the correct config path
    python main_sdtd.py -c stream_config.json
fi
