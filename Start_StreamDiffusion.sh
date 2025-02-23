
                #!/bin/sh
                # Unset PYTHONPATH to avoid TD Python interference
                unset PYTHONPATH
                
                cd "$(dirname "$0")"
                
                if False; then
                    echo "Current directory: $(pwd)"
                    echo "PATH: $PATH"
                    echo "Available Python versions:"
                    which -a python python3
                fi
                
                if [ -d "venv" ]; then
                    source venv/bin/activate
                    if False; then
                        echo "Using venv at: $(which python)"
                        echo "Python version: $(python --version)"
                        echo "Python packages installed:"
                        pip list
                    fi
                    python streamdiffusionTD/main_sdtd.py
                else
                    source .venv/bin/activate
                    if False; then
                        echo "Using venv at: $(which python)"
                        echo "Python version: $(python --version)" 
                        echo "Python packages installed:"
                        pip list
                    fi
                    python streamdiffusionTD/main_sdtd.py
                fi
                
                