
            #!/bin/bash
            
            # Unset PYTHONPATH to prevent conflicts
            unset PYTHONPATH
            
            echo "Current directory: $PWD"
            cd "/Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD"
            echo "Changed directory to: $PWD"
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            if [ ! -d "venv" ]; then
                echo "Creating Python venv at: /Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD/venv"
                python3 -m venv venv
            else:
                echo "Virtual environment already exists at: /Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD/venv"
            fi

            echo "Attempting to activate virtual environment..."
            source venv/bin/activate

            if [ -z "$VIRTUAL_ENV" ]; then
                echo "Failed to activate virtual environment. Please check the path and ensure the venv exists."
                echo "Path to venv: /Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD/venv"
                echo "VIRTUAL_ENV: $VIRTUAL_ENV"
                exit 1
            else:
                echo "Virtual environment activated."
            fi

            echo "Installing"
            
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip 
            
            echo "Installing huggingface_hub==0.24.6 first to prevent version conflicts..."
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org huggingface_hub==0.24.6
            
            echo "Installing 'wheel' to ensure successful building of packages..."
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org wheel

            echo "Installing numpy==1.24.1 first to ensure correct version..."
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy==1.24.1

            echo "Installing dependencies with pip from the activated virtual environment..."
            pip install  --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu --trusted-host download.pytorch.org
            pip  --trusted-host pypi.org --trusted-host files.pythonhosted.org install .
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org -r streamdiffusionTD/requirements_mac.txt

            echo "Ensuring huggingface_hub version is correct..."
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org --force-reinstall huggingface_hub==0.24.6

            echo "Ensuring numpy version is still correct..."
            pip install  --trusted-host pypi.org --trusted-host files.pythonhosted.org --force-reinstall numpy==1.24.1

            echo "Installation Finished"
            read -p "Press any key to continue..."
        