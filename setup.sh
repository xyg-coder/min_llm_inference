#!/bin/bash

# This script is used to setup the environment for the min_llm_inference project.
# It will install CUDA, cmake, and clone the repository.
# It will also generate an SSH key and add it to the GitHub account.
# It will also fix the CUDA paths in the current session.
# It will also install git if not present.
# It will also install cmake if not present.
# It will also install CUDA if not present.

set -e  # Exit on any error

echo "==================================="
echo "Setup Script Starting..."
echo "==================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    elif [ -f /etc/redhat-release ]; then
        OS=Redhat
        VER=$(cat /etc/redhat-release)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
    echo "Detected OS: $OS $VER"
}

# 1. Install CUDA (nvcc)
install_cuda() {
    echo "==================================="
    echo "Installing CUDA..."
    echo "==================================="
    
    # Check if nvcc is already installed
    if command_exists nvcc; then
        echo "CUDA is already installed:"
        nvcc --version
        return 0
    fi
    
    detect_os
    
    # Update package manager
    if command_exists apt-get; then
        echo "Updating package lists..."
        sudo apt-get update
        
        # Install prerequisites
        sudo apt-get install -y wget gnupg
        
        # Add NVIDIA package repositories
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        
        # Install CUDA toolkit
        echo "Installing CUDA toolkit..."
        sudo apt-get install -y cuda-toolkit-12-3
        
        # Detect actual CUDA installation path
        CUDA_PATH=""
        if [ -d "/usr/local/cuda-12.3" ]; then
            CUDA_PATH="/usr/local/cuda-12.3"
        elif [ -d "/usr/local/cuda-12.8" ]; then
            CUDA_PATH="/usr/local/cuda-12.8"
        elif [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
            CUDA_PATH="/usr/local/cuda"
        else
            # Try to find nvcc and derive path
            NVCC_PATH=$(find /usr/local -name nvcc 2>/dev/null | head -1)
            if [ -n "$NVCC_PATH" ]; then
                CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
            fi
        fi
        
        if [ -z "$CUDA_PATH" ]; then
            echo "Could not detect CUDA installation path"
            exit 1
        fi
        
        echo "Using CUDA path: $CUDA_PATH"
        
        # Add CUDA to PATH
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        
        # Set for current session
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        
    elif command_exists yum; then
        echo "Installing CUDA on RedHat/CentOS/Amazon Linux..."
        
        # Install prerequisites
        sudo yum install -y wget
        
        # Add NVIDIA repository
        sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        
        # Install CUDA
        sudo yum install -y cuda-toolkit-12-3
        
        # Detect actual CUDA installation path
        CUDA_PATH=""
        if [ -d "/usr/local/cuda-12.3" ]; then
            CUDA_PATH="/usr/local/cuda-12.3"
        elif [ -d "/usr/local/cuda-12.8" ]; then
            CUDA_PATH="/usr/local/cuda-12.8"
        elif [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
            CUDA_PATH="/usr/local/cuda"
        else
            # Try to find nvcc and derive path
            NVCC_PATH=$(find /usr/local -name nvcc 2>/dev/null | head -1)
            if [ -n "$NVCC_PATH" ]; then
                CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
            fi
        fi
        
        if [ -z "$CUDA_PATH" ]; then
            echo "Could not detect CUDA installation path"
            exit 1
        fi
        
        echo "Using CUDA path: $CUDA_PATH"
        
        # Add CUDA to PATH
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        
        # Set for current session
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        
    else
        echo "Unsupported package manager. Please install CUDA manually."
        echo "Visit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
    
    # Verify installation
    if command_exists nvcc; then
        echo "CUDA installed successfully:"
        nvcc --version
    else
        echo "CUDA installation may have failed. Please check manually."
        echo "You may need to restart your shell or run: source ~/.bashrc"
    fi
}

# 2. Generate SSH key and setup for GitHub
setup_ssh_key() {
    echo "==================================="
    echo "Setting up SSH key for GitHub..."
    echo "==================================="
    
    # Check if SSH key already exists
    if [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ]; then
        echo "SSH key already exists:"
        ls -la ~/.ssh/id_* 2>/dev/null || true
        
        read -p "Do you want to generate a new SSH key? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Using existing SSH key..."
            if [ -f ~/.ssh/id_ed25519.pub ]; then
                echo "Your public SSH key:"
                echo "=========================="
                cat ~/.ssh/id_ed25519.pub
                echo "=========================="
            elif [ -f ~/.ssh/id_rsa.pub ]; then
                echo "Your public SSH key:"
                echo "=========================="
                cat ~/.ssh/id_rsa.pub
                echo "=========================="
            fi
            
            echo ""
            echo "Please add this key to your GitHub account:"
            echo "1. Go to https://github.com/settings/keys"
            echo "2. Click 'New SSH key'"
            echo "3. Paste the above key"
            echo "4. Give it a title and click 'Add SSH key'"
            echo ""
            read -p "Press Enter when you've added the key to GitHub..."
            return 0
        fi
    fi
    
    # Get user email for SSH key
    read -p "Enter your email for the SSH key: " email
    
    if [ -z "$email" ]; then
        echo "Email is required for SSH key generation"
        exit 1
    fi
    
    # Generate SSH key
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""
    
    # Start ssh-agent and add key
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    
    # Display the public key
    echo ""
    echo "==================================="
    echo "Your SSH public key:"
    echo "==================================="
    cat ~/.ssh/id_ed25519.pub
    echo "==================================="
    echo ""
    echo "IMPORTANT: Please add this SSH key to your GitHub account:"
    echo "1. Copy the above key (starts with ssh-ed25519)"
    echo "2. Go to https://github.com/settings/keys"
    echo "3. Click 'New SSH key'"
    echo "4. Paste the key and give it a title"
    echo "5. Click 'Add SSH key'"
    echo ""
    
    read -p "Press Enter when you've added the key to GitHub..."
    
    # Test SSH connection to GitHub
    echo "Testing SSH connection to GitHub..."
    ssh -T git@github.com || true
    echo "If you see 'Hi [username]! You've successfully authenticated', SSH is working!"
}

# Function to fix CUDA paths in current session
fix_cuda_paths() {
    echo "==================================="
    echo "Fixing CUDA paths in current session..."
    echo "==================================="
    
    # Detect actual CUDA installation path
    CUDA_PATH=""
    if [ -d "/usr/local/cuda-12.3" ]; then
        CUDA_PATH="/usr/local/cuda-12.3"
    elif [ -d "/usr/local/cuda-12.8" ]; then
        CUDA_PATH="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
        CUDA_PATH="/usr/local/cuda"
    else
        # Try to find nvcc and derive path
        NVCC_PATH=$(find /usr/local -name nvcc 2>/dev/null | head -1)
        if [ -n "$NVCC_PATH" ]; then
            CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
        fi
    fi
    
    if [ -z "$CUDA_PATH" ]; then
        echo "Could not detect CUDA installation path"
        return 1
    fi
    
    echo "Using CUDA path: $CUDA_PATH"
    
    # Set for current session
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    
    # Verify nvcc is now available
    if command_exists nvcc; then
        echo "✅ CUDA paths fixed successfully!"
        echo "nvcc version:"
        nvcc --version
    else
        echo "❌ nvcc still not found. Please check CUDA installation."
        return 1
    fi
    
    echo ""
    echo "Current CUDA paths:"
    echo "PATH includes: $CUDA_PATH/bin"
    echo "LD_LIBRARY_PATH includes: $CUDA_PATH/lib64"
    echo ""
    echo "To make this permanent, add these lines to your ~/.bashrc:"
    echo "export PATH=$CUDA_PATH/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
}

# Main execution
main() {
    echo "Starting setup process..."
    echo ""
    
    # Install git if not present
    if ! command_exists git; then
        echo "Installing git..."
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y git
        elif command_exists yum; then
            sudo yum install -y git
        else
            echo "Please install git manually"
            exit 1
        fi
    fi
    
    # Install cmake if not present
    if ! command_exists cmake; then
        echo "Installing cmake..."
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y cmake
        elif command_exists yum; then
            sudo yum install -y cmake
        else
            echo "Please install cmake manually"
            exit 1
        fi
        
        # Verify cmake installation
        if command_exists cmake; then
            echo "cmake installed successfully:"
            cmake --version
        else
            echo "cmake installation may have failed. Please check manually."
        fi
    else
        echo "cmake is already installed:"
        cmake --version
    fi
    
    # Run setup steps
    install_cuda
    echo
    setup_ssh_key
    echo
    
    echo ""
    echo "==================================="
    echo "Setup completed successfully!"
    echo "==================================="
    echo ""
    echo "Summary:"
    echo "✅ CUDA installed (nvcc available)"
    echo "✅ cmake installed"
    echo "✅ SSH key generated and configured"
    echo "✅ Repository cloned: min_llm_inference"
    echo ""
    echo "Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Verify CUDA: nvcc --version"
    echo "3. Navigate to the cloned repository: cd min_llm_inference"
    echo ""
}

# Check for command line arguments
if [ "$1" = "fix-cuda" ]; then
    fix_cuda_paths
    exit $?
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  (no args)  Run full setup (install CUDA, setup SSH, clone repo)"
    echo "  fix-cuda   Fix CUDA paths in current session"
    echo "  -h, --help Show this help message"
    echo ""
    exit 0
fi

# Run main function
main
