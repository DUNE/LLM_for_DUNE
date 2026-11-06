#!/usr/bin/env python3
"""
DUNE-GPT Setup Script
"""

import subprocess
import shutil
import shlex
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required. You have {sys.version}")
        return False
    print(f"✅ Python version OK: {sys.version}")
    return True

def setup_environment():
    """Set up the environment"""
    print("DUNE-GPT Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create environment file if it doesn't exist
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔄 Creating .env file from template...")
        template = env_example.read_text().replace(
            "your_username_here", os.getenv("USER", "your_username_here")
        )
        with open(env_file, "w") as f:
            f.write(template)
        print("✅ Created .env file. Please edit it with your credentials.")
    
    # Install requirements into the active Python environment
    python_executable = shlex.quote(sys.executable)
    uv_executable = shutil.which("uv")

    if uv_executable:
        command = (
            f"{shlex.quote(uv_executable)} pip install "
            f"--python {python_executable} -r requirements.txt"
        )
    else:
        command = f"{python_executable} -m pip install -r requirements.txt"

    if not run_command(command, "Installing Python dependencies"):
        return False
    
    
    # Make CLI executable
    try:
        os.chmod("cli.py", 0o755)
        print("✅ Made CLI executable")
    except:
        pass
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your credentials")
    print("2. Run: python cli.py index")
    print("3. Run: python cli.py serve")
    print("\nFor help: python cli.py --help")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 