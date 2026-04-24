# Robot-EMG Setup Guide

This document outlines the strict technical requirements for setting up the Python environment and dependencies required for the SO-101 robotic arm and OpenBCI pipeline. 

## 1. Virtual Environment Creation

You must isolate the dependencies to avoid system-level conflicts. Run the following command in your terminal:

```bash
python -m venv robot_env
```

# 2. Environment Activation

You must activate the virtual environment before installing any packages.

Linux / macOS:
```
source robot_env/bin/activate
```

Windows:
```
robot_env\Scripts\activate
```

# 3. Dependency Installation

The system requires specific libraries for data processing, hardware communication, and servo control. Install all required dependencies at once using pip:

```bash
pip install numpy pandas brainflow vassar-feetech-servo-sdk
```

# 4. System Verification

To confirm the environment is properly configured, execute the environment check script:

```
python test.py
```

If the setup is correct, the terminal will report successful imports for Python, Numpy, Pandas, Brainflow, and the Vassar SDK. Do not proceed to hardware operation until this verification passes.
