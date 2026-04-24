# SO-101 Robotic Arm — Realistic Integration Plan (Tight Deadline)

## 0. System Architecture

```text
[ EMG Sensor ] -> [ Classifier (Flex, Fist, etc.) ]
                          ↓
              [ Python Logic (Mapping classes to target poses) ]
                          ↓
              [ Vassar Feetech Servo SDK ]
                          ↓
              [ Feetech Driver Board (USB -> TTL multiplexer) ]
                          ↓
              [ STS3215 Servo Bus (Daisy-chained) ]

1. Hardware & Setup Constraints (CRITICAL)

Before writing code, you must secure the hardware. Software cannot fix physical power failures.

    Power Supply: Do not use a standard 2A USB wall plug. Six STS3215 servos under load will draw 10+ Amps. You need a dedicated 6V-8.4V DC power supply rated for at least 10A, or a high-discharge 2S LiPo battery plugged into the XT60/barrel jack of your driver board.

    Wiring: Daisy-chain the servos. Plug Servo 1 into the driver board, Servo 2 into Servo 1, and so on. Connect the driver board to the PC via USB.

    Environment Setup: Create a virtual environment and install the required SDK.
    Bash

    python -m venv robot_env
    source robot_env/bin/activate  # On Windows: robot_env\Scripts\activate
    pip install vassar-feetech-servo-sdk

2. Phase 1: Kinesthetic Teaching (The Kinematics Shortcut)

Do not use Inverse Kinematics (IK). Instead, physically move the arm to the desired positions and record the raw encoder values. Save this code as teach.py.
Python

import time
from vassar_feetech_servo_sdk import ServoController

# Initialize the 6 joints of the SO-101
JOINT_IDS = [1, 2, 3, 4, 5, 6]

def main():
    # Connect and immediately disable torque so you can move the arm by hand
    controller = ServoController(servo_ids=JOINT_IDS, servo_type="sts")
    controller.connect()
    controller.disable_all_servos()

    print("Torque disabled. Move the arm to your target position.")
    print("Press Ctrl+C to lock and record the position.")

    try:
        while True:
            positions = controller.read_all_positions()
            # Print current positions to monitor in real-time
            print(f"Current Encoder Values: {positions}", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n--- Recorded Position Dictionary ---")
        print("Copy this into your main.py file:")
        print(controller.read_all_positions())
    finally:
        controller.disconnect()

if __name__ == "__main__":
    main()

Workflow:

    Run python teach.py.

    Manually position the arm to represent the response for "Flex Bicep".

    Press Ctrl+C.

    Copy the printed dictionary into your main.py file.

    Repeat for "Make a Fist" and your "Home/Rest" position.

3. Phase 2: Safe Playback and Integration

The Vassar SDK's default speed is maximum velocity. You must explicitly define speed and acceleration limits, or the arm will break itself. Save this code as main.py.
Python

import time
from vassar_feetech_servo_sdk import ServoController

JOINT_IDS = [1, 2, 3, 4, 5, 6]

# Hardcoded dictionaries generated from teach.py
POSE_HOME = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
POSE_FLEX = {1: 2048, 2: 1500, 3: 1800, 4: 2048, 5: 2048, 6: 1200}
POSE_FIST = {1: 1500, 2: 2500, 3: 1200, 4: 2048, 5: 2048, 6: 2500}

# Global speed and acceleration limits (Tune these carefully)
SAFE_SPEED = 50   # roughly 36 RPM
SAFE_ACCEL = 40

def initialize_arm():
    controller = ServoController(servo_ids=JOINT_IDS, servo_type="sts")
    controller.connect()
    
    # Read current state to ensure connection is valid
    current = controller.read_all_positions()
    print(f"Boot up raw positions: {current}")
    
    # Slowly move to home position on startup to prevent violent snapping
    print("Moving to HOME position...")
    controller.write_position(POSE_HOME, speed=30, acceleration=20)
    time.sleep(3) # Wait for movement to finish
    
    return controller

def main():
    controller = initialize_arm()
    current_state = "home"
    
    try:
        while True:
            # --- REPLACE THIS WITH YOUR ACTUAL EMG CLASSIFIER LOGIC ---
            # emg_class = get_emg_classification()
            emg_class = input("Enter simulated EMG class (flex/fist/home/quit): ").strip().lower()
            # ---------------------------------------------------------
            
            # Prevent command spamming (only send command if state changes)
            if emg_class == current_state:
                continue
                
            if emg_class == "flex":
                controller.write_position(POSE_FLEX, speed=SAFE_SPEED, acceleration=SAFE_ACCEL)
                current_state = "flex"
            elif emg_class == "fist":
                controller.write_position(POSE_FIST, speed=SAFE_SPEED, acceleration=SAFE_ACCEL)
                current_state = "fist"
            elif emg_class == "home":
                controller.write_position(POSE_HOME, speed=SAFE_SPEED, acceleration=SAFE_ACCEL)
                current_state = "home"
            elif emg_class == "quit":
                break
                
            time.sleep(0.1)

    finally:
        print("Shutting down. Disabling torque.")
        controller.disable_all_servos()
        controller.disconnect()

if __name__ == "__main__":
    main()

4. Known Bugs & Necessary Precautions

    Servo ID Conflicts: Out of the box, all Feetech servos are programmed as ID 1. You cannot plug all six into the board at once. You must plug them in one by one and use the change_servo_id.py script provided in the Vassar SDK repository to set them to IDs 1 through 6 before assembling the arm.

    Command Spamming Chokeholds: Notice the if emg_class == current_state: continue block in main.py. Your EMG classifier will output data at a high frequency. If you call write_position() 100 times a second with the exact same target, the serial bus will choke. You must only send commands on state transitions.

    Gravity Sag: The SO-101 has a long lever arm. When moving to a predetermined position under its own weight, it might sag slightly below what you recorded during the "Kinesthetic Teaching" phase when your hands were supporting it. You will likely need to manually adjust your recorded encoder values by a few digits to compensate for this droop.