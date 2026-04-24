import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from vassar_feetech_servo_sdk import ServoController

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'
JOINT_IDS = [1, 2, 3, 4, 5, 6]

# Thresholds (Tune these by watching the terminal while resting vs flexing)
THRESH_BICEP = 150.0  
THRESH_FIST = 150.0   
COOLDOWN = 2.0 # Seconds to wait after a trigger

# --- ROBOT POSITIONS (From your Kinesthetic Teaching) ---
POSE_HOME = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
POSE_PICK = {1: 1500, 2: 1800, 3: 1500, 4: 2048, 5: 2048, 6: 2048}
POSE_PLACE = {1: 2500, 2: 1800, 3: 1500, 4: 2048, 5: 2048, 6: 2048}

# Gripper States (Assuming joint 6 is the gripper)
GRIP_OPEN = 1200
GRIP_CLOSED = 2500

def calculate_rms(data_chunk):
    return np.sqrt(np.mean(np.square(data_chunk)))

def initialize_hardware():
    # 1. Init Robot
    controller = ServoController(servo_ids=JOINT_IDS, servo_type="sts")
    controller.connect()
    controller.write_position(POSE_HOME, speed=30, acceleration=20)
    
    # 2. Init OpenBCI
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()
    
    return controller, board

def run_toggle_fsm():
    controller, board = initialize_hardware()
    
    # State Trackers
    pos_state = 0       # 0: Home, 1: Pick, 2: Place
    gripper_open = True # True: Open, False: Closed
    last_trigger_time = 0
    
    print("\n--- System Ready ---")
    print("Flex Bicep -> Cycle Position (Home -> Pick -> Place)")
    print("Make Fist  -> Toggle Gripper (Open/Close)")
    
    try:
        while True:
            data = board.get_board_data()
            
            if data.shape[1] > 0:
                # Cyton indices: Ch1 is row 1, Ch2 is row 2
                rms_bicep = calculate_rms(data[1, :])
                rms_fist = calculate_rms(data[2, :])
                
                print(f"Bicep RMS: {rms_bicep:.0f} | Fist RMS: {rms_fist:.0f}", end='\r')
                
                # Check for triggers (if cooldown has passed)
                if (time.time() - last_trigger_time) > COOLDOWN:
                    
                    # TRIGGER 1: BICEP FLEX (Cycle Positions)
                    if rms_bicep > THRESH_BICEP:
                        print(f"\n[ BICEP TRIGGER ] Moving Arm...")
                        
                        # We must preserve the current gripper state when moving the arm
                        current_grip_val = GRIP_OPEN if gripper_open else GRIP_CLOSED
                        
                        if pos_state == 0:
                            pose = POSE_PICK.copy()
                            pos_state = 1
                        elif pos_state == 1:
                            pose = POSE_PLACE.copy()
                            pos_state = 2
                        else:
                            pose = POSE_HOME.copy()
                            pos_state = 0
                            
                        # Inject the correct gripper value into the new pose
                        pose[6] = current_grip_val 
                        controller.write_position(pose, speed=40, acceleration=30)
                        last_trigger_time = time.time()
                    
                    # TRIGGER 2: HARD FIST (Toggle Gripper)
                    elif rms_fist > THRESH_FIST:
                        print(f"\n[ FIST TRIGGER ] Toggling Gripper...")
                        
                        gripper_open = not gripper_open # Flip boolean
                        new_grip_val = GRIP_OPEN if gripper_open else GRIP_CLOSED
                        
                        # Read current full arm position, update only joint 6
                        current_pose = controller.read_all_positions()
                        current_pose[6] = new_grip_val
                        
                        # Move gripper fast
                        controller.write_position(current_pose, speed=80, acceleration=50)
                        last_trigger_time = time.time()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down manually.")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        controller.disable_all_servos()
        controller.disconnect()
        print("Hardware safely disconnected.")

if __name__ == "__main__":
    run_toggle_fsm()