import sys
import time
import threading
import queue
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# --- 1. Actuation Thread ---
def servo_worker(command_queue, servo_port):
    # Initialize Vassar SDK here using servo_port
    # controller = ServoController(servo_port)
    print(f"Servo worker started on {servo_port}")
    
    while True:
        try:
            # Block until a command is available
            command = command_queue.get(timeout=1.0)
            if command == "STOP":
                break
            
            # Execute hardware command
            # controller.move(command['id'], command['position'])
            print(f"Executing: {command}")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Hardware error: {e}")


def ToggleGripper():
    # This function would contain the logic to toggle the gripper's state
    # For example, it could check the current state and send the appropriate command to open or close the gripper
    pass

def goToPosition(position):
    # This function would contain the logic to move the gripper to a specific position
    # It would send the command to the servo controller to move to the desired position
    pass

# --- 2. Main DSP Pipeline ---
def main(bci_port, servo_port):
    # Setup Actuation Queue and Thread
    cmd_queue = queue.Queue()
    actuator_thread = threading.Thread(target=servo_worker, args=(cmd_queue, servo_port), daemon=True)
    actuator_thread.start()

    # Setup BrainFlow
    params = BrainFlowInputParams()
    params.serial_port = bci_port
    board_id = BoardIds.CYTON_BOARD.value # Change to your specific board
    board = BoardShim(board_id, params)
    
    try:
        board.prepare_session()
        board.start_stream(45000) # Ring buffer size
        print("Stream started.")
        
        # Determine the channels for your board
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        
        # E.g., process 250ms of data at a time
        window_size = int(sampling_rate * 0.25) 
        
        while True:
            # Sleep to allow buffer to fill (soft real-time)
            time.sleep(0.05) 
            
            # Get latest data without removing it from the buffer
            data = board.get_current_board_data(window_size)
            
            if data.shape[1] < window_size:
                continue # Not enough data yet
                
            eeg_data = data[eeg_channels, :]
            
            # --- DSP LOGIC HERE ---
            # Example: Calculate variance on channel 0
            # variance = np.var(eeg_data[0])
            
            # --- DECISION LOGIC HERE ---
            # If condition met, send non-blocking command to queue
            # if variance > threshold:
            #     cmd_queue.put({"id": 1, "position": 100})
            
    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        cmd_queue.put("STOP")
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        actuator_thread.join()
        print("Pipeline shut down safely.")

if __name__ == "__main__":
    # Do not hardcode physical locations. Pass them as arguments or config.
    bci_port = "/dev/ttyUSB0" 
    servo_port = "/dev/ttyUSB1"
    main(bci_port, servo_port)