import time
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import os

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'  # use ls /dev/ttyUSB* to find the correct port 
RECORD_SECONDS = 10    
ACTION_NAME = "fist"   

def record_data():
    params = BrainFlowInputParams()
    params.serial_port = PORT
    
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    
    try:
        print(f"Connecting to OpenBCI on {PORT}...")
        board.prepare_session()
        
        print("Starting stream... Get ready!")
        board.start_stream()
        time.sleep(2) # Give the board 2 seconds to stabilize
        
        print(f"\n[ RECORDING '{ACTION_NAME.upper()}' FOR {RECORD_SECONDS} SECONDS ]")
        time.sleep(RECORD_SECONDS)
        
        print("\nRecording finished. Downloading buffer...")
        # Get all data and empty the buffer
        data = board.get_board_data()
        
        # Cyton rows 1 through 8 contain the raw microvolt ExG data

        # depending on the board and the channels used, you may need to adjust the row indices. AND ALSO [*]
        emg_channels = data[1:3, :] 
        
        # Convert to Pandas DataFrame and transpose so columns are channels, rows are time
        df = pd.DataFrame(emg_channels).T
        df.columns = [f"Channel_{i}" for i in range(1, 3)]
        #                                             [*] 
        
        # 1. Define the nested target directory (e.g., Data/fist, Data/flex)
        save_dir = os.path.join("Data", ACTION_NAME)
        
        # 2. Force the OS to recursively create the folders
        # exist_ok=True prevents crashes if the folder is already there
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. Construct the filename (Keeping the action name in the file is good practice in case files are moved)
        filename = f"{ACTION_NAME}_{int(time.time())}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # 4. Save the file to the nested path
        df.to_csv(filepath, index=False)
        
        print(f"Success! Data saved to {filepath}")
        print(f"Shape: {df.shape} (Samples x Channels)")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Check your port name and ensure the dongle switch is on GPIO6.")

    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Hardware port released safely.")

if __name__ == "__main__":
    record_data()