#!/usr/bin/env python
import sys
import os
import time
import threading
import termios
import tty

# Add parent directory to path to import so101_api
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from so101_api import SO101ARM

class FSMController:
    def __init__(self, arm_instance):
        self.arm = arm_instance
        self.current_state = "HOME"
        self.target_task = None  # Can be "A", "B", "ABORT", or None
        self.running = True
        self.holding_object = False

        # Input threading setup
        self.input_thread = threading.Thread(target=self._keyboard_listener_thread)
        self.input_thread.daemon = True

    def start(self):
        self.arm.connect()
        self.holding_object = self.arm.is_grasping()
        
        print("\n--- FSM Controller Started ---")
        print("Controls: '1' = Go to A | '2' = Go to B | 'q' = Abort/Home | 'esc' = Exit")
        
        self.input_thread.start()
        self._run_fsm_loop()

    def _get_waypoint_name(self, loc, z_level):
        """Generates waypoint string based on current grip state."""
        grip_state = "grip" if self.holding_object else "open"
        return f"{loc}_{z_level}_{grip_state}"

    def _run_fsm_loop(self):
        active_loc = None  # Tracks which location we are actively processing
        
        try:
            while self.running:
                self.holding_object = self.arm.is_grasping()

                if self.current_state == "HOME":
                    if self.target_task in ["A", "B"]:
                        active_loc = self.target_task
                        self.current_state = "APPROACHING_TARGET"

                # --- GENERIC LOCATION LOGIC ---
                elif self.current_state == "APPROACHING_TARGET":
                    wp = self._get_waypoint_name(active_loc, "top")
                    self.arm.move_to(wp, wait=True)
                    
                    if self.target_task != active_loc:
                        print(f"\n[FSM] Interrupted at {active_loc}_top. Routing Home.")
                        self.current_state = "HOME"
                        self.arm.move_to("home", wait=True) #addition. needed?
                    else:
                        self.current_state = "DESCENDING_TARGET"

                elif self.current_state == "DESCENDING_TARGET":
                    wp = self._get_waypoint_name(active_loc, "bot")
                    self.arm.move_to(wp, wait=True)
                    
                    if self.target_task != active_loc:
                        print(f"\n[FSM] Interrupted at {active_loc}_bot. Forcing ABORT_RETRACT.")
                        self.current_state = "ABORT_RETRACT"
                    else:
                        self.current_state = "ACTUATING_TARGET"

                elif self.current_state == "ACTUATING_TARGET":
                    # Record intention
                    intended_to_drop = self.holding_object
                    
                    # Actuate
                    if intended_to_drop:
                        self.arm.move_to(f"{active_loc}_bot_open", wait=True)
                    else:
                        self.arm.move_to(f"{active_loc}_bot_grip", wait=True)
                    
                    time.sleep(0.1) # Mechanical settling time before reading load
                    self.holding_object = self.arm.is_grasping()
                    
                    # VERIFICATION CHECK
                    if intended_to_drop and self.holding_object:
                        print("\n[FAULT] Failed to drop object. Object stuck.")
                        self.current_state = "ABORT_RETRACT"
                    elif not intended_to_drop and not self.holding_object:
                        print("\n[FAULT] Failed to grip object. Thin air grasped.")
                        self.current_state = "ABORT_RETRACT"
                    else:
                        self.current_state = "ASCENDING_TARGET"

                elif self.current_state == "ASCENDING_TARGET":
                    wp = self._get_waypoint_name(active_loc, "top")
                    self.arm.move_to(wp, wait=True)
                    
                    self.arm.move_to("home", wait=True)
                    self.target_task = None
                    active_loc = None
                    self.current_state = "HOME"

                # --- ESCAPE ROUTING ---
                elif self.current_state == "ABORT_RETRACT":
                    if active_loc:
                        # Extract straight up from wherever we are
                        wp_top = self._get_waypoint_name(active_loc, "top")
                        self.arm.move_to(wp_top, wait=True)
                    
                    self.arm.move_to("home", wait=True)
                    
                    # Reset all tasks and state after clearing the workspace
                    self.target_task = None
                    active_loc = None
                    self.current_state = "HOME"

                time.sleep(0.02) # Yield thread

        except Exception as e:
            print(f"\nHardware Fault in FSM Loop: {e}")
        finally:
            self.running = False

    def _keyboard_listener_thread(self):
        """Reads raw keystrokes from Linux terminal without requiring 'Enter'."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                ch = sys.stdin.read(1)
                
                if ch == '1':
                    self.target_task = "A"
                    print("\r\n[Input] Command Received: Execute A")
                elif ch == '2':
                    self.target_task = "B"
                    print("\r\n[Input] Command Received: Execute B")
                elif ch == 'q':
                    self.target_task = "ABORT"
                    print("\r\n[Input] Command Received: ABORT!")
                elif ch == '\x1b': # Escape key
                    self.running = False
                    print("\r\n[Input] Exit command received. Shutting down...")
                    break
                    
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    arm = SO101ARM()
    controller = FSMController(arm)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        controller.running = False
        arm.disconnect()