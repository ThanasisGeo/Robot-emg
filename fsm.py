#!/usr/bin/env python
import sys
import os
import time
import threading
import termios
import tty
import select

# Add parent directory to path to import so101_api
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from so101_api import SO101ARM

class FSMController:
    def __init__(self, arm_instance):
        self.arm = arm_instance
        self.current_state = "HOME"
        self.target_task = None
        self.running = True
        self.holding_object = False

        self.input_thread = threading.Thread(target=self._keyboard_listener_thread)
        self.input_thread.daemon = True # Safe now, main thread handles terminal cleanup

    def start(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            # Terminal goes to raw mode inside the main thread's protection
            tty.setraw(fd)
            
            self.arm.connect()
            self.holding_object = self.arm.is_grasping()
            
            print("--- FSM Controller Started ---", end="\r\n")
            print("Controls: '1' = Go to A | '2' = Go to B | 'q' = Abort/Home | 'esc' = Exit", end="\r\n")
            
            self.input_thread.start()
            self._run_fsm_loop()
            
        except Exception as e:
            print(f"[Fatal Error] Main thread crashed: {e}", end="\r\n")
        finally:
            # Guaranteed terminal restoration
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.running = False
            self.arm.disconnect()
            print("[System] Terminal restored and arm disconnected. Exiting.", end="\r\n")

    def _get_waypoint_name(self, loc, z_level):
        grip_state = "grip" if self.holding_object else "open"
        return f"{loc}_{z_level}_{grip_state}"

    def _get_home_waypoint(self):
        grip_state = "grip" if self.holding_object else "open"
        return f"home_{grip_state}"

    def _run_fsm_loop(self):
        active_loc = None
        
        while self.running:
            if self.current_state == "HOME":
                if self.target_task in ["A", "B"]:
                    active_loc = self.target_task
                    self.current_state = "APPROACHING_TARGET"

            # --- GENERIC LOCATION LOGIC ---
            elif self.current_state == "APPROACHING_TARGET":
                wp = self._get_waypoint_name(active_loc, "top")
                self.arm.move_to(wp, wait=True)
                
                if self.target_task != active_loc:
                    print(f"[FSM] Interrupted at {active_loc}_top. Routing Home.", end="\r\n")
                    self.arm.move_to(self._get_home_waypoint(), wait=True)
                    self.current_state = "HOME"
                else:
                    self.current_state = "DESCENDING_TARGET"

            elif self.current_state == "DESCENDING_TARGET":
                wp = self._get_waypoint_name(active_loc, "bot")
                self.arm.move_to(wp, wait=True)
                
                if self.target_task != active_loc:
                    print(f"[FSM] Interrupted at {active_loc}_bot. Forcing ABORT_RETRACT.", end="\r\n")
                    self.current_state = "ABORT_RETRACT"
                else:
                    self.current_state = "ACTUATING_TARGET"

            elif self.current_state == "ACTUATING_TARGET":
                intended_to_drop = self.holding_object
                
                # 1. Execute Actuation
                if intended_to_drop:
                    self.arm.move_to(f"{active_loc}_bot_open", wait=True)
                else:
                    self.arm.move_to(f"{active_loc}_bot_grip", wait=True)
                
                # 2. Wait for Servo PID to build torque (CRITICAL FIX)
                time.sleep(0.5) 
                
                # 3. Verify Physical Truth once
                actual_grip_state = self.arm.is_grasping()
                
                # 4. State Update and Fault Evaluation
                if intended_to_drop:
                    if actual_grip_state: # Tried to open, but still holding it
                        print("[FAULT] Failed to drop object. Object stuck.", end="\r\n")
                        self.current_state = "ABORT_RETRACT"
                    else:
                        self.holding_object = False # Success
                        self.current_state = "ASCENDING_TARGET"
                else: # Intended to pick
                    if not actual_grip_state: # Tried to close, but grasped thin air
                        print("[FAULT] Failed to grip object. Thin air grasped.", end="\r\n")
                        self.holding_object = False # FSM internal state updated to empty
                        
                        # Open gripper cleanly before retreating to avoid snagging
                        self.arm.move_to(f"{active_loc}_bot_open", wait=True)
                        self.current_state = "ABORT_RETRACT"
                    else:
                        self.holding_object = True # Success
                        self.current_state = "ASCENDING_TARGET"

            elif self.current_state == "ASCENDING_TARGET":
                wp = self._get_waypoint_name(active_loc, "top")
                self.arm.move_to(wp, wait=True)
                
                self.arm.move_to(self._get_home_waypoint(), wait=True)
                self.target_task = None
                active_loc = None
                self.current_state = "HOME"

            # --- ESCAPE ROUTING ---
            elif self.current_state == "ABORT_RETRACT":
                if active_loc:
                    wp_top = self._get_waypoint_name(active_loc, "top")
                    self.arm.move_to(wp_top, wait=True)
                
                self.arm.move_to(self._get_home_waypoint(), wait=True)
                
                self.target_task = None
                active_loc = None
                self.current_state = "HOME"

            time.sleep(0.02) # Yield thread

    def _keyboard_listener_thread(self):
        """Reads raw keystrokes using select to avoid blocking shutdown."""
        while self.running:
            # Wait up to 0.1s for input, allowing loop to check self.running
            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                ch = sys.stdin.read(1)
                
                if ch == '1':
                    self.target_task = "A"
                    print("[Input] Command Received: Execute A", end="\r\n")
                elif ch == '2':
                    self.target_task = "B"
                    print("[Input] Command Received: Execute B", end="\r\n")
                elif ch == 'q':
                    self.target_task = "ABORT"
                    print("[Input] Command Received: ABORT!", end="\r\n")
                elif ch == '\x1b': # Escape key
                    self.running = False
                    print("[Input] Exit command received. Shutting down...", end="\r\n")
                    break

if __name__ == "__main__":
    arm = SO101ARM()
    controller = FSMController(arm)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("[System] Process interrupted by user.", end="\r\n")
        controller.running = False
