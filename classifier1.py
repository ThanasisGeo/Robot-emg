import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

"""
Real-Time EMG Muscle Activation Classification using OpenBCI Cyton + BrainFlow

Description:
This script performs real-time EMG acquisition, signal processing, and gesture/state
classification using two EMG channels from an OpenBCI Cyton board.

Muscle channels:
- Bicep (EXG channel 1)
- Forearm (EXG channel 3)

Processing pipeline:
1. Acquire raw EMG data from the Cyton board via BrainFlow.
2. Apply signal conditioning:
   - 20–100 Hz bandpass filter to isolate EMG frequencies
   - 50 Hz notch filter to suppress mains interference
   - full-wave rectification
   - moving-average smoothing to extract the EMG envelope
3. Perform baseline calibration while muscles are relaxed:
   - estimate resting mean and standard deviation
   - compute adaptive activation thresholds
4. Detect sustained muscle activation using debounced threshold logic
   to reduce false positives from transient spikes/noise.
5. Classify muscle states in real time into:
   - Rest
   - Forearm flex
   - Bicep flex
   - Both muscles flexed
6. Display live visualization of:
   - bicep EMG envelope
   - forearm EMG envelope
   - classified output state over time

Purpose:
This implementation is intended for real-time EMG-based human-machine interaction,
gesture recognition, prosthetic control experiments, or muscle activity monitoring.

"""

# =====================
# Settings
# =====================

SERIAL_PORT = "COM3"
BOARD_ID = BoardIds.CYTON_BOARD.value

FS = 250

BICEP_CHANNEL_NUMBER = 1      # Cyton EXG channel 1
FOREARM_CHANNEL_NUMBER = 3    # Cyton EXG channel 3

LOWCUT = 20
HIGHCUT = 100
NOTCH_FREQ = 50
NOTCH_Q = 30

SMOOTH_SEC = 0.2
CALIBRATION_SEC = 5
CALIBRATION_IGNORE_SEC = 1
WARMUP_SEC = 3

PLOT_WINDOW_SEC = 10

BICEP_HIGH_K = 10

FOREARM_LOW_K = 3
FOREARM_HIGH_K = 10

ON_TIME_SEC = 0.2
OFF_TIME_SEC = 0.4


# =====================
# BrainFlow setup
# =====================

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(BOARD_ID, params)

board.prepare_session()
board.start_stream()

print("BrainFlow stream started.")
print(f"Warming up for {WARMUP_SEC} seconds...")

time.sleep(WARMUP_SEC)
board.get_board_data()

print("Warm-up complete.")


# =====================
# Channel setup
# =====================

exg_channels = BoardShim.get_exg_channels(BOARD_ID)

bicep_channel = exg_channels[BICEP_CHANNEL_NUMBER - 1]
forearm_channel = exg_channels[FOREARM_CHANNEL_NUMBER - 1]

print("EXG channels:", exg_channels)
print("Using bicep channel:", bicep_channel)
print("Using forearm channel:", forearm_channel)


# =====================
# Filters
# =====================

def make_bandpass(fs, lowcut, highcut, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return b, a


bp_b, bp_a = make_bandpass(FS, LOWCUT, HIGHCUT)
notch_b, notch_a = iirnotch(NOTCH_FREQ, NOTCH_Q, FS)


class EMGProcessor:
    def __init__(self):
        self.bp_state = lfilter_zi(bp_b, bp_a) * 0
        self.notch_state = lfilter_zi(notch_b, notch_a) * 0
        self.smooth_buffer = deque(maxlen=int(SMOOTH_SEC * FS))

    def process_sample(self, raw_value):
        filtered, self.bp_state = lfilter(
            bp_b,
            bp_a,
            [raw_value],
            zi=self.bp_state
        )

        filtered, self.notch_state = lfilter(
            notch_b,
            notch_a,
            filtered,
            zi=self.notch_state
        )

        rectified = abs(filtered[0])

        self.smooth_buffer.append(rectified)
        envelope = np.mean(self.smooth_buffer)

        return envelope


bicep_processor = EMGProcessor()
forearm_processor = EMGProcessor()


# =====================
# Debounced activation detector
# =====================

class ActivationDetector:
    def __init__(self, fs, on_time_sec=0.2, off_time_sec=0.4):
        self.on_required = int(on_time_sec * fs)
        self.off_required = int(off_time_sec * fs)

        self.active = False
        self.on_count = 0
        self.off_count = 0

    def update(self, value, threshold):
        if not self.active:
            if value > threshold:
                self.on_count += 1

                if self.on_count >= self.on_required:
                    self.active = True
                    self.off_count = 0
            else:
                self.on_count = 0

        else:
            if value < threshold:
                self.off_count += 1

                if self.off_count >= self.off_required:
                    self.active = False
                    self.on_count = 0
                    self.off_count = 0
            else:
                self.off_count = 0

        return self.active


bicep_high_detector = ActivationDetector(FS, ON_TIME_SEC, OFF_TIME_SEC)

forearm_low_detector = ActivationDetector(FS, ON_TIME_SEC, OFF_TIME_SEC)
forearm_high_detector = ActivationDetector(FS, ON_TIME_SEC, OFF_TIME_SEC)


# =====================
# Calibration
# =====================

print(f"Relax both muscles for {CALIBRATION_SEC} seconds...")

bicep_calibration = []
forearm_calibration = []

start = time.time()

while time.time() - start < CALIBRATION_SEC:

    data = board.get_board_data()

    if data.shape[1] == 0:
        continue

    for i in range(data.shape[1]):

        raw_bicep = data[bicep_channel, i]
        raw_forearm = data[forearm_channel, i]

        bicep_env = bicep_processor.process_sample(raw_bicep)
        forearm_env = forearm_processor.process_sample(raw_forearm)

        elapsed = time.time() - start

        if elapsed > CALIBRATION_IGNORE_SEC:
            bicep_calibration.append(bicep_env)
            forearm_calibration.append(forearm_env)

    time.sleep(0.005)


bicep_rest_mean = np.mean(bicep_calibration)
bicep_rest_std = np.std(bicep_calibration)

forearm_rest_mean = np.mean(forearm_calibration)
forearm_rest_std = np.std(forearm_calibration)

bicep_high_threshold = bicep_rest_mean + BICEP_HIGH_K * bicep_rest_std

forearm_low_threshold = forearm_rest_mean + FOREARM_LOW_K * forearm_rest_std
forearm_high_threshold = forearm_rest_mean + FOREARM_HIGH_K * forearm_rest_std


print("Calibration complete.")

print("\nBicep:")
print("Rest mean:", bicep_rest_mean)
print("Rest std:", bicep_rest_std)
print("High threshold:", bicep_high_threshold)

print("\nForearm:")
print("Rest mean:", forearm_rest_mean)
print("Rest std:", forearm_rest_std)
print("Low threshold:", forearm_low_threshold)
print("High threshold:", forearm_high_threshold)


# =====================
# Classification
# =====================

def classify_state(bicep_high, forearm_low, forearm_high):
    if bicep_high:
        if forearm_high:
            return "both flexed", 3
        else:
            return "bicep flex", 2

    elif forearm_low:
        return "forearm flex", 1

    else:
        return "rest", 0


# =====================
# Live plot
# =====================

plt.ion()

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 12),
    sharex=True
)

line_bicep, = axes[0].plot([], [], label="Bicep Envelope")
axes[0].axhline(bicep_high_threshold, color="red", linestyle="--", label="Bicep High Threshold")
axes[0].set_ylabel("Bicep")
axes[0].set_title("Bicep EMG Envelope")
axes[0].grid(True)
axes[0].legend()

line_forearm, = axes[1].plot([], [], label="Forearm Envelope")
axes[1].axhline(forearm_low_threshold, color="orange", linestyle="--", label="Forearm Low Threshold")
axes[1].axhline(forearm_high_threshold, color="red", linestyle="--", label="Forearm High Threshold")
axes[1].set_ylabel("Forearm")
axes[1].set_title("Forearm EMG Envelope")
axes[1].grid(True)
axes[1].legend()

line_state, = axes[2].plot([], [], label="State")
axes[2].set_ylim(-0.5, 3.5)
axes[2].set_yticks([0, 1, 2, 3])
axes[2].set_yticklabels(["rest", "forearm", "bicep", "both"])
axes[2].set_ylabel("Class")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Real-Time Classification")
axes[2].grid(True)
axes[2].legend()


buffer_len = int(PLOT_WINDOW_SEC * FS)

time_buffer = deque(maxlen=buffer_len)

bicep_env_buffer = deque(maxlen=buffer_len)
forearm_env_buffer = deque(maxlen=buffer_len)

state_buffer = deque(maxlen=buffer_len)

sample_counter = 0
last_state = None


try:
    while True:
        data = board.get_board_data()

        if data.shape[1] == 0:
            plt.pause(0.001)
            continue

        for i in range(data.shape[1]):

            raw_bicep = data[bicep_channel, i]
            raw_forearm = data[forearm_channel, i]

            bicep_env = bicep_processor.process_sample(raw_bicep)
            forearm_env = forearm_processor.process_sample(raw_forearm)

            bicep_high = bicep_high_detector.update(
                bicep_env,
                bicep_high_threshold
            )

            forearm_low = forearm_low_detector.update(
                forearm_env,
                forearm_low_threshold
            )

            forearm_high = forearm_high_detector.update(
                forearm_env,
                forearm_high_threshold
            )

            state_name, state_code = classify_state(
                bicep_high,
                forearm_low,
                forearm_high
            )

            if state_name != last_state:
                print(state_name)
                last_state = state_name

            now = sample_counter / FS

            time_buffer.append(now)

            bicep_env_buffer.append(bicep_env)
            forearm_env_buffer.append(forearm_env)

            state_buffer.append(state_code)

            sample_counter += 1

        line_bicep.set_data(time_buffer, bicep_env_buffer)
        line_forearm.set_data(time_buffer, forearm_env_buffer)

        line_state.set_data(time_buffer, state_buffer)

        if len(time_buffer) > 1:
            x_min = max(0, time_buffer[-1] - PLOT_WINDOW_SEC)
            x_max = time_buffer[-1]

            for ax in axes:
                ax.set_xlim(x_min, x_max)

            axes[0].relim()
            axes[0].autoscale_view(scalex=False, scaley=True)

            axes[1].relim()
            axes[1].autoscale_view(scalex=False, scaley=True)

        plt.pause(0.001)


except KeyboardInterrupt:
    print("Stopping...")

finally:
    board.stop_stream()
    board.release_session()
    print("Session released.")