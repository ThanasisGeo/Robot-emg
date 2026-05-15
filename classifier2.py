import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


"""
Real-Time Multi-State EMG Classification using OpenBCI Cyton and BrainFlow

Description:
This script implements a real-time surface electromyography (sEMG) acquisition,
signal processing, calibration, and multi-state muscle activation classification
system using an OpenBCI Cyton board.

Two EMG channels are monitored:
- Bicep muscle (EXG channel 1)
- Forearm muscle (EXG channel 3)

System workflow:

1. Hardware initialization
   - Connects to the OpenBCI Cyton board through BrainFlow
   - Starts continuous EMG data streaming
   - Performs an initial warm-up period to allow:
       * filter stabilization
       * board buffer flushing
       * electrode-skin interface stabilization

2. Real-time EMG preprocessing

3. Multi-phase supervised calibration
   The user performs four calibration phases:
   - Rest (both muscles relaxed)
   - Bicep-only contraction
   - Forearm-only contraction
   - Simultaneous bicep + forearm contraction

   During each phase:
   - initial transient samples are ignored
   - EMG envelope statistics are collected
   - median activation levels are estimated

   These calibration phases provide subject-specific activation signatures for
   robust threshold generation.

4. Adaptive threshold estimation
   Classification thresholds are computed from calibration medians:

   - Bicep high threshold:
     separates true bicep activation from forearm cross-talk

   - Forearm low threshold:
     separates forearm activation from resting baseline

   - Forearm high threshold:
     separates isolated forearm activation from simultaneous co-contraction

   Safety constraints enforce minimum threshold margins above resting noise
   to reduce false triggering.

5. Debounced activation detection
   Activation decisions require sustained threshold crossings:
   - ON delay: 0.2 seconds
   - OFF delay: 0.4 seconds

   This hysteresis-like debouncing reduces false positives caused by transient
   spikes, movement artifacts, and signal fluctuations.

6. Real-time classification
   The system continuously classifies EMG activity into four states:

   0 -> Rest
   1 -> Forearm flex
   2 -> Bicep flex
   3 -> Both muscles flexed

   Classification logic combines threshold detector outputs from both channels.

7. Live visualization
   Real-time plots display:
   - bicep EMG envelope with activation threshold
   - forearm EMG envelope with low/high thresholds
   - classified output state over time


Notes:
Thresholds are user-specific and recalibrated each run, improving robustness
against electrode placement changes, signal amplitude variation, and
inter-subject physiological differences.
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
CALIBRATION_SEC = 4
CALIBRATION_IGNORE_SEC = 0.5

WARMUP_SEC = 3

PLOT_WINDOW_SEC = 10

REST_MARGIN_STD = 3

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

def collect_calibration_phase(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    phase_name,
    duration_sec,
    ignore_sec=0.5
):
    print(f"\n{phase_name}")
    print(f"Starting in 2 seconds...")
    time.sleep(2)
    board.get_board_data()

    bicep_values = []
    forearm_values = []

    start = time.time()

    while time.time() - start < duration_sec:
        data = board.get_board_data()

        if data.shape[1] == 0:
            time.sleep(0.005)
            continue

        for i in range(data.shape[1]):
            raw_bicep = data[bicep_channel, i]
            raw_forearm = data[forearm_channel, i]

            bicep_env = bicep_processor.process_sample(raw_bicep)
            forearm_env = forearm_processor.process_sample(raw_forearm)

            elapsed = time.time() - start

            if elapsed > ignore_sec:
                bicep_values.append(bicep_env)
                forearm_values.append(forearm_env)

        time.sleep(0.005)

    return np.array(bicep_values), np.array(forearm_values)



rest_b, rest_f = collect_calibration_phase(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    phase_name=f"Relax both muscles for {CALIBRATION_SEC} seconds...",
    duration_sec=CALIBRATION_SEC,
    ignore_sec=CALIBRATION_IGNORE_SEC
)

bicep_only_b, bicep_only_f = collect_calibration_phase(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    phase_name=f"Flex ONLY the bicep for {CALIBRATION_SEC} seconds...",
    duration_sec=CALIBRATION_SEC,
    ignore_sec=CALIBRATION_IGNORE_SEC
)

forearm_only_b, forearm_only_f = collect_calibration_phase(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    phase_name=f"Flex ONLY the forearm for {CALIBRATION_SEC} seconds...",
    duration_sec=CALIBRATION_SEC,
    ignore_sec=CALIBRATION_IGNORE_SEC
)

both_b, both_f = collect_calibration_phase(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    phase_name=f"Flex BOTH bicep and forearm for {CALIBRATION_SEC} seconds...",
    duration_sec=CALIBRATION_SEC,
    ignore_sec=CALIBRATION_IGNORE_SEC
)

####################### OR MEAN HERE? #######################
rest_b_med = np.median(rest_b)
rest_f_med = np.median(rest_f)

rest_b_std = np.std(rest_b)
rest_f_std = np.std(rest_f)

bicep_only_b_med = np.median(bicep_only_b)
bicep_only_f_med = np.median(bicep_only_f)

forearm_only_b_med = np.median(forearm_only_b)
forearm_only_f_med = np.median(forearm_only_f)

both_b_med = np.median(both_b)
both_f_med = np.median(both_f)


bicep_high_threshold = 0.5 * (forearm_only_b_med + bicep_only_b_med)

forearm_low_threshold = 0.5 * (rest_f_med + forearm_only_f_med)

forearm_high_threshold = 0.5 * (bicep_only_f_med + both_f_med)


# Safety: prevent thresholds from being too close to rest
bicep_min_threshold = rest_b_med + REST_MARGIN_STD * rest_b_std
forearm_min_threshold = rest_f_med + REST_MARGIN_STD * rest_f_std

bicep_high_threshold = max(bicep_high_threshold, bicep_min_threshold)
forearm_low_threshold = max(forearm_low_threshold, forearm_min_threshold)
forearm_high_threshold = max(forearm_high_threshold, forearm_low_threshold + forearm_min_threshold)
# forearm_high_threshold = max(forearm_high_threshold, forearm_low_threshold * 1.5)


print("\nCalibration complete.")

print("\nMedian envelopes:")
print("Rest:          bicep =", rest_b_med, "forearm =", rest_f_med)
print("Bicep only:    bicep =", bicep_only_b_med, "forearm =", bicep_only_f_med)
print("Forearm only:  bicep =", forearm_only_b_med, "forearm =", forearm_only_f_med)
print("Both:          bicep =", both_b_med, "forearm =", both_f_med)

print("\nThresholds:")
print("Bicep high threshold:", bicep_high_threshold)
print("Forearm low threshold:", forearm_low_threshold)
print("Forearm high threshold:", forearm_high_threshold)


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