import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


# =====================
# Settings
# =====================

SERIAL_PORT = "COM3"      # Windows example. Linux/Mac: "/dev/ttyUSB0" or "/dev/tty.usbserial..."
BOARD_ID = BoardIds.CYTON_BOARD.value

FS = 250
EMG_CHANNEL_NUMBER = 1    # 1 = first EXG channel, 2 = second EXG channel, etc.

LOWCUT = 20
HIGHCUT = 100
NOTCH_FREQ = 50
NOTCH_Q = 30

SMOOTH_SEC = 0.2
CALIBRATION_SEC = 5

PLOT_WINDOW_SEC = 10
THRESHOLD_K = 10
UPPER_FACTOR = 400

WARMUP_SEC = 3
CALIBRATION_IGNORE_SEC = 1


# =====================
# BrainFlow setup
# =====================

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(BOARD_ID, params)

board.prepare_session()
board.start_stream()

print("BrainFlow stream started.")

print(f"Warming up for {WARMUP_SEC} second...")

time.sleep(WARMUP_SEC)

# Clear anything collected during warm-up
board.get_board_data()

print("Warm-up complete.")

print(f"Relax your muscle for {CALIBRATION_SEC} seconds...")


# =====================
# Channel setup
# =====================

exg_channels = BoardShim.get_exg_channels(BOARD_ID)
emg_channel = exg_channels[EMG_CHANNEL_NUMBER - 1]

print("EXG channels:", exg_channels)
print("Using EMG channel:", emg_channel)


# =====================
# Filters
# =====================

def make_bandpass(fs, lowcut, highcut, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return b, a


bp_b, bp_a = make_bandpass(FS, LOWCUT, HIGHCUT)
notch_b, notch_a = iirnotch(NOTCH_FREQ, NOTCH_Q, FS)

bp_state = lfilter_zi(bp_b, bp_a) * 0
notch_state = lfilter_zi(notch_b, notch_a) * 0

smooth_samples = int(SMOOTH_SEC * FS)
smooth_buffer = deque(maxlen=smooth_samples)


# =====================
# Processing function
# =====================

def process_sample(raw_value):
    global bp_state, notch_state

    filtered, bp_state = lfilter(bp_b, bp_a, [raw_value], zi=bp_state)
    filtered, notch_state = lfilter(notch_b, notch_a, filtered, zi=notch_state)

    rectified = abs(filtered[0])

    smooth_buffer.append(rectified)
    envelope = np.mean(smooth_buffer)

    return envelope


# =====================
# Calibration
# =====================

calibration_values = []
start = time.time()

while time.time() - start < CALIBRATION_SEC:

    data = board.get_board_data()

    if data.shape[1] == 0:
        continue

    for i in range(data.shape[1]):

        raw_value = data[emg_channel, i]

        envelope = process_sample(raw_value)

        elapsed = time.time() - start

        # Ignore first second
        if elapsed > CALIBRATION_IGNORE_SEC:
            calibration_values.append(envelope)

    time.sleep(0.005)

rest_mean = np.mean(calibration_values)
rest_std = np.std(calibration_values)

lower_threshold = rest_mean + THRESHOLD_K * rest_std
upper_threshold = lower_threshold + UPPER_FACTOR * rest_std

if upper_threshold <= lower_threshold:
    upper_threshold = lower_threshold + 1e-6

print("Calibration complete.")
print("Rest mean:", rest_mean)
print("Rest std:", rest_std)
print("Lower threshold:", lower_threshold)
print("Upper threshold:", upper_threshold)


# =====================
# Live plot
# =====================

plt.ion()

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)

line_envelope, = axes[0].plot([], [], label="Processed EMG Envelope")
axes[0].axhline(lower_threshold, color="red", linestyle="--", label="ON Threshold")
#axes[0].axhline(OFF_THRESHOLD, color="orange", linestyle="--", label="OFF Threshold")
axes[0].set_ylabel("Envelope")
axes[0].set_title("Processed EMG Signal")
axes[0].grid(True)
axes[0].legend()

line_cont, = axes[1].plot([], [], label="Continuous Activation 0-1")
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_ylabel("Activation")
axes[1].set_title("Continuous Activation")
axes[1].grid(True)
axes[1].legend()

line_binary, = axes[2].plot([], [], label="Binary Activation")
axes[2].set_ylim(-0.1, 1.1)
axes[2].set_ylabel("ON/OFF")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Binary Activation")
axes[2].grid(True)
axes[2].legend()

time_buffer = deque(maxlen=int(PLOT_WINDOW_SEC * FS))
envelope_buffer = deque(maxlen=int(PLOT_WINDOW_SEC * FS))
cont_buffer = deque(maxlen=int(PLOT_WINDOW_SEC * FS))
binary_buffer = deque(maxlen=int(PLOT_WINDOW_SEC * FS))

sample_counter = 0

try:
    while True:
        data = board.get_board_data()

        if data.shape[1] == 0:
            plt.pause(0.001)
            continue

        for i in range(data.shape[1]):
            raw_value = data[emg_channel, i]

            envelope = process_sample(raw_value)

            if envelope > lower_threshold:
                binary_activation = 1
            else:
                binary_activation = 0

            cont_activation = (envelope - lower_threshold) / (upper_threshold - lower_threshold)
            cont_activation = np.clip(cont_activation, 0, 1)

            now = sample_counter / FS

            time_buffer.append(now)
            envelope_buffer.append(envelope)
            cont_buffer.append(cont_activation)
            binary_buffer.append(binary_activation)

            sample_counter += 1

        line_envelope.set_data(time_buffer, envelope_buffer)
        line_cont.set_data(time_buffer, cont_buffer)
        line_binary.set_data(time_buffer, binary_buffer)

        if len(time_buffer) > 1:
            axes[0].set_xlim(
                max(0, time_buffer[-1] - PLOT_WINDOW_SEC),
                time_buffer[-1]
            )

            axes[0].relim()
            axes[0].autoscale_view(scalex=False, scaley=True)

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    board.stop_stream()
    board.release_session()
    print("Session released.")