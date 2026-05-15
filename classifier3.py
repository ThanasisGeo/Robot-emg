import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


"""
Real-Time Event-Based EMG Gesture Classification using OpenBCI Cyton and BrainFlow

Description:
This script implements a real-time event-based surface electromyography (sEMG)
classification system using two EMG channels from an OpenBCI Cyton board.

The system monitors:
- Bicep muscle activity using Cyton EXG channel 1
- Forearm muscle activity using Cyton EXG channel 3

Unlike simple sample-by-sample threshold classifiers, this implementation detects
complete muscle activation events, extracts features from each event window, and
classifies the event after the contraction has finished.

System workflow:

1. Hardware initialization

2. EMG preprocessing

3. Rest calibration
   The user first performs a relaxed-muscle calibration phase.
   During this phase, the system estimates baseline envelope levels for both
   channels using median values.

   These baseline values are used to:
   - remove resting noise from later EMG samples
   - compute a total activation noise threshold
   - determine when a real muscle activation event begins and ends

4. Event detection
   Muscle activity is monitored using the combined cleaned activation:

       total_activation =
           max(bicep_envelope - bicep_rest_baseline, 0)
         + max(forearm_envelope - forearm_rest_baseline, 0)

   An EventCollector detects activation events using debounced timing rules:
   - An event starts only after activation stays above the noise threshold for
     the required ON duration.
   - An event ends only after activation remains below the threshold for the
     required OFF duration.

   This reduces false detections caused by short spikes, movement artifacts,
   and noisy EMG fluctuations.

5. Event-window collection
   For each detected contraction, the system stores the bicep and forearm EMG
   envelope samples across the entire event window.

   A short pre-event buffer is also included so the beginning of the contraction
   is not lost when debounce confirmation occurs.

6. Feature extraction
   After an event ends, the system extracts features from the captured event:

   - total_mean:
     average combined activation strength

   - total_rms:
     root mean square of the combined activation, representing event intensity

   - b_mean:
     average proportion of total activation contributed by the bicep channel

   - f_mean:
     average proportion of total activation contributed by the forearm channel

   These features describe both the strength and muscle contribution pattern of
   the contraction.

7. Feature-based supervised calibration
   The user performs repeated calibration contractions for:
   - bicep-only flexion
   - forearm-only flexion
   - simultaneous bicep and forearm flexion

   For each repetition, the system detects a complete event, extracts its
   features, and stores the resulting feature values.

   The final classification thresholds are computed from these calibration
   features:
   - total RMS threshold for identifying simultaneous contraction
   - bicep contribution threshold for identifying bicep-dominant events
   - forearm contribution threshold for identifying forearm-dominant events

8. Real-time event classification
   During live operation, each detected EMG event is classified into one of four
   output states:

   0 -> uncertain / rest
   1 -> forearm flex
   2 -> bicep flex
   3 -> both muscles flexed
   3.5 -> currently collecting an event

   Classification is performed only after an event is completed, using the
   extracted event features rather than isolated samples.

9. Live visualization
   The script displays real-time plots of:
   - bicep EMG envelope
   - forearm EMG envelope
   - classified state over time

   The state plot shows when the system is resting, collecting an event, or has
   classified a completed contraction.

Purpose:
This implementation is designed for robust EMG-based gesture recognition where
classification should be based on complete muscle activation patterns rather than
instantaneous threshold crossings.


Notes:
This approach is more robust than simple real-time thresholding because it
classifies complete contraction events using calibrated features. However, it
also introduces a short delay, since classification occurs after the event has
ended.
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
CALIBRATION_IGNORE_SEC = 1

WARMUP_SEC = 3

PLOT_WINDOW_SEC = 10

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


def reset_processors():
    return EMGProcessor(), EMGProcessor()

def warmup_processors(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    warmup_sec=0.5
):
    start = time.time()

    while time.time() - start < warmup_sec:
        data = board.get_board_data()

        if data.shape[1] == 0:
            time.sleep(0.005)
            continue

        for i in range(data.shape[1]):
            raw_bicep = data[bicep_channel, i]
            raw_forearm = data[forearm_channel, i]

            bicep_processor.process_sample(raw_bicep)
            forearm_processor.process_sample(raw_forearm)

        time.sleep(0.005)

bicep_processor = EMGProcessor()
forearm_processor = EMGProcessor()


# =====================
# Debounced activation detector
# =====================

class EventCollector:
    def __init__(self, fs, threshold, above_time_sec=0.2, below_time_sec=0.4):
        self.threshold = threshold

        self.above_required = int(above_time_sec * fs)
        self.below_required = int(below_time_sec * fs)

        self.in_event = False
        self.above_count = 0
        self.below_count = 0

        self.pre_bicep = deque(maxlen=self.above_required)
        self.pre_forearm = deque(maxlen=self.above_required)

        self.bicep_window = []
        self.forearm_window = []

    def update(self, bicep_env, forearm_env, total_activation):
        finished_event = None

        if not self.in_event:
            self.pre_bicep.append(bicep_env)
            self.pre_forearm.append(forearm_env)

            if total_activation > self.threshold:
                self.above_count += 1

                if self.above_count >= self.above_required:
                    self.in_event = True
                    self.below_count = 0

                    self.bicep_window = list(self.pre_bicep)
                    self.forearm_window = list(self.pre_forearm)

            else:
                self.above_count = 0

        else:
            self.bicep_window.append(bicep_env)
            self.forearm_window.append(forearm_env)

            if total_activation < self.threshold:
                self.below_count += 1

                if self.below_count >= self.below_required:
                    if self.below_required > 0:
                        bicep_final = self.bicep_window[:-self.below_required]
                        forearm_final = self.forearm_window[:-self.below_required]
                    else:
                        bicep_final = self.bicep_window
                        forearm_final = self.forearm_window

                    finished_event = (
                        np.array(bicep_final),
                        np.array(forearm_final)
                    )

                    self.in_event = False
                    self.above_count = 0
                    self.below_count = 0
                    self.bicep_window = []
                    self.forearm_window = []

            else:
                self.below_count = 0

        return finished_event
    


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
    ignore_sec=1.0
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



# =====================
# Rest calibration
# =====================

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

rest_b_med = np.median(rest_b)
rest_f_med = np.median(rest_f)

mean_bicep_noise = rest_b_med
mean_forearm_noise = rest_f_med

rest_total = (np.maximum(rest_b - mean_bicep_noise, 0) + np.maximum(rest_f - mean_forearm_noise, 0))

noise_threshold = np.mean(rest_total) + 3 * np.std(rest_total)

print("\nRest calibration complete.")
print("Event noise threshold:", noise_threshold)



# =====================
# Classification
# =====================

def compute_event_features(
    bicep_window,
    forearm_window,
    mean_bicep_noise,
    mean_forearm_noise
):
    b_clean = np.maximum(bicep_window - mean_bicep_noise, 0)
    f_clean = np.maximum(forearm_window - mean_forearm_noise, 0)

    total = b_clean + f_clean

    total_r = total + 1e-9

    b_ratio = b_clean / total_r
    f_ratio = f_clean / total_r

    total_mean = np.mean(total)
    total_rms = np.sqrt(np.mean(total ** 2))

    b_mean = np.mean(b_ratio)
    f_mean = np.mean(f_ratio)

    return {
        "total_mean": total_mean,
        "total_rms": total_rms,
        "b_mean": b_mean,
        "f_mean": f_mean
    }


def collect_feature_calibration(
    phase_name,
    repetitions,
    board,
    bicep_channel,
    forearm_channel,
    mean_bicep_noise,
    mean_forearm_noise,
    noise_threshold
):
    print(f"\n{phase_name}")

    total_rms_values = []
    b_mean_values = []
    f_mean_values = []

    for rep in range(repetitions):
        print(f"\nRepetition {rep + 1}/{repetitions}")
        print("Starting in 2 seconds...")
        time.sleep(2)
        board.get_board_data()

        bicep_processor = EMGProcessor()
        forearm_processor = EMGProcessor()
        warmup_processors(
            board,
            bicep_channel,
            forearm_channel,
            bicep_processor,
            forearm_processor,
            warmup_sec=0.5
        )

        board.get_board_data()
        collector = EventCollector(
            fs=FS,
            threshold=noise_threshold,
            above_time_sec=ON_TIME_SEC,
            below_time_sec=OFF_TIME_SEC
        )

        while True:
            data = board.get_board_data()

            if data.shape[1] == 0:
                time.sleep(0.005)
                continue

            event_found = False

            for i in range(data.shape[1]):
                raw_bicep = data[bicep_channel, i]
                raw_forearm = data[forearm_channel, i]

                bicep_env = bicep_processor.process_sample(raw_bicep)
                forearm_env = forearm_processor.process_sample(raw_forearm)

                b_clean_now = max(bicep_env - mean_bicep_noise, 0)
                f_clean_now = max(forearm_env - mean_forearm_noise, 0)

                total_activation = b_clean_now + f_clean_now

                finished_event = collector.update(
                    bicep_env,
                    forearm_env,
                    total_activation
                )

                if finished_event is not None:
                    bicep_window, forearm_window = finished_event

                    features = compute_event_features(
                        bicep_window,
                        forearm_window,
                        mean_bicep_noise,
                        mean_forearm_noise
                    )

                    total_rms_values.append(features["total_rms"])
                    b_mean_values.append(features["b_mean"])
                    f_mean_values.append(features["f_mean"])

                    print(features)

                    event_found = True
                    break

            if event_found:
                break

    return (
        np.array(total_rms_values),
        np.array(b_mean_values),
        np.array(f_mean_values)
    )



def classify_event(total_rms, b_mean, f_mean,
                   total_thresh=230,
                   b_thresh=0.4,
                   f_thresh=0.56):

    if total_rms > total_thresh: #and b_mean > b_thresh and f_mean > f_thresh
        return "both flexed", 3

    elif b_mean > b_thresh:
        return "bicep flex", 2

    elif f_mean > f_thresh:
        return "forearm flex", 1

    return "uncertain", 0



bicep_total_rms, bicep_b_mean, bicep_f_mean = collect_feature_calibration(
    phase_name="Flex ONLY bicep 5 times",
    repetitions=5,
    board=board,
    bicep_channel=bicep_channel,
    forearm_channel=forearm_channel,
    mean_bicep_noise=mean_bicep_noise,
    mean_forearm_noise=mean_forearm_noise,
    noise_threshold=noise_threshold
)

forearm_total_rms, forearm_b_mean, forearm_f_mean = collect_feature_calibration(
    phase_name="Flex ONLY forearm 5 times",
    repetitions=5,
    board=board,
    bicep_channel=bicep_channel,
    forearm_channel=forearm_channel,
    mean_bicep_noise=mean_bicep_noise,
    mean_forearm_noise=mean_forearm_noise,
    noise_threshold=noise_threshold
)

both_total_rms, both_b_mean, both_f_mean = collect_feature_calibration(
    phase_name="Flex BOTH muscles 5 times",
    repetitions=5,
    board=board,
    bicep_channel=bicep_channel,
    forearm_channel=forearm_channel,
    mean_bicep_noise=mean_bicep_noise,
    mean_forearm_noise=mean_forearm_noise,
    noise_threshold=noise_threshold
)

total_thresh = np.mean(both_total_rms) - np.std(both_total_rms)

b_thresh = np.mean(bicep_b_mean) - np.std(bicep_b_mean)

f_thresh = np.mean(forearm_f_mean) - np.std(forearm_f_mean)

print("\nFeature thresholds:")
print("total_thresh =", total_thresh)
print("b_thresh =", b_thresh)
print("f_thresh =", f_thresh)

bicep_processor, forearm_processor = reset_processors()
warmup_processors(
    board,
    bicep_channel,
    forearm_channel,
    bicep_processor,
    forearm_processor,
    warmup_sec=0.5
)

board.get_board_data()

event_collector = EventCollector(
    fs=FS,
    threshold=noise_threshold,
    above_time_sec=ON_TIME_SEC,
    below_time_sec=OFF_TIME_SEC
)

board.get_board_data()

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
axes[0].set_ylabel("Bicep")
axes[0].set_title("Bicep EMG Envelope")
axes[0].grid(True)
axes[0].legend()

line_forearm, = axes[1].plot([], [], label="Forearm Envelope")
axes[1].set_ylabel("Forearm")
axes[1].set_title("Forearm EMG Envelope")
axes[1].grid(True)
axes[1].legend()

line_state, = axes[2].plot([], [], label="State")
axes[2].set_ylim(-0.5, 4.0)
axes[2].set_yticks([0, 1, 2, 3, 3.5])
axes[2].set_yticklabels(["rest", "forearm", "bicep", "both", "collecting"])
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
current_state_name = "rest"
current_state_code = 0


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

            b_clean_now = max(bicep_env - mean_bicep_noise, 0)
            f_clean_now = max(forearm_env - mean_forearm_noise, 0)

            total_activation = b_clean_now + f_clean_now

            finished_event = event_collector.update(
                bicep_env,
                forearm_env,
                total_activation
            )

            if finished_event is not None:
                bicep_window, forearm_window = finished_event

                features = compute_event_features(
                    bicep_window,
                    forearm_window,
                    mean_bicep_noise,
                    mean_forearm_noise
                )

                current_state_name, current_state_code = classify_event(
                    features["total_rms"],
                    features["b_mean"],
                    features["f_mean"],
                    total_thresh=total_thresh,
                    b_thresh=b_thresh,
                    f_thresh=f_thresh
                )

                print("\nEvent classified:", current_state_name)
                print(features)

            elif event_collector.in_event:
                current_state_name = "collecting"
                current_state_code = 3.5
        #    else:
        #        if event_collector.in_event:
        #            current_state_name = "collecting"
        #            current_state_code = 3.5
        #        else:
        #            current_state_name = "rest"
        #            current_state_code = 0



            now = sample_counter / FS

            time_buffer.append(now)

            bicep_env_buffer.append(bicep_env)
            forearm_env_buffer.append(forearm_env)

            state_buffer.append(current_state_code)

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