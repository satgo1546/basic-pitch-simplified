#!/usr/bin/env python3
# encoding: utf-8

# This module is comprised of PyTorch layers from NNAudio and ported to TensorFlow:
# https://github.com/KinWaiCheuk/nnAudio
# The above code is released under an MIT license.

import os
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.io, scipy.signal
import tensorflow as tf
import pretty_midi


FFT_HOP = 256
N_FFT = 8 * FFT_HOP

CONTOURS_BINS_PER_SEMITONE = 3
# base frequency of the CENTRAL bin of the first semitone (i.e., the
# second bin if annotations_bins_per_semitone is 3)
ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
AUDIO_SAMPLE_RATE = 22050
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP

# ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

# AUDIO_N_SAMPLES is the number of samples in the (clipped) audio that we use as input to the models
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

MIDI_OFFSET = 21
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87

def midi_to_hz(notes: ArrayLike):
    return 440 * 2 ** ((notes - 69) / 12)


def hz_to_midi(frequencies: ArrayLike):
    return 12 * (np.log2(frequencies) - np.log2(440)) + 69


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
) -> Tuple[NDArray, int, NDArray, NDArray]:
    """Automatically create CQT kernels in time domain
    """

    next_power_of_2 = math.ceil(np.log2(np.ceil(Q * fs / fmin)))
    fftLen = 2 ** next_power_of_2
    freqs = fmin * 2 ** (np.arange(n_bins) / bins_per_octave)
    tempKernel = np.zeros((n_bins, fftLen), dtype=np.complex64)
    lengths = np.ceil(Q * fs / freqs)
    for k in range(n_bins):
        freq = freqs[k]
        _l = math.ceil(Q * fs / freq)

        # Centering the kernels, pad more zeros on RHS
        start = math.ceil(fftLen / 2 - _l / 2) - _l % 2

        sig = (
            scipy.signal.get_window("hann", _l, fftbins=True)
            * np.exp(2j * np.pi * np.r_[-_l // 2 : _l // 2] * freq / fs)
            / _l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + _l] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + _l] = sig

    return tempKernel, fftLen, lengths, freqs


class CQT(tf.keras.layers.Layer):
    """This layer calculates the CQT of the input signal.
    Input signal should be in either of the following shapes.
    1. (len_audio)
    2. (num_audio, len_audio)
    3. (num_audio, 1, len_audio)
    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.

    This layer uses about 1MB of memory per second of input audio with its default arguments.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.
    hop_length : int
        The hop (or stride) size. Default value is 512.
    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.
    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``
    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.
    bins_per_octave : int
        Number of bins per octave. Default is 12.
    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.
    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``
    Returns
    -------
    spectrogram : tf.Tensor

    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)``.
    """

    def __init__(
        self,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        basis_norm: int = 1,
    ):
        super().__init__()

        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.basis_norm = basis_norm

    def build(self, input_shape: tf.TensorShape) -> None:
        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = 1 / (2 ** (1 / self.bins_per_octave) - 1)

        # This command produces the filter kernel coefficients
        self.lowpass_filter = scipy.signal.firwin2(
            256,  # kernel_length
            # We specify a list of key frequencies for which we will require
            # that the filter match a specific output gain.
            # From [0.0 to passband_max] is the frequency range we want to keep
            # untouched and [stopband_min, 1.0] is the range we want to remove
            [
                0.0,
                # band_center = 0.5; transition_bandwidth = 0.001
                0.5 / 1.001,  # passband_max
                0.5 * 1.001,  # stopband_min
                1.0,
            ],
            # We specify a list of output gains to correspond to the key
            # frequencies listed above.
            # The first two gains are 1.0 because they correspond to the first
            # two key frequencies. the second two are 0.0 because they
            # correspond to the stopband frequencies
            [1.0, 1.0, 0.0, 0.0],
        ).astype(np.float32)

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(self.bins_per_octave, self.n_bins)
        self.n_octaves = int(np.ceil(float(self.n_bins) / self.bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = ANNOTATIONS_BASE_FREQUENCY * 2 ** (self.n_octaves - 1)
        remainder = self.n_bins % self.bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((self.bins_per_octave - 1) / self.bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / self.bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (1 - 1 / self.bins_per_octave)  # Adjusting the top minium bins

        # Preparing CQT kernels
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            AUDIO_SAMPLE_RATE,
            self.fmin_t,
            n_filters,
            self.bins_per_octave,
            norm=self.basis_norm,
        )

        # For the normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = ANNOTATIONS_BASE_FREQUENCY * 2 ** (np.arange(self.n_bins) / self.bins_per_octave)
        self.frequencies = freqs

        self.lengths = np.ceil(Q * AUDIO_SAMPLE_RATE / freqs)

        self.basis = basis
        # NOTE(psobot): this is where the implementation here starts to differ from CQT2010.

        # These cqt_kernel is already in the frequency domain
        self.cqt_kernels_real = tf.expand_dims(basis.real.astype(self.dtype), 1)
        self.cqt_kernels_imag = tf.expand_dims(basis.imag.astype(self.dtype), 1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = x[:, None, :]

        hop = FFT_HOP

        # Getting the top octave CQT
        CQTs = []
        for i in range(self.n_octaves):
            if i:
                hop //= 2
                """Downsample the given tensor using the given filter kernel.
                The input tensor is expected to have shape `(n_batches, channels, width)`,
                and the filter kernel is expected to have shape `(num_output_channels,)` (i.e.: 1D)

                If match_torch_exactly is passed, we manually pad the input rather than having TensorFlow do so with "SAME".
                The result is subtly different than Torch's output, but it is compatible with TensorFlow Lite (as of v2.4.1).
                """
                x = tf.pad(x, [
                    [0, 0],
                    [0, 0],
                    [(self.lowpass_filter.shape[-1] - 1) // 2, (self.lowpass_filter.shape[-1] - 1) // 2],
                ])
                # Store this tensor in the shape `(n_batches, width, channels)`
                x = tf.transpose(x, [0, 2, 1])
                x = tf.nn.conv1d(x, self.lowpass_filter[:, None, None], padding="VALID", stride=2)
                x = tf.transpose(x, [0, 2, 1])
            """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
            for how to multiple the STFT result with the CQT kernel
            [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
            a constant Q transform.” (1992)."""

            # When center is True, the STFT window will be put in the middle, and paddings at the beginning and ending are required.
            padded = tf.pad(x, [[0, 0], [0, 0], [self.n_fft // 2, self.n_fft // 2]], "REFLECT")
            CQT_real = tf.transpose(
                tf.nn.conv1d(
                    tf.transpose(padded, [0, 2, 1]),
                    tf.transpose(self.cqt_kernels_real, [2, 1, 0]),
                    padding="VALID",
                    stride=hop,
                ),
                [0, 2, 1],
            )
            CQT_imag = -tf.transpose(
                tf.nn.conv1d(
                    tf.transpose(padded, [0, 2, 1]),
                    tf.transpose(self.cqt_kernels_imag, [2, 1, 0]),
                    padding="VALID",
                    stride=hop,
                ),
                [0, 2, 1],
            )
            CQTs.insert(
                0,
                tf.stack((CQT_real, CQT_imag), axis=-1)
            )
        CQT = tf.concat(CQTs, axis=1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins

        # Normalize again to get same result as librosa
        CQT *= tf.math.sqrt(tf.cast(self.lengths.reshape((-1, 1, 1)), self.dtype))

        # Transpose the output to match the output of the other spectrogram layers.
        # Getting CQT Amplitude
        return tf.transpose(tf.norm(CQT, axis=-1), [0, 2, 1])


def get_model() -> tf.keras.Model:
    """Basic Pitch's model implementation.
    """
    # The number of harmonics to use in the harmonic stacking layer.
    n_harmonics = 8
    # input representation
    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))  # (batch, time, ch)
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = min(
        math.ceil(12 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES,
        math.floor(12 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)),
    )
    x = tf.squeeze(inputs, -1)
    x = CQT(
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    """
    Take an input with a shape of (batch, y, z) and rescale each (y, z) to dB, scaled 0 - 1.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """
    # square to convert magnitude to power
    x = 10 * tf.math.log(tf.math.square(x) + 1e-10)
    x -= tf.reshape(tf.math.reduce_min(x, axis=[1, 2]), [tf.shape(x)[0], 1, 1])
    x = tf.math.divide_no_nan(x, tf.reshape(
        tf.math.reduce_max(x, axis=[1, 2]),
        [tf.shape(x)[0], 1, 1],
    ))
    x = tf.expand_dims(x, -1)
    # Apply batch normalization after computing the CQT.
    x = tf.keras.layers.BatchNormalization()(x)

    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """
    tf.debugging.assert_equal(tf.shape(x).shape, 4)  # (n_batch, n_times, n_freqs, 1)
    channels = []
    for shift in [
        round(12 * CONTOURS_BINS_PER_SEMITONE * np.log2(h))
        for h in [0.5] + list(range(1, n_harmonics))
    ]:
        if shift == 0:
            padded = x
        elif shift > 0:
            paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
            padded = tf.pad(x[:, :, shift:, :], paddings)
        elif shift < 0:
            paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
            padded = tf.pad(x[:, :, :shift, :], paddings)
        channels.append(padded)
    x = tf.concat(channels, axis=-1)
    x = x[:, :, : N_FREQ_BINS_CONTOURS, :]  # return only the first n_output_freqs frequency channels

    # contour layers - fully convolutional
    x_contours = tf.keras.layers.Conv2D(32, (5, 5), padding="same")(x)

    x_contours = tf.keras.layers.BatchNormalization()(x_contours)
    x_contours = tf.nn.relu(x_contours)

    x_contours = tf.keras.layers.Conv2D(8, (3, 3 * 13), padding="same")(x)

    x_contours = tf.keras.layers.BatchNormalization()(x_contours)
    x_contours = tf.nn.relu(x_contours)

    x_contours = tf.keras.layers.Conv2D(1, (5, 5), padding="same")(x_contours)
    x_contours = tf.sigmoid(x_contours)
    x_contours = tf.squeeze(x_contours, -1)  # contour output

    # reduced contour output as input to notes
    x_contours_reduced = tf.expand_dims(x_contours, -1)

    x_contours_reduced = tf.keras.layers.Conv2D(
        32, (7, 7), padding="same", strides=(1, 3)
    )(x_contours_reduced)
    x_contours_reduced = tf.nn.relu(x_contours_reduced)

    # note output layer
    x_notes_pre = tf.keras.layers.Conv2D(1, (7, 3), padding="same", activation="sigmoid")(
        x_contours_reduced
    )
    x_notes = tf.squeeze(x_notes_pre, -1)

    # onset output layer

    # onsets - fully convolutional
    x_onset = tf.keras.layers.Conv2D(32, (5, 5), padding="same", strides=(1, 3))(x)
    x_onset = tf.keras.layers.BatchNormalization()(x_onset)
    x_onset = tf.nn.relu(x_onset)
    x_onset = tf.concat([x_notes_pre, x_onset], axis=3)
    x_onset = tf.keras.layers.Conv2D(1, (3, 3), padding="same")(x_onset)
    x_onset = tf.sigmoid(x_onset)
    x_onset = tf.squeeze(x_onset, -1)

    outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def unwrap_output(output: tf.Tensor, audio_original_length: int, n_overlapping_frames: int) -> NDArray:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    raw_output = output.numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length


def predict(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> Tuple[Dict[str, NDArray], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
    """Run a single prediction.

    Args:
        audio_path: File path for the audio to run inference on.
        model_or_model_path: Path to load the Keras saved model from. Can be local or on GCS.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in frames.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
    Returns:
        The model output, midi data and note events from a single prediction
    """
    model = get_model()
    model.load_weights(os.path.join(os.path.dirname(__file__), "nmp"))

    """Run the model on the input audio path.

        model: A loaded keras model to run inference with.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.
    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original_samplerate, audio_original = scipy.io.wavfile.read(audio_path)
    audio_original = scipy.signal.resample(
        (
            np.float32(audio_original)
            if audio_original.ndim == 1
            else np.mean(audio_original, axis=1, dtype=np.float32)
        )
        / (
            np.iinfo(audio_original.dtype).max
            if np.issubdtype(audio_original.dtype, np.integer)
            else 1
        ),
        int(len(audio_original) * AUDIO_SAMPLE_RATE / audio_original_samplerate),
        window=("kaiser", 12.984585247040012),
    )
    audio_original_length = len(audio_original)
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
    """
    audio_windowed = tf.expand_dims(
        tf.signal.frame(
            np.pad(audio_original, (overlap_len // 2, 0)),
            AUDIO_N_SAMPLES,
            hop_size,
            pad_end=True,
            pad_value=0,
        ),
        axis=-1,
    )

    model_output = model(audio_windowed)
    model_output = {k: unwrap_output(model_output[k], audio_original_length, n_overlapping_frames) for k in model_output}

    """Convert model output to MIDI

    Args:
        output: A dictionary with shape
            {
                'frame': array of shape (n_times, n_freqs),
                'onset': array of shape (n_times, n_freqs),
                'contour': array of shape (n_times, 3*n_freqs)
            }
            representing the output of the basic pitch model.
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        min_note_len: The minimum allowed note length in frames.
        min_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        max_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        include_pitch_bends: If True, include pitch bends.

    Returns:
        midi : pretty_midi.PrettyMIDI object
        note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
    """
    frames = model_output["note"]
    onsets = model_output["onset"]
    contours = model_output["contour"]

    # convert minimum_note_length to frames
    min_note_len = round(minimum_note_length / 1000 * AUDIO_SAMPLE_RATE / FFT_HOP)
    note_events = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
    )
    """Given note events and contours, estimate pitch bends per note.
    Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
    The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
    the start and end times of each note event.

    Args:
        contours: Matrix of estimated pitch contours
        note_events: note event tuple
        n_bins_tolerance: Pitch bend estimation range. Defaults to 25.

    Returns:
        note events with pitch bends
    """
    # Convert model frames to time.
    times_s = np.arange(contours.shape[0])
    # Here is a magic number, but it's needed to align properly.
    times_s = times_s * FFT_HOP / AUDIO_SAMPLE_RATE - times_s // ANNOT_N_FRAMES * (FFT_HOP / AUDIO_SAMPLE_RATE * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) + 0.0018)
    n_bins_tolerance = 25
    freq_gaussian = scipy.signal.gaussian(n_bins_tolerance * 2 + 1, std=5)
    for i, (start_idx, end_idx, pitch_midi, amplitude) in enumerate(note_events):
        # Convert midi pitch to conrresponding index in contour matrix.
        freq_idx = round(12 * CONTOURS_BINS_PER_SEMITONE * np.log2(midi_to_hz(pitch_midi) / ANNOTATIONS_BASE_FREQUENCY))
        freq_start_idx = max(freq_idx - n_bins_tolerance, 0)
        freq_end_idx = min(N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1)

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                max(0, n_bins_tolerance - freq_idx) : len(freq_gaussian)
                - max(0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1))
            ]
        )
        pb_shift = n_bins_tolerance - max(0, n_bins_tolerance - freq_idx)

        note_events[i] = (
            times_s[start_idx],
            times_s[end_idx],
            pitch_midi,
            amplitude,
            # this is in units of 1/3 semitones
            np.argmax(pitch_bend_submatrix, axis=1) - pb_shift,
        )
    """Create a pretty_midi object from note events

    Args:
        note_events : list of tuples [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            where amplitude is a number between 0 and 1

    Returns:
        pretty_midi.PrettyMIDI() object
    """
    midi_data = pretty_midi.PrettyMIDI()

    """Drop pitch bends from any notes that overlap in time with another note"""
    note_events = sorted(note_events)
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:  # start j > end i
                break
            note_events[i] = note_events[i][:-1] + (None,)  # last field is pitch bend
            note_events[j] = note_events[j][:-1] + (None,)

    instrument = pretty_midi.Instrument(
        pretty_midi.instrument_name_to_program("Electric Piano 1")
    )
    for start_time, end_time, note_number, amplitude, pitch_bend in note_events:
        note = pretty_midi.Note(
            velocity=int(np.round(127 * amplitude)),
            pitch=note_number,
            start=start_time,
            end=end_time,
        )
        instrument.notes.append(note)
        if not pitch_bend:
            continue
        pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
        pitch_bend_midi_ticks = np.round(np.array(pitch_bend) * 4096 / CONTOURS_BINS_PER_SEMITONE).astype(int)
        # This supports pitch bends up to 2 semitones
        # If we estimate pitch bends above/below 2 semitones, crop them here when adding them to the midi file
        pitch_bend_midi_ticks[pitch_bend_midi_ticks > N_PITCH_BEND_TICKS - 1] = N_PITCH_BEND_TICKS - 1
        pitch_bend_midi_ticks[pitch_bend_midi_ticks < -N_PITCH_BEND_TICKS] = -N_PITCH_BEND_TICKS
        for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend_midi_ticks):
            instrument.pitch_bends.append(pretty_midi.PitchBend(pb_midi, pb_time))
    midi_data.instruments.append(instrument)

    return model_output, midi_data, note_events


def constrain_frequency(
    x: NDArray, max_freq: Optional[float], min_freq: Optional[float]
):
    """Zero out activations above or below the max/min frequencies

    Args:
        x: Onset/frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep, in Hz.
        min_freq: the minimum frequency to keep, in Hz.
    """
    if max_freq is not None:
        x[:, round(hz_to_midi(max_freq)) - MIDI_OFFSET :] = 0
    if min_freq is not None:
        x[:, : round(hz_to_midi(min_freq)) - MIDI_OFFSET] = 0


def output_to_notes_polyphonic(
    frames: NDArray,
    onsets: NDArray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    max_freq: Optional[float],
    min_freq: Optional[float],
    energy_tol: int = 11,
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        energy_tol: Drop notes below this energy.

    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1
    """

    n_frames = frames.shape[0]

    constrain_frequency(onsets, max_freq, min_freq)
    constrain_frequency(frames, max_freq, min_freq)
    """Infer onsets from large changes in frame amplitudes, in addition to the predicted onsets.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    # Number of differences used to detect onsets.
    n_diff = 2
    frame_diff = np.min(
        [frames - np.roll(frames, n + 1, axis=0) for n in range(n_diff)], axis=0
    )
    frame_diff = np.clip(frame_diff, 0, None)
    frame_diff[:n_diff, :] = 0
    # Rescale to have the same maximum as onsets.
    frame_diff *= np.max(onsets) / np.max(frame_diff)
    # use the max of the predicted onsets and the differences
    # The maximum between the predicted onsets and its differences.
    onsets = np.maximum(onsets, frame_diff)

    peak_thresh_mat = np.zeros_like(onsets)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = frames >= frame_thresh

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx]:
                k = 0
            else:
                k += 1
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # add the note
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + MIDI_OFFSET,
                amplitude,
            )
        )

    while np.any(remaining_energy):
        i_start, freq_idx = np.unravel_index(np.argmax(remaining_energy), remaining_energy.shape)
        remaining_energy[i_start, freq_idx] = 0

        # forward pass
        i = i_start + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx]:
                k = 0
            else:
                k += 1

            remaining_energy[i, freq_idx] = 0
            if freq_idx < MAX_FREQ_IDX:
                remaining_energy[i, freq_idx + 1] = 0
            if freq_idx > 0:
                remaining_energy[i, freq_idx - 1] = 0

            i += 1

        i_end = i - 1 - k  # go back to frame above threshold

        assert i_start >= 0, "{}".format(i_start)
        assert i_end < n_frames

        if i_end - i_start <= min_note_len:
            # note is too short, skip it
            continue

        # add the note
        amplitude = np.mean(frames[i_start:i_end, freq_idx])
        note_events.append(
            (
                i_start,
                i_end,
                freq_idx + MIDI_OFFSET,
                amplitude,
            )
        )

    return note_events

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    predict("input.wav")[1].write("output.mid")
