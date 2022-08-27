#!/usr/bin/env python
# encoding: utf-8

# This module is comprised of PyTorch layers from NNAudio and ported to TensorFlow:
# https://github.com/KinWaiCheuk/nnAudio
# The above code is released under an MIT license.

import os
import math
import warnings
from typing import Any, List, Dict, Optional, Tuple, Union

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


def midi_to_hz(notes: ArrayLike):
    return 440 * (2 ** ((np.asanyarray(notes) - 69) / 12))


def hz_to_midi(frequencies: ArrayLike):
    return 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(440)) + 69


def create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = 256,
    transition_bandwidth: float = 0.03,
    dtype: tf.dtypes.DType = tf.float32,
) -> np.ndarray:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through. Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is
    the Nyquist frequency of the signal BEFORE downsampling.
    """

    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passband_max] is the frequency range we want to keep
    # untouched and [stopband_min, 1.0] is the range we want to remove
    key_frequencies = [0.0, passband_max, stopband_min, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gain_at_key_frequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filter_kernel = scipy.signal.firwin2(kernel_length, key_frequencies, gain_at_key_frequencies)

    return tf.constant(filter_kernel, dtype=dtype)


def next_power_of_2(A: int) -> int:
    """A helper function to calculate the next nearest number to the power of 2."""
    return math.ceil(np.log2(A))


def get_window_dispatch(window: Union[str, Tuple[str, float]], N: int, fftbins: bool = True) -> NDArray:
    if isinstance(window, str):
        return scipy.signal.get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return scipy.signal.get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
    window: str = "hann",
) -> Tuple[NDArray, int, NDArray, NDArray]:
    """Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** next_power_of_2(np.ceil(Q * fs / fmin))
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    tempKernel = np.zeros((n_bins, fftLen), dtype=np.complex64)
    lengths = np.ceil(Q * fs / freqs)
    for k in range(n_bins):
        freq = freqs[k]
        _l = math.ceil(Q * fs / freq)

        # Centering the kernels, pad more zeros on RHS
        start = math.ceil(fftLen / 2 - _l / 2) - _l % 2

        sig = (
            get_window_dispatch(window, _l, fftbins=True)
            * np.exp(2j * np.pi * np.r_[-_l // 2 : _l // 2] * freq / fs)
            / _l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + _l] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + _l] = sig

    return tempKernel, fftLen, lengths, freqs


def get_cqt_complex(
    x: tf.Tensor,
    cqt_kernels_real: tf.Tensor,
    cqt_kernels_imag: tf.Tensor,
    hop_length: int,
    padding: tf.keras.layers.Layer,
) -> tf.Tensor:
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    try:
        x = padding(x)  # When center is True, we need padding at the beginning and ending
    except Exception:
        warnings.warn(
            f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
            "padding with reflection mode might not be the best choice, try using constant padding",
            UserWarning,
        )
        x = tf.pad(x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2))
    CQT_real = tf.transpose(
        tf.nn.conv1d(
            tf.transpose(x, [0, 2, 1]),
            tf.transpose(cqt_kernels_real, [2, 1, 0]),
            padding="VALID",
            stride=hop_length,
        ),
        [0, 2, 1],
    )
    CQT_imag = -tf.transpose(
        tf.nn.conv1d(
            tf.transpose(x, [0, 2, 1]),
            tf.transpose(cqt_kernels_imag, [2, 1, 0]),
            padding="VALID",
            stride=hop_length,
        ),
        [0, 2, 1],
    )

    return tf.stack((CQT_real, CQT_imag), axis=-1)


def downsampling_by_n(x: tf.Tensor, filter_kernel: tf.Tensor, n: float) -> tf.Tensor:
    """Downsample the given tensor using the given filter kernel.
    The input tensor is expected to have shape `(n_batches, channels, width)`,
    and the filter kernel is expected to have shape `(num_output_channels,)` (i.e.: 1D)

    If match_torch_exactly is passed, we manually pad the input rather than having TensorFlow do so with "SAME".
    The result is subtly different than Torch's output, but it is compatible with TensorFlow Lite (as of v2.4.1).
    """

    paddings = [
        [0, 0],
        [0, 0],
        [(filter_kernel.shape[-1] - 1) // 2, (filter_kernel.shape[-1] - 1) // 2],
    ]
    padded = tf.pad(x, paddings)

    # Store this tensor in the shape `(n_batches, width, channels)`
    padded_nwc = tf.transpose(padded, [0, 2, 1])
    result_nwc = tf.nn.conv1d(padded_nwc, filter_kernel[:, None, None], padding="VALID", stride=n)
    result_ncw = tf.transpose(result_nwc, [0, 2, 1])
    return result_ncw


class ReflectionPad1D(tf.keras.layers.Layer):
    """Replica of Torch's nn.ReflectionPad1D in TF.
    """

    def __init__(self, padding: Union[int, Tuple[int]] = 1, **kwargs: Any):
        self.padding = padding
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]
        super(ReflectionPad1D, self).__init__(**kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.pad(x, [[0, 0], [0, 0], [self.padding, self.padding]], "REFLECT")


def pad_center(data: np.ndarray, size: int, axis: int = -1, **kwargs: Any) -> np.ndarray:
    """Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad `data`
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ValueError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(("Target size ({:d}) must be at least input size ({:d})").format(size, n))

    return np.pad(data, lengths, **kwargs)


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
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.70,
        fmax: Optional[float] = None,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        basis_norm: int = 1,
        trainable: bool = False,
    ):
        super().__init__()

        self.sample_rate: Union[float, int] = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.basis_norm = basis_norm
        self.trainable = trainable

    def build(self, input_shape: tf.TensorShape) -> None:
        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = 1 / (2 ** (1 / self.bins_per_octave) - 1)

        self.lowpass_filter = create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.001)

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(self.bins_per_octave, self.n_bins)
        self.n_octaves = int(np.ceil(float(self.n_bins) / self.bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = self.fmin * 2 ** (self.n_octaves - 1)
        remainder = self.n_bins % self.bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((self.bins_per_octave - 1) / self.bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / self.bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (1 - 1 / self.bins_per_octave)  # Adjusting the top minium bins
        if fmax_t > self.sample_rate / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(fmax_t)
            )

        # Preparing CQT kernels
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            self.sample_rate,
            self.fmin_t,
            n_filters,
            self.bins_per_octave,
            norm=self.basis_norm,
        )

        # For the normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = self.fmin * 2.0 ** (np.r_[0 : self.n_bins] / np.float(self.bins_per_octave))
        self.frequencies = freqs

        self.lengths = np.ceil(Q * self.sample_rate / freqs)

        self.basis = basis
        # NOTE(psobot): this is where the implementation here starts to differ from CQT2010.

        # These cqt_kernel is already in the frequency domain
        self.cqt_kernels_real = tf.expand_dims(basis.real.astype(self.dtype), 1)
        self.cqt_kernels_imag = tf.expand_dims(basis.imag.astype(self.dtype), 1)

        if self.trainable:
            self.cqt_kernels_real = tf.Variable(initial_value=self.cqt_kernels_real, trainable=True)
            self.cqt_kernels_imag = tf.Variable(initial_value=self.cqt_kernels_imag, trainable=True)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        self.padding = ReflectionPad1D(self.n_fft // 2)

        rank = len(input_shape)
        if rank == 2:
            self.reshape_input = lambda x: x[:, None, :]
        elif rank == 1:
            self.reshape_input = lambda x: x[None, None, :]
        elif rank == 3:
            self.reshape_input = lambda x: x
        else:
            raise ValueError(f"Input shape must be rank <= 3, found shape {input_shape}")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.reshape_input(x)  # type: ignore

        hop = self.hop_length

        # Getting the top octave CQT
        CQT = get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)

        x_down = x  # Preparing a new variable for downsampling

        for _ in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_n(x_down, self.lowpass_filter, 2)
            CQT1 = get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
            CQT = tf.concat((CQT1, CQT), axis=1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins

        # Normalize again to get same result as librosa
        CQT *= tf.math.sqrt(tf.cast(self.lengths.reshape((-1, 1, 1)), self.dtype))

        # Transpose the output to match the output of the other spectrogram layers.
        # Getting CQT Amplitude
        return tf.transpose(tf.norm(CQT, axis=-1), [0, 2, 1])


class HarmonicStacking(tf.keras.layers.Layer):
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

    def __init__(
        self, bins_per_semitone: int, harmonics: List[float], n_output_freqs: int
    ):
        """Downsample frequency by stride, upsample channels by 4."""
        super().__init__(trainable=False)
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [
            round(12 * self.bins_per_semitone * np.log2(h)) for h in self.harmonics
        ]
        self.n_output_freqs = n_output_freqs

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # (n_batch, n_times, n_freqs, 1)
        tf.debugging.assert_equal(tf.shape(x).shape, 4)
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
                padded = tf.pad(x[:, :, shift:, :], paddings)
            elif shift < 0:
                paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
                padded = tf.pad(x[:, :, :shift, :], paddings)
            else:
                raise ValueError

            channels.append(padded)
        x = tf.concat(channels, axis=-1)
        x = x[:, :, : self.n_output_freqs, :]  # return only the first n_output_freqs frequency channels
        return x


class FlattenAudioCh(tf.keras.layers.Layer):
    """Layer which removes a "channels" dimension of size 1.

    Input shape: (batch, time, 1)
    Output shape: (batch, time)
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """x: (batch, time, ch)"""
        shapes = tf.keras.backend.int_shape(x)
        tf.assert_equal(shapes[2], 1)
        return tf.keras.layers.Reshape([shapes[1]])(x)  # ignore batch size


class FlattenFreqCh(tf.keras.layers.Layer):
    """Layer to flatten the frequency channel and make each channel
    part of the frequency dimension.

    Input shape: (batch, time, freq, ch)
    Output shape: (batch, time, freq*ch)
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        shapes = tf.keras.backend.int_shape(x)
        return tf.keras.layers.Reshape([shapes[1], shapes[2] * shapes[3]])(x)  # ignore batch size


class NormalizedLog(tf.keras.layers.Layer):
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """

    def build(self, input_shape: tf.Tensor) -> None:
        self.squeeze_batch = lambda batch: batch
        rank = input_shape.rank
        if rank == 4:
            assert input_shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
            self.squeeze_batch = lambda batch: tf.squeeze(batch, axis=1)
        else:
            assert rank == 3, f"Only ranks 3 and 4 are supported!. Received rank {rank} for {input_shape}."

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.squeeze_batch(inputs)  # type: ignore
        # convert magnitude to power
        power = tf.math.square(inputs)
        log_power = 10 * tf.math.log(power + 1e-10)

        log_power_min = tf.reshape(tf.math.reduce_min(log_power, axis=[1, 2]), [tf.shape(inputs)[0], 1, 1])
        log_power_offset = log_power - log_power_min
        log_power_offset_max = tf.reshape(
            tf.math.reduce_max(log_power_offset, axis=[1, 2]),
            [tf.shape(inputs)[0], 1, 1],
        )
        log_power_normalized = tf.math.divide_no_nan(log_power_offset, log_power_offset_max)

        return tf.reshape(log_power_normalized, tf.shape(inputs))

tfkl = tf.keras.layers

MAX_N_SEMITONES = int(
    np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY))
)


def get_cqt(inputs: tf.Tensor, n_harmonics: int, use_batchnorm: bool) -> tf.Tensor:
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = np.min(
        [
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ]
    )
    x = FlattenAudioCh()(inputs)
    x = CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = NormalizedLog()(x)
    x = tf.expand_dims(x, -1)
    if use_batchnorm:
        x = tfkl.BatchNormalization()(x)
    return x


def get_model() -> tf.keras.Model:
    """Basic Pitch's model implementation.

    Args:
        n_harmonics: The number of harmonics to use in the harmonic stacking layer.
        n_filters_contour: Number of filters for the contour convolutional layer.
        n_filters_onsets: Number of filters for the onsets convolutional layer.
        n_filters_notes: Number of filters for the notes convolutional layer.
        no_contours: Whether or not to include contours in the output.
    """
    n_harmonics = 8
    # input representation
    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))  # (batch, time, ch)
    x = get_cqt(inputs, n_harmonics, True)

    x = HarmonicStacking(
        CONTOURS_BINS_PER_SEMITONE,
        [0.5] + list(range(1, n_harmonics)),
        N_FREQ_BINS_CONTOURS,
    )(x)

    # contour layers - fully convolutional
    x_contours = tfkl.Conv2D(32, (5, 5), padding="same")(x)

    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    x_contours = tfkl.Conv2D(8, (3, 3 * 13), padding="same")(x)

    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    x_contours = tfkl.Conv2D(1, (5, 5), padding="same", activation="sigmoid")(
        x_contours
    )
    x_contours = FlattenFreqCh()(x_contours)  # contour output

    # reduced contour output as input to notes
    x_contours_reduced = tf.expand_dims(x_contours, -1)

    x_contours_reduced = tfkl.Conv2D(
        32, (7, 7), padding="same", strides=(1, 3)
    )(x_contours_reduced)
    x_contours_reduced = tfkl.ReLU()(x_contours_reduced)

    # note output layer
    x_notes_pre = tfkl.Conv2D(1, (7, 3), padding="same", activation="sigmoid")(
        x_contours_reduced
    )
    x_notes = FlattenFreqCh()(x_notes_pre)

    # onset output layer

    # onsets - fully convolutional
    x_onset = tfkl.Conv2D(32, (5, 5), padding="same", strides=(1, 3))(x)
    x_onset = tfkl.BatchNormalization()(x_onset)
    x_onset = tfkl.ReLU()(x_onset)
    x_onset = tfkl.Concatenate(axis=3)([x_notes_pre, x_onset])
    x_onset = tfkl.Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x_onset)

    x_onset = FlattenFreqCh()(x_onset)

    outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def window_audio_file(audio_original: tf.Tensor, hop_size: int) -> Tuple[tf.Tensor, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    audio_windowed = tf.expand_dims(
        tf.signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times


def get_audio_input(
    audio_path: str, overlap_len: int, hop_size: int
) -> Tuple[tf.Tensor, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
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
    audio_original = np.pad(audio_original, (overlap_len // 2, 0))
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, audio_original_length


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


def run_inference(
    audio_path: str, model: tf.keras.Model
) -> Dict[str, NDArray]:
    """Run the model on the input audio path.

    Args:
        audio_path: The audio to run inference on.
        model: A loaded keras model to run inference with.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)

    output = model(audio_windowed)
    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}

    return unwrapped_output


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
    model_output = run_inference(audio_path, model)

    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
    )

    return model_output, midi_data, note_events


MIDI_OFFSET = 21
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87


def model_output_to_notes(
    output: Dict[str, NDArray],
    onset_thresh: float,
    frame_thresh: float,
    infer_onsets: bool = True,
    min_note_len: int = 5,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    include_pitch_bends: bool = True,
) -> Tuple[pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
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
    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    estimated_notes = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=infer_onsets,
        min_note_len=min_note_len,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    if include_pitch_bends:
        estimated_notes_with_pitch_bend = get_pitch_bends(contours, estimated_notes)
    else:
        estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

    times_s = model_frames_to_time(contours.shape[0])
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    return note_events_to_midi(estimated_notes_time_seconds), estimated_notes_time_seconds


def midi_pitch_to_contour_bin(pitch_midi: int) -> NDArray:
    """Convert midi pitch to conrresponding index in contour matrix

    Args:
        pitch_midi: pitch in midi

    Returns:
        index in contour matrix
    """
    return 12 * CONTOURS_BINS_PER_SEMITONE * np.log2(midi_to_hz(pitch_midi) / ANNOTATIONS_BASE_FREQUENCY)


def get_pitch_bends(
    contours: np.ndarray, note_events: List[Tuple[int, int, int, float]], n_bins_tolerance: int = 25
) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
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
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = scipy.signal.gaussian(window_length, std=5)
    note_events_with_pitch_bends = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end_idx = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                np.max([0, n_bins_tolerance - freq_idx]) : window_length
                - np.max([0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)])
            ]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])

        bends: Optional[List[int]] = list(
            np.argmax(pitch_bend_submatrix, axis=1) - pb_shift
        )  # this is in units of 1/3 semitones
        note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return note_events_with_pitch_bends


def note_events_to_midi(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
) -> pretty_midi.PrettyMIDI:
    """Create a pretty_midi object from note events

    Args:
        note_events : list of tuples [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            where amplitude is a number between 0 and 1

    Returns:
        pretty_midi.PrettyMIDI() object

    """
    mid = pretty_midi.PrettyMIDI()
    note_events_with_pitch_bends = drop_overlapping_pitch_bends(note_events_with_pitch_bends)

    instrument = pretty_midi.Instrument(
        pretty_midi.instrument_name_to_program("Electric Piano 1")
    )
    for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
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
    mid.instruments.append(instrument)

    return mid


def drop_overlapping_pitch_bends(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]]
) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
    """Drop pitch bends from any notes that overlap in time with another note"""
    note_events = sorted(note_events_with_pitch_bends)
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:  # start j > end i
                break
            note_events[i] = note_events[i][:-1] + (None,)  # last field is pitch bend
            note_events[j] = note_events[j][:-1] + (None,)

    return note_events


def get_infered_onsets(onsets: NDArray, frames: NDArray, n_diff: int = 2) -> NDArray:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences

    return max_onsets_diff


def constrain_frequency(
    onsets: NDArray, frames: NDArray, max_freq: Optional[float], min_freq: Optional[float]
) -> Tuple[NDArray, NDArray]:
    """Zero out activations above or below the max/min frequencies

    Args:
        onsets: Onset activation matrix (n_times, n_freqs)
        frames: Frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep.
        min_freq: the minimum frequency to keep.

    Returns:
       The onset and frame activation matrices, with frequencies outside the min and max
       frequency set to 0.
    """
    if max_freq is not None:
        max_freq_idx = int(np.round(hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0

    return onsets, frames


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = np.arange(n_frames) * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.arange(n_frames) // ANNOT_N_FRAMES
    window_offset = FFT_HOP / AUDIO_SAMPLE_RATE * (
        ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP
    ) + 0.0018  # this is a magic number, but it's needed for this to align properly
    times = original_times - window_offset * window_numbers
    return times


def output_to_notes_polyphonic(
    frames: NDArray,
    onsets: NDArray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
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

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

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
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
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

    energy_shape = remaining_energy.shape

    while np.max(remaining_energy) > frame_thresh:
        i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
        remaining_energy[i_mid, freq_idx] = 0

        # forward pass
        i = i_mid + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:

            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0

            remaining_energy[i, freq_idx] = 0
            if freq_idx < MAX_FREQ_IDX:
                remaining_energy[i, freq_idx + 1] = 0
            if freq_idx > 0:
                remaining_energy[i, freq_idx - 1] = 0

            i += 1

        i_end = i - 1 - k  # go back to frame above threshold

        # backward pass
        i = i_mid - 1
        k = 0
        while i > 0 and k < energy_tol:

            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0

            remaining_energy[i, freq_idx] = 0
            if freq_idx < MAX_FREQ_IDX:
                remaining_energy[i, freq_idx + 1] = 0
            if freq_idx > 0:
                remaining_energy[i, freq_idx - 1] = 0

            i -= 1

        i_start = i + 1 + k  # go back to frame above threshold
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