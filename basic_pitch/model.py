import numpy as np
import tensorflow as tf

from basic_pitch import nn
from basic_pitch import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch import nnaudio, signal

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
    x = nn.FlattenAudioCh()(inputs)
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = signal.NormalizedLog()(x)
    x = tf.expand_dims(x, -1)
    if use_batchnorm:
        x = tfkl.BatchNormalization()(x)
    return x


def model() -> tf.keras.Model:
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

    x = nn.HarmonicStacking(
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
    x_contours = nn.FlattenFreqCh()(x_contours)  # contour output

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
    x_notes = nn.FlattenFreqCh()(x_notes_pre)

    # onset output layer

    # onsets - fully convolutional
    x_onset = tfkl.Conv2D(32, (5, 5), padding="same", strides=(1, 3))(x)
    x_onset = tfkl.BatchNormalization()(x_onset)
    x_onset = tfkl.ReLU()(x_onset)
    x_onset = tfkl.Concatenate(axis=3)([x_notes_pre, x_onset])
    x_onset = tfkl.Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x_onset)

    x_onset = nn.FlattenFreqCh()(x_onset)

    outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}

    return tf.keras.Model(inputs=inputs, outputs=outputs)
