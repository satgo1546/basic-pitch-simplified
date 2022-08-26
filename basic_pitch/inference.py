#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray

from tensorflow import Tensor, signal, keras
import numpy as np
import librosa
import pretty_midi

from basic_pitch import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch import note_creation as infer
import basic_pitch.models


def window_audio_file(audio_original: Tensor, hop_size: int) -> Tuple[Tensor, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    from tensorflow import expand_dims  # imporing this here so the module loads faster

    audio_windowed = expand_dims(
        signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
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
) -> Tuple[Tensor, List[Dict[str, int]], int]:
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

    audio_original, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length


def unwrap_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int) -> NDArray:
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
    audio_path: str, model: keras.Model
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
    model = basic_pitch.models.model()
    model.load_weights(os.path.join(os.path.dirname(__file__), "nmp"))
    model_output = run_inference(audio_path, model)

    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
    )

    return model_output, midi_data, note_events
