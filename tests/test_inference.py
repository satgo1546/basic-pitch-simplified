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
import unittest

import librosa
import pretty_midi
import numpy as np

import nmp
from nmp import ANNOTATIONS_N_SEMITONES

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources")


class TestPredict(unittest.TestCase):
    def test_predict(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        model_output, midi_data, note_events = nmp.predict(
            test_audio_path,
        )
        assert set(model_output.keys()) == set(["note", "onset", "contour"])
        assert model_output["note"].shape == model_output["onset"].shape
        assert isinstance(midi_data, pretty_midi.PrettyMIDI)
        lowest_supported_midi = 21
        note_pitch_min = [n[2] >= lowest_supported_midi for n in note_events]
        note_pitch_max = [n[2] <= lowest_supported_midi + ANNOTATIONS_N_SEMITONES for n in note_events]
        assert all(note_pitch_min)
        assert all(note_pitch_max)
        assert isinstance(note_events, list)

    def test_predict_onset_threshold(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        for onset_threshold in [0, 0.3, 0.8, 1]:
            nmp.predict(
                test_audio_path,
                onset_threshold=onset_threshold,
            )

    def test_predict_frame_threshold(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        for frame_threshold in [0, 0.3, 0.8, 1]:
            nmp.predict(
                test_audio_path,
                frame_threshold=frame_threshold,
            )

    def test_predict_min_note_length(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        for minimum_note_length in [10, 100, 1000]:
            _, _, note_events = nmp.predict(
                test_audio_path,
                minimum_note_length=minimum_note_length,
            )
            min_len_s = minimum_note_length / 1000.0
            note_lengths = [n[1] - n[0] >= min_len_s for n in note_events]
            assert all(note_lengths)

    def test_predict_min_freq(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        for minimum_frequency in [40, 80, 200, 2000]:
            _, _, note_events = nmp.predict(
                test_audio_path,
                minimum_frequency=minimum_frequency,
            )
            min_freq_midi = np.round(librosa.hz_to_midi(minimum_frequency))
            note_pitch = [n[2] >= min_freq_midi for n in note_events]
            assert all(note_pitch)

    def test_predict_max_freq(self) -> None:
        test_audio_path = os.path.join(RESOURCES_PATH, "vocadito_10.wav")
        for maximum_frequency in [40, 80, 200, 2000]:
            _, _, note_events = nmp.predict(
                test_audio_path,
                maximum_frequency=maximum_frequency,
            )
            max_freq_midi = np.round(librosa.hz_to_midi(maximum_frequency))
            note_pitch = [n[2] <= max_freq_midi for n in note_events]
            assert all(note_pitch)
