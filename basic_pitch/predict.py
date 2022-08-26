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

import argparse
import os

from basic_pitch import ICASSP_2022_MODEL_PATH


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main() -> None:
    """Handle command line arguments. Entrypoint for this script."""
    parser = argparse.ArgumentParser(description="Predict midi from audio.")
    parser.add_argument("audio_paths", type=str, nargs="+", help="Space separated paths to the input audio files.")
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="The minimum likelihood for an onset to occur, between 0 and 1.",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="The minimum likelihood for a frame to sustain, between 0 and 1.",
    )
    parser.add_argument(
        "--minimum-note-length",
        type=float,
        default=58,
        help="The minimum allowed note length, in miliseconds.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=None,
        help="The minimum allowed note frequency, in Hz.",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=None,
        help="The maximum allowed note frequency, in Hz.",
    )
    parser.add_argument(
        "--multiple-pitch-bends",
        action="store_true",
        help="Allow overlapping notes in midi file to have pitch bends. Note: this will map each "
        "pitch to its own instrument",
    )
    parser.add_argument("--no-melodia", default=False, action="store_true", help="Skip the melodia trick.")
    args = parser.parse_args()

    # tensorflow is very slow to import
    # this import is here so that the help messages print faster
    from basic_pitch.inference import predict_and_save

    audio_path_list = args.audio_paths

    predict_and_save(
        audio_path_list,
        args.onset_threshold,
        args.frame_threshold,
        args.minimum_note_length,
        args.minimum_frequency,
        args.maximum_frequency,
        args.multiple_pitch_bends,
        not args.no_melodia,
    )

    print("\n✨ Done ✨\n")


if __name__ == "__main__":
    main()
