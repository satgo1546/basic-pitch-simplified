Basic Pitch, simplified
=======================

This is a fork of
[Spotify's Basic Pitch for Automatic Music Transcription](https://basicpitch.io/),
[open-sourced](https://github.com/spotify/basic-pitch/blob/main/LICENSE "Apache License 2.0")
inference only in May 2022.
I try to reduce the code to ease understanding.
The core algorithm remains intact.

The upstream repository, as of August 2022, contains lots of unused code,
such as leftovers from PyTorch → TensorFlow conversion,
[bugs](https://github.com/spotify/basic-pitch/issues/21),
and loss functions and kernel initializers for training,
yet still no training code is given.

Most dependencies are removed to also ease installation.
While pip is a great tool,
it doesn't attempt to handle peculiar NumPy ABI version requests made by Numba,
which is required by resampy,
which is used by librosa,
which does the input preprocessing in Basic Pitch.
In my simplified version, it gets replaced by `scipy.signal.resample`,
and therefore, the output is slightly different from the original.

This repository records a learning experience.
Those interested are referred to
[spotify/basic-pitch](https://github.com/spotify/basic-pitch),
where links to the paper,
the conference video,
a convenient pip package with a command line interface and Python API,
and a TypeScript port
are provided.

Usage
-----

Install the following dependencies.
I test with the versions in the brackets.
Older versions should work, too.

- Python (3.10.6)
- NumPy (1.23.1)
- SciPy (1.9.0)
- TensorFlow (2.9.1)
- pretty_midi (0.2.9)

librosa, resampy and mir\_eval are no longer required.

To transcribe a sound file,
convert it to WAV,
name it `input.wav`,
and do `python -m basic_pitch`.
A MIDI sequence will be saved.
No additional information is saved.
