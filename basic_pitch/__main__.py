import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from basic_pitch.inference import predict

predict("input.wav")[1].write("output.mid")