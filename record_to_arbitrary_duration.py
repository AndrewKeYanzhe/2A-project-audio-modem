#!/usr/bin/env python3
"""

This code records audio of an arbitrary duration, stopped by pressing "ctrl c"

The audio is recorded to a .wav file, and this .wav file is read into a list. 

Receiver code can be integrated by add into the process_audio() function. 

"""
import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import librosa
import matplotlib.pyplot as plt

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def process_audio():
    print("hi")

    file_path = args.filename

    audio_data, sr = librosa.load(file_path, sr=None)
    audio_data2, sr2 = librosa.load(file_path, sr=None)


    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(audio_data2, label=file_path.split('/')[-1])
    # plt.plot(audio_data2, label=file_path2.split('/')[-1])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Waveforms of {} and {}".format(file_path.split('/')[-1], file_path.split('/')[-1]))
    plt.legend()  # Show legend with file names
    plt.show()
    print(type)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        # args.samplerate = int(device_info['default_samplerate'])
        args.samplerate = 48000
    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                        suffix='.wav', dir='')
        # args.filename = "unlimited_duration.wav"

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                      channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, callback=callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())
except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    process_audio()
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))


