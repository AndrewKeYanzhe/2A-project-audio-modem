import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import chirp
import logging

class ChirpSignalGenerator:
    """
    The ChirpSignalGenerator class generates a linear chirp signal with a circular prefix and silence,
    normalizes the signal, saves it as a WAV file, and plots it in the time domain.

    Attributes:
        fs (int): Sampling rate in Hz.
        t_chirp (float): Duration of the chirp in seconds.
        t_silence (float): Duration of silence in seconds.
        f_low (int): Start frequency of the chirp in Hz.
        f_high (int): End frequency of the chirp in Hz.
        prefix_duration (float): Duration of the circular prefix in seconds.
        full_signal (np.ndarray): The generated chirp signal with prefix and silence.
        t (np.ndarray): The time array matching the signal length.
    
    Methods:
        generate_chirp_signal(): Generates the chirp signal with prefix and silence.
        normalize_signal(): Normalizes the signal to 16-bit integer values and floating-point values.
        save_as_wav(file_path): Saves the signal as a WAV file.
        plot_signal(): Plots the signal in the time domain.
        play_signal(): Plays the signal using the sounddevice library.
    """

    def __init__(self, fs=48000, t_chirp=1.365, t_silence=0.0, f_low=20, f_high=8000, prefix_duration=0.02):
        self.fs = fs
        self.t_chirp = t_chirp
        self.t_silence = t_silence
        self.f_low = f_low
        self.f_high = f_high
        self.prefix_duration = prefix_duration
        self.full_signal = None
        self.t = None

    def generate_chirp_signal(self):
        """Generates the chirp signal with prefix and silence."""
        t_chirp_only = np.linspace(0, self.t_chirp, int(self.fs * self.t_chirp), endpoint=False)
        chirp_signal = chirp(t_chirp_only, f0=self.f_low, f1=self.f_high, t1=self.t_chirp, method='linear')

        # Create circular prefix (last prefix_duration seconds of chirp)
        prefix = chirp_signal[-int(self.fs * self.prefix_duration):]

        # Generate silence
        silence = np.zeros(int(self.fs * self.t_silence))

        # Combine all parts
        self.full_signal = np.concatenate([silence, prefix, chirp_signal, silence])

        # Calculate total duration
        t_total = 2 * self.t_silence + self.prefix_duration + self.t_chirp

        # Ensure the time array matches the signal length
        self.t = np.linspace(0, t_total, len(self.full_signal), endpoint=False)

    def normalize_signal(self):
        """Normalizes the signal to 16-bit integer values and floating-point values."""
        signal_int = np.int16(self.full_signal / np.max(np.abs(self.full_signal)) * 32767)
        signal_normalized = self.full_signal / np.max(np.abs(self.full_signal))
        return signal_int, signal_normalized

    def save_as_wav(self, file_path):
        """Saves the signal as a WAV file."""
        signal_int, _ = self.normalize_signal()
        write(file_path, self.fs, signal_int)
        logging.debug(f"Signal has been saved to {file_path}.")

    def plot_signal(self):
        """Plots the signal in the time domain."""
        _, signal_normalized = self.normalize_signal()
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, signal_normalized)
        plt.title('Linear Chirp with Circular Prefix and Silence: Time Domain')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()

    def play_signal(self):
        """Plays the signal using the sounddevice library."""
        _, signal_normalized = self.normalize_signal()
        sd.play(signal_normalized, self.fs)
        sd.wait()  # Wait until the sound has finished playing



if __name__ == "__main__":
    # Example usage
    generator = ChirpSignalGenerator()
    generator.generate_chirp_signal()
    generator.save_as_wav('recordings/transmitted_linear_chirp_with_prefix_and_silence.wav')

    # If you want to play the signal, call the play_signal method
    # generator.play_signal()