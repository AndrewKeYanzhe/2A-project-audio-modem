"""
    The AudioProcessor class provides methods to load audio files, save audio data to a CSV file,
    plot the waveform of the audio, record audio using sounddevice, and truncate audio data based on a threshold.

    Attributes:
        file_path (str): The path to the audio file.
        sr (int, optional): The sample rate for loading the audio. If None, the default sample rate is used.
        audio_data (np.ndarray): The audio data loaded from the file or recorded.
    
    Methods:
        load_audio(): Loads audio data from the specified file.
        save_to_csv(csv_file_path): Saves the loaded audio data to a CSV file.
        plot_waveforms(): Plots the waveform of the loaded audio data.
        record_audio(duration, sr): Records audio for a specified duration and sample rate.
        truncate_list(threshold): Truncates the audio data at the first occurrence of a value exceeding the threshold.
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

class AudioProcessor:
    """
    The AudioProcessor class provides methods to load audio files, save audio data to a CSV file,
    plot the waveform of the audio, record audio using sounddevice, and truncate audio data based on a threshold.

    Attributes:
        file_path (str): The path to the audio file.
        sr (int, optional): The sample rate for loading the audio. If None, the default sample rate is used.
        audio_data (np.ndarray): The audio data loaded from the file or recorded.
    
    Methods:
        load_audio(): Loads audio data from the specified file.
        save_to_csv(csv_file_path): Saves the loaded audio data to a CSV file.
        plot_waveforms(): Plots the waveform of the loaded audio data.
        record_audio(duration, sr): Records audio for a specified duration and sample rate.
        truncate_list(threshold): Truncates the audio data at the first occurrence of a value exceeding the threshold.
    """

    def __init__(self, file_path=None, sr=None):
        self.file_path = file_path
        self.sr = sr
        self.audio_data = None

    def load_audio(self):
        self.audio_data, self.sr = librosa.load(self.file_path, sr=self.sr)
        return self.audio_data, self.sr

    def save_to_csv(self, csv_file_path):
        if self.audio_data is not None:
            np.savetxt(csv_file_path, self.audio_data, delimiter=',', fmt='%1.4f')
            print(f"Data written to {csv_file_path} successfully.")
        else:
            print("Audio data is not loaded. Please load the audio data first.")
    
    def plot_waveforms(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.audio_data, label=self.file_path.split('/')[-1] if self.file_path else 'Recorded Audio')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Waveform of Loaded Audio File")
        plt.legend()  # Show legend with file names
        plt.show()

    def record_audio(self, duration, sr=48000):
        self.sr = sr
        self.audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
        sd.wait()
        self.audio_data = self.audio_data.flatten()
        return self.audio_data

    def truncate_list(self, threshold):
        index = next((i for i, val in enumerate(self.audio_data) if abs(val) > threshold), None)
        if index is not None:
            self.audio_data = self.audio_data[:index]
        return self.audio_data


if __name__ == '__main__':
    # Example usage
    file_path1 = "recordings/chirp_1k_3k.m4a"
    csv_file_path = 'audio_data2.csv'

    audio_processor = AudioProcessor(file_path1)
    audio_processor.load_audio()
    audio_processor.save_to_csv(csv_file_path)
    audio_processor.plot_waveforms()

    # Record and truncate audio
    record_duration = 60  # seconds
    audio_processor.record_audio(record_duration)
    audio_processor.truncate_list(2000)
    audio_processor.save_to_csv('recorded_audio.csv')
    audio_processor.plot_waveforms()
