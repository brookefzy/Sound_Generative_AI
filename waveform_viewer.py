# waveform_viewer.py
import argparse
import torchaudio
import matplotlib.pyplot as plt


def plot_waveform(path):
    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    plt.figure(figsize=(10, 4))
    plt.plot(wav.t().squeeze().numpy())
    plt.title(f"{path} (sr={sr})")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot waveform for WAV files")
    parser.add_argument("files", nargs="+", help="Paths of wav files to display")
    args = parser.parse_args()
    for f in args.files:
        plot_waveform(f)
    plt.show()


if __name__ == "__main__":
    main()
