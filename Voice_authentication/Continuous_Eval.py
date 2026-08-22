import torch
import numpy as np
from scipy.spatial.distance import cosine
import webrtcvad
import wave
import myconfig

# Load Pre-trained Model
def load_model(model_path):
    """Load the pre-trained authentication model."""
    model = torch.load(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# Voice Activity Detection (VAD)
def is_speaking(audio_path, vad_mode=3):
    """Detect if the user is speaking in an audio file."""
    vad = webrtcvad.Vad(vad_mode)
    with wave.open(audio_path, "rb") as wf:
        assert wf.getframerate() == 16000, "Audio must be 16 kHz."
        assert wf.getnchannels() == 1, "Audio must be mono."
        frame_duration = 30  # 30 ms
        frame_size = int(wf.getframerate() * frame_duration / 1000) * 2  # 2 bytes per sample
        frames = wf.readframes(frame_size // 2)

        while len(frames) == frame_size:
            if vad.is_speech(frames, wf.getframerate()):
                return True
            frames = wf.readframes(frame_size // 2)
    return False

# Generate Embedding
def get_embedding(model, audio_path, device):
    """Extract voice embedding from the audio using the model."""
    # Mock embedding function (replace with actual audio preprocessing and model inference)
    audio_tensor = torch.rand(1, 80, 200).to(device)  # Example input shape (Batch, Features, Time)
    with torch.no_grad():
        embedding = model(audio_tensor)
    return embedding.cpu().numpy().flatten()

# Authenticate User
def authenticate_user(model, device, sample_audio_path, auth_audio_path, threshold=0.7):
    """
    Authenticate a user by comparing embeddings of two audio files.

    :param model: Pre-trained authentication model.
    :param device: Torch device (CPU/GPU).
    :param sample_audio_path: Path to the sample audio (reference embedding).
    :param auth_audio_path: Path to the authentication audio.
    :param threshold: Cosine similarity threshold for authentication.
    :return: Boolean indicating authentication success.
    """
    # Check if the authentication audio has speech
    if not is_speaking(auth_audio_path):
        print("No speech detected in authentication audio.")
        return False

    # Generate embeddings
    sample_embedding = get_embedding(model, sample_audio_path, device)
    auth_embedding = get_embedding(model, auth_audio_path, device)

    # Compute similarity
    similarity = 1 - cosine(sample_embedding, auth_embedding)
    print(f"Cosine Similarity: {similarity:.2f}")

    if similarity > threshold:
        print("Authentication successful.")
        return True
    else:
        print("Authentication failed.")
        return False

# Continuous Authentication
def continuous_authentication(model_path, sample_audio_path, live_audio_segments, threshold=0.7):
    """
    Continuously authenticate a user using live audio segments.

    :param model_path: Path to the pre-trained model.
    :param sample_audio_path: Path to the sample audio (reference embedding).
    :param live_audio_segments: List of paths to live audio segments.
    :param threshold: Cosine similarity threshold for authentication.
    """
    model, device = load_model(model_path)

    for segment_path in live_audio_segments:
        print(f"Processing segment: {segment_path}")
        authenticated = authenticate_user(model, device, sample_audio_path, segment_path, threshold)

        if authenticated:
            print("User authenticated for this segment.")
        else:
            print("Authentication failed for this segment.")

# Example Usage
if __name__ == "__main__":
    model_path = myconfig.SAVED_MODEL_PATH
    sample_audio = "path/to/sample_audio.wav"
    live_audio_segments = ["path/to/live_segment1.wav", "path/to/live_segment2.wav"]

    continuous_authentication(model_path, sample_audio, live_audio_segments)