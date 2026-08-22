import os
import sys
import torch

# Ensure repository root is on sys.path so package imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Voice_authentication import model, myconfig, feature_extraction

MODEL_PATH = os.path.join('models', 'CNN.pt')
SAMPLE_PATH = os.path.join('LibriSpeech', 'test-clean', '6829', '68769', '6829-68769-0009.flac')

print('MODEL_PATH=', MODEL_PATH)
print('SAMPLE_PATH=', SAMPLE_PATH)

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")
if not os.path.exists(SAMPLE_PATH):
    raise SystemExit(f"Sample audio not found: {SAMPLE_PATH}")

print('Loading encoder...')
encoder = model.get_speaker_encoder(MODEL_PATH)
encoder.eval()
print('Encoder loaded. Device:', myconfig.DEVICE)

print('Extracting features...')
features = feature_extraction.extract_features(SAMPLE_PATH)
print('Features shape:', features.shape)

print('Preparing input tensor...')
input_tensor = torch.unsqueeze(torch.from_numpy(features).float(), dim=0).to(myconfig.DEVICE)
print('Input tensor shape:', input_tensor.shape)

with torch.no_grad():
    out = encoder(input_tensor)
    print('Encoder output shape:', out.shape)
    print('Encoder output (first 5 values):', out[0, :5].cpu().numpy())
