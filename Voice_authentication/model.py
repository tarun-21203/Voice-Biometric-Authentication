import torch.nn as nn
from . import myconfig
import os
import time
# def create_model():
#     if myconfig.USE_TRANSFORMER:
#         return TransformerModel()
#     else:
#         return LSTMModel()

# class LSTMModel(nn.Module):
#     def __init__(self):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=myconfig.N_MFCC,
#             hidden_size=myconfig.LSTM_HIDDEN_SIZE,
#             num_layers=myconfig.LSTM_NUM_LAYERS,
#             bidirectional=myconfig.BI_LSTM,
#             batch_first=True
#         )
#         self.fc = nn.Linear(myconfig.LSTM_HIDDEN_SIZE * 2 if myconfig.BI_LSTM else myconfig.LSTM_HIDDEN_SIZE, 10)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         return self.fc(lstm_out[:, -1, :])

# class TransformerModel(nn.Module):
#     def __init__(self):
#         super(TransformerModel, self).__init__()
#         self.transformer = nn.Transformer(
#             d_model=myconfig.TRANSFORMER_DIM,
#             nhead=myconfig.TRANSFORMER_HEADS,
#             num_encoder_layers=myconfig.TRANSFORMER_ENCODER_LAYERS,
#             batch_first=True
#         )
#         self.fc = nn.Linear(myconfig.TRANSFORMER_DIM, 10)

#     def forward(self, x):
#         transformer_out = self.transformer(x)
#         return self.fc(transformer_out[:, -1, :])



import torch
from torch import nn

class BaseSpeakerEncoder(nn.Module):
    def _load_from(self, saved_model):
        var_dict=torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(myconfig.N_MFCC, myconfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=myconfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=1)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, myconfig.TRANSFORMER_DIM).to(
            myconfig.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]

class LstmSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)
        if saved_model:
            self._load_from(saved_model)

    def _aggregate_frames(self, batch_output):
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]

    def forward(self, x):
        D = 2 if myconfig.BI_LSTM else 1
        h0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        c0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)

class CnnSpeakerEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(CnnSpeakerEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=myconfig.N_MFCC, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, 256)
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.mean(x, dim=2)
        output = self.fc(x)
        return output

# class CnnSpeakerEncoder(BaseSpeakerEncoder):
#     def __init__(self, saved_model=""):
#         super(CnnSpeakerEncoder, self).__init__()
#         # Added more convolutional layers
#         self.conv1 = nn.Conv1d(in_channels=myconfig.N_MFCC, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

#         # Batch normalization for each conv layer
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(256)

#         # Fully connected layer and dropout
#         self.fc = nn.Linear(256, 256)
#         self.dropout = nn.Dropout(0.3)  # Added dropout for regularization

#         if saved_model:
#             self._load_from(saved_model)

    # def forward(self, x):
    #     x = x.transpose(1, 2)
    #     x = torch.relu(self.bn1(self.conv1(x)))
    #     x = torch.relu(self.bn2(self.conv2(x)))
    #     x = torch.relu(self.bn3(self.conv3(x)))
    #     x = torch.relu(self.bn4(self.conv4(x)))
    #     x = torch.mean(x, dim=2)
    #     x = self.dropout(x)
    #     output = self.fc(x)
    #     return output

class HMMEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(HMMEncoder, self).__init__()
        # Hidden states and transition probabilities
        self.hidden_states = nn.Linear(myconfig.N_MFCC, 256)
        self.transition_probs = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 256)
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        # Initial hidden state
        h_t = torch.sigmoid(self.hidden_states(x))

        for _ in range(myconfig.TRAINING_STEPS):
            # Calculate next state
            h_t = torch.sigmoid(self.transition_probs(h_t))

        # Final output
        output = self.output_layer(h_t)
        return output



def save_model(saved_model_path, encoder, losses, start_time):
    """Save model to disk."""
    training_time = time.time() - start_time
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".pt"):
        saved_model_path += ".pt"
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "losses": losses,
                "training_time": training_time},
               saved_model_path)


def get_speaker_encoder(load_from=""):
    if myconfig.Use_Model=="CNN":
      return CnnSpeakerEncoder(load_from).to(myconfig.DEVICE)
    elif myconfig.Use_Model=="TF":
        return TransformerSpeakerEncoder(load_from).to(myconfig.DEVICE)
    elif myconfig.Use_Model=="LSTM":
        return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)
    elif myconfig.Use_Model=="HMM":
      return HMMEncoder(load_from).to(myconfig.DEVICE)
