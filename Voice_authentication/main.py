# @title Default title text

import multiprocessing
import os
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import myconfig
import dataset
import model

def get_triplet_loss(anchor, pos, neg,):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0.0))


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = torch.reshape(batch_output, (batch_size, 3, batch_output.shape[1]))     #batch_output_reshaped.shape=[8,3,128]
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],     #all the 1st row will be anchor
        batch_output_reshaped[:, 1, :],     #all the 2nd row will be positive
        batch_output_reshaped[:, 2, :])     #all the 3rd row will be negative
    loss = torch.mean(batch_loss)
    return loss


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


def train_network(speaker_to_utterance, num_steps, saved_model="", pool=None):
    losses = []
    start_time = time.time()
    encoder = model.get_speaker_encoder()

    #Train
    optimizer = torch.optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        #build batch input
        batch_input = dataset.get_batched_triplet_input(speaker_to_utterance, myconfig.BATCH_SIZE, pool)
        batch_output = encoder(batch_input)     #batch_output.shape=[24,64*2]
        loss = get_triplet_loss_from_batch_output(batch_output, myconfig.BATCH_SIZE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"step: {step}/{num_steps} loss: {loss.item()}")

    #     saving model
        if saved_model is not None and (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0:
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint,encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def run_training():
    print("Training data:", myconfig.TRAIN_DATA_DIR)
    speaker_to_utterance = dataset.get_librispeech_speaker_to_utterance(myconfig.TRAIN_DATA_DIR)
    print("Hi",speaker_to_utterance)
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(speaker_to_utterance,
                                myconfig.TRAINING_STEPS,
                                myconfig.SAVED_MODEL_PATH + "TFnew",
                                pool)
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.show()



if __name__ == '__main__':
    run_training()