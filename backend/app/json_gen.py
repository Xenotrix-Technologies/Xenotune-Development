import json
import random
from threading import Lock

import torch
import torch.nn as nn
import torch.optim as optim

from app.config import CONFIG_PATH

_config_lock = Lock()


NOTE_VOCAB = sorted(set([
    "C2", "D2", "E2", "F2", "G2", "A2", "B2",
    "C3", "D3", "E3", "F#3", "G3", "A3", "B3",
    "C4", "D4", "E4", "F#4", "G4", "A4",
    "C5", "D5", "E5", "F#5", "G5", "A5", "B5"
]))
note_to_int = {n: i for i, n in enumerate(NOTE_VOCAB)}
int_to_note = {i: n for n, i in note_to_int.items()}


class NoteLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim=32, hidden_dim=64, output_dim=None):
        super(NoteLSTM, self).__init__()
        output_dim = output_dim or input_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def load_config():
  with _config_lock:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config):
 with _config_lock:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print("✅ Config saved.")


def generate_training_data(seq_len=4, num_seq=500):
    X, y = [], []
    for _ in range(num_seq):
        seq = np.random.choice(NOTE_VOCAB, size=seq_len + 1)
        X.append([note_to_int[n] for n in seq[:-1]])
        y.append(note_to_int[seq[-1]])
    return torch.tensor(X), torch.tensor(y)


def build_and_train_model(X, y, epochs=20):
    input_dim = len(NOTE_VOCAB)
    model = NoteLSTM(input_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model


def generate_notes(model, start_seq=None, length=6):
    model.eval()
    if start_seq is None:
        start_seq = [random.randint(0, len(NOTE_VOCAB) - 1) for _ in range(4)]

    seq = start_seq[:]
    for _ in range(length):
        input_seq = torch.tensor(seq[-4:]).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_seq)
            next_idx = torch.argmax(pred, dim=1).item()
        seq.append(next_idx)

    notes = [int_to_note[i] for i in seq]
    chords = [notes[i:i+3] for i in range(len(notes) - 2)]
    return notes, chords


def update_config(config, model):
    for mode, data in config.items():
        if not isinstance(data, dict):
            print(f"⚠️ Skipping mode '{mode}': expected dict, got {type(data).__name__}")
            continue

        instruments = data.get("instruments", [])
        if not isinstance(instruments, list):
            print(f"⚠️ Skipping instruments in mode '{mode}': expected list, got {type(instruments).__name__}")
            continue

        for instrument in instruments:
            if not isinstance(instrument, dict):
                print(f"⚠️ Skipping instrument: expected dict, got {type(instrument).__name__}")
                continue

            existing_notes = instrument.get("notes", [])
            start = [note_to_int.get(n, random.randint(0, len(NOTE_VOCAB)-1)) for n in existing_notes[:4]]
            while len(start) < 4:
                start.append(random.randint(0, len(NOTE_VOCAB) - 1))

            notes, chords = generate_notes(model, start_seq=start)
            instrument["notes"] = notes
            instrument["chords"] = chords
    return config


def main():
    print("📥 Loading config...")
    config = load_config()

    print("🧠 Generating training data...")
    X, y = generate_training_data()

    print("🎶 Training model...")
    model = build_and_train_model(X, y)

    print("🎼 Updating config with generated notes and chords...")
    updated_config = update_config(config, model)

    print("💾 Saving updated config...")
    save_config(updated_config)
    print("✅ Process complete.")


if __name__ == "__main__":
    main()
