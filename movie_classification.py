# PyTorch is the main deep learning framework we use here.
import torch
# nn (neural network) module contains building blocks like layers, loss functions, etc.
import torch.nn as nn
# DataLoader feeds data to the model in batches; Dataset is a base class for our custom data.
from torch.utils.data import DataLoader, Dataset
# Hugging Face's `datasets` library — gives us easy access to common datasets like IMDB.
from datasets import load_dataset
# A tokenizer splits raw text into individual words/tokens (e.g. "I loved it" → ["i", "loved", "it"]).
from torchtext.data.utils import get_tokenizer
# Builds a vocabulary: a lookup table that maps each word to a unique integer ID.
from torchtext.vocab import build_vocab_from_iterator
# pad_sequence pads shorter sequences with zeros so all sequences in a batch are the same length.
from torch.nn.utils.rnn import pad_sequence
# tqdm shows a progress bar in the terminal so you can see training progress.
from tqdm import tqdm

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Hyperparameters are settings you choose BEFORE training. They control how the model learns.

# Only keep the 25,000 most common words in our vocabulary; rare words are ignored.
MAX_VOCAB_SIZE = 25_000
# Each word is represented as a vector of 128 numbers (its "meaning" in math form).
EMBEDDING_DIM  = 128
# The LSTM's internal memory size — how much information it can hold at each step.
HIDDEN_DIM     = 256
# Stack 2 LSTM layers on top of each other for a deeper (more powerful) model.
N_LAYERS       = 2
# Dropout randomly disables 50% of neurons during training to prevent overfitting
# (overfitting = memorizing training data instead of learning general patterns).
DROPOUT        = 0.5
# Process 64 reviews at a time instead of one-by-one — faster and more stable training.
BATCH_SIZE     = 64
# An epoch = one full pass through the entire training dataset. We do 5 passes total.
N_EPOCHS       = 5
# Cap each review at 500 words; longer reviews are truncated to save memory.
MAX_SEQ_LEN    = 500
# How big each weight update step is. 1e-3 = 0.001 — a common safe starting value.
LEARNING_RATE  = 1e-3

# Use the GPU if one is available (much faster), otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data ───────────────────────────────────────────────────────────────────────

print("Loading IMDB dataset...")
# Downloads the IMDB movie review dataset (25k train, 25k test).
# Each sample has a "text" (the review) and a "label" (0=negative, 1=positive).
dataset = load_dataset("imdb")

# "basic_english" tokenizer: lowercases text and splits on spaces/punctuation.
# Example: "It's great!" → ["it", "'s", "great", "!"]
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data):
    # A generator function — yields one tokenized review at a time.
    # Generators are memory-efficient because they don't load everything at once.
    for item in data:
        yield tokenizer(item["text"])


print("Building vocabulary...")
# Scans all training reviews and collects the most common MAX_VOCAB_SIZE words.
vocab = build_vocab_from_iterator(
    yield_tokens(dataset["train"]),
    max_tokens=MAX_VOCAB_SIZE,
    # Special tokens:
    #   <pad> — used to fill shorter sequences to match the longest one in a batch.
    #   <unk> — used for any word not in the vocabulary (unknown word).
    specials=["<pad>", "<unk>"],
)
# Any word not found in the vocab will map to the <unk> index by default.
vocab.set_default_index(vocab["<unk>"])

# Store the integer ID for the padding token so we can reference it later.
PAD_IDX = vocab["<pad>"]


# A PyTorch Dataset wraps our raw data and tells PyTorch how to access individual samples.
class IMDBDataset(Dataset):
    def __init__(self, data):
        # Store the raw list of {"text": ..., "label": ...} dictionaries.
        self.data = data

    def __len__(self):
        # PyTorch calls this to know how many samples exist in the dataset.
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch calls this to fetch a single sample by index.

        # Tokenize the review text and keep only the first MAX_SEQ_LEN tokens.
        tokens = tokenizer(self.data[idx]["text"])[:MAX_SEQ_LEN]
        # Convert each token string to its integer ID using the vocabulary.
        ids    = vocab(tokens)
        # 0 = negative review, 1 = positive review.
        label  = self.data[idx]["label"]
        # Return as PyTorch tensors — the format the model expects.
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    # Called by DataLoader to combine individual samples into a batch.
    # `batch` is a list of (ids_tensor, label_tensor) tuples.

    # Unzip the list of tuples into two separate tuples: all texts, all labels.
    texts, labels = zip(*batch)
    # Pad all text tensors to the same length (the longest one in the batch).
    # batch_first=True → output shape is (batch_size, seq_len) instead of (seq_len, batch_size).
    # Shorter sequences are padded with PAD_IDX (0) at the end.
    return pad_sequence(texts, batch_first=True, padding_value=PAD_IDX), torch.stack(labels)


# DataLoader wraps the dataset and handles batching, shuffling, and calling collate_fn.
train_loader = DataLoader(
    IMDBDataset(dataset["train"]),
    batch_size=BATCH_SIZE,
    shuffle=True,          # Shuffle training data each epoch so the model doesn't memorize order.
    collate_fn=collate_fn,
)
test_loader  = DataLoader(
    IMDBDataset(dataset["test"]),
    batch_size=BATCH_SIZE,
    shuffle=False,         # No need to shuffle test data — we just want to evaluate it.
    collate_fn=collate_fn,
)


# ── Model ──────────────────────────────────────────────────────────────────────
# We inherit from nn.Module — the base class for all PyTorch models.
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()  # Always call the parent class constructor first.

        # Embedding layer: converts integer word IDs into dense float vectors.
        # Think of it as a lookup table: word_id → vector of numbers representing meaning.
        # padding_idx tells the model to treat <pad> tokens as having no meaning (zero vector).
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM (Long Short-Term Memory): reads the sequence of word vectors one step at a time
        # and builds up a "summary" of what it has read, remembering important context.
        # bidirectional=True means it reads the text both forward AND backward,
        # so the final representation captures context from both directions.
        self.lstm = nn.LSTM(
            embedding_dim,      # Input size: each word vector has this many numbers.
            hidden_dim,         # Hidden state size: the LSTM's internal memory width.
            num_layers=n_layers,# Stack multiple LSTM layers for deeper representation.
            bidirectional=True, # Read sequence left→right AND right→left.
            dropout=dropout if n_layers > 1 else 0,  # Dropout between LSTM layers.
            batch_first=True,   # Input shape is (batch, seq_len, features).
        )

        # Fully connected (linear) layer: maps the LSTM output to 2 class scores.
        # hidden_dim * 2 because we concatenate forward + backward hidden states.
        # Output size 2: one score for "Negative", one score for "Positive".
        self.fc      = nn.Linear(hidden_dim * 2, 2)

        # Dropout layer: randomly zeros out neurons during training to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # This method defines the data flow through the model (the "forward pass").
        # `text` shape: (batch_size, seq_len) — integers representing word IDs.

        # 1. Look up word embeddings, then apply dropout.
        #    Shape becomes: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text))

        # 2. Pass embeddings through the LSTM.
        #    We only care about `hidden` (the final hidden state), not the full output sequence.
        #    `hidden` shape: (n_layers * 2, batch_size, hidden_dim)
        #    The *2 is because it's bidirectional (forward + backward for each layer).
        _, (hidden, _) = self.lstm(embedded)

        # 3. Grab the last layer's hidden states from both directions and concatenate them.
        #    hidden[-2] = last forward layer's hidden state
        #    hidden[-1] = last backward layer's hidden state
        #    After concat, shape: (batch_size, hidden_dim * 2)
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))

        # 4. Pass through the linear layer to get raw scores (logits) for each class.
        #    Output shape: (batch_size, 2)
        return self.fc(hidden)


# Instantiate the model and move it to GPU (or CPU).
model     = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, PAD_IDX).to(device)
# Adam optimizer adjusts the model's weights after each batch to reduce the loss.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# CrossEntropyLoss measures how wrong the model's predictions are.
# It combines a softmax (converts raw scores to probabilities) with a log-loss penalty.
criterion = nn.CrossEntropyLoss()


# ── Training / Evaluation ──────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None):
    # Reuse this function for both training (pass optimizer) and evaluation (pass None).
    training = optimizer is not None

    # model.train() enables dropout and other training-specific behaviours.
    # model.eval() disables them so evaluation is deterministic.
    model.train() if training else model.eval()

    total_loss, total_correct = 0.0, 0

    # During training we need gradients (for backprop); during eval we don't.
    # torch.no_grad() saves memory and speeds things up when we don't need gradients.
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for texts, labels in tqdm(loader, desc="Train" if training else "Eval ", leave=False):
            # Move data to the same device as the model (GPU or CPU).
            texts, labels = texts.to(device), labels.to(device)

            # Forward pass: feed the batch through the model to get predictions.
            preds = model(texts)

            # Compute how wrong the predictions are compared to the true labels.
            loss  = criterion(preds, labels)

            if training:
                # Zero out gradients from the previous batch — they accumulate by default.
                optimizer.zero_grad()
                # Backpropagation: compute gradients of the loss w.r.t. every parameter.
                loss.backward()
                # Clip gradients to a max norm of 1.0 — prevents "exploding gradients"
                # (a problem where updates become too large and destabilise training).
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update the model's weights using the computed gradients.
                optimizer.step()

            # Accumulate loss and correct predictions over the whole epoch.
            total_loss    += loss.item()
            # argmax picks the class with the highest score; compare with true labels.
            total_correct += (preds.argmax(dim=1) == labels).sum().item()

    # Average loss per batch and fraction of correctly classified samples.
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


print(f"\nTraining on {device}\n{'─'*60}")
# Loop over each epoch (one full pass through the training data).
for epoch in range(1, N_EPOCHS + 1):
    # Train for one epoch and get the training loss/accuracy.
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
    # Evaluate on the test set (no optimizer passed → evaluation mode).
    te_loss, te_acc = run_epoch(model, test_loader)
    print(
        f"Epoch {epoch}/{N_EPOCHS}  |  "
        f"Train  loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%  |  "
        f"Test   loss={te_loss:.4f}  acc={te_acc*100:.2f}%"
    )

# Save the trained model weights to disk so you can reload them later without retraining.
torch.save(model.state_dict(), "sentiment_model.pt")
print("\nModel saved → sentiment_model.pt")


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(review: str) -> tuple[str, float]:
    # Switch to eval mode: disables dropout so predictions are consistent.
    model.eval()

    # Tokenize and truncate the input review, then convert tokens to integer IDs.
    ids    = vocab(tokenizer(review)[:MAX_SEQ_LEN])

    # Convert to a tensor and add a batch dimension with unsqueeze(0).
    # The model expects shape (batch_size, seq_len); unsqueeze makes it (1, seq_len).
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():  # No gradient needed for inference.
        # Get raw scores (logits) for each class: [score_negative, score_positive].
        logits = model(tensor)
        # Softmax converts the raw scores into probabilities that sum to 1.
        probs  = torch.softmax(logits, dim=1)[0]
        # Pick the class with the highest score (0=Negative, 1=Positive).
        pred   = logits.argmax(dim=1).item()

    # Return a human-readable label and the model's confidence for that label.
    return ("Positive" if pred == 1 else "Negative"), probs[pred].item()


# A few sample reviews to demonstrate the model after training.
samples = [
    "This movie was absolutely fantastic! The acting was superb and the plot was gripping.",
    "Terrible film. Boring, predictable, and a complete waste of time.",
    "Some nice cinematography but the story falls flat in the second half.",
]

print(f"\n{'─'*60}\nSample Predictions\n{'─'*60}")
for review in samples:
    sentiment, confidence = predict(review)
    # Print the first 70 characters of the review plus the model's verdict.
    print(f"  Review    : {review[:70]}...")
    print(f"  Prediction: {sentiment}  ({confidence:.1%} confidence)\n")
