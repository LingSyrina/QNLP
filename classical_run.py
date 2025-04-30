import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import os

# --- Reproducibility ---
SEED = 12
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
#dataset = 'SICKinferenceRun'
dataset = 'SICKrelatednessRun'
mode = 'random'
# --- Config ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
MAX_LEN = 64

# --- Data Loading ---
def read_data(filename):
    labels, sentence_pairs = [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            s1, s2, label = row[0], row[1], float(row[2])
            labels.append(label)
            sentence_pairs.append((s1, s2))
    return labels, sentence_pairs

train_labels, train_data = read_data(f'{dataset}/mc_pair_train_data.csv')
dev_labels, dev_data = read_data(f'{dataset}/mc_pair_dev_data.csv')
test_labels, test_data = read_data(f'{dataset}/mc_pair_test_data.csv')

# --- Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#encoder = AutoModel.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_config(config) #add for random embedding

# --- Dataset ---
class SentencePairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2 = self.pairs[idx]
        return s1, s2, self.labels[idx]

def collate_fn(batch):
    s1, s2, labels = zip(*batch)
    tokens = tokenizer(list(s1), list(s2), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.float)
    return tokens, labels

train_loader = DataLoader(SentencePairDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(SentencePairDataset(dev_data, dev_labels), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(SentencePairDataset(test_data, test_labels), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- Model ---
class SentencePairRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size  # ← fixes the mismatch
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        outputs = self.encoder(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        return self.fc(cls_embeddings).squeeze()


model = SentencePairRegressor(encoder).cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --- Training Loop ---
train_losses, val_mses = [], []
best = {'mse': float('inf'), 'epoch': 0}

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for tokens, labels in train_loader:
        tokens = {k: v.cuda() for k, v in tokens.items()}
        labels = labels.cuda()
        optimizer.zero_grad()
        preds = model(tokens)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for tokens, labels in dev_loader:
            tokens = {k: v.cuda() for k, v in tokens.items()}
            preds = model(tokens).cpu()
            val_preds.extend(preds.numpy())
            val_targets.extend(labels.numpy())
    val_mse = np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)
    val_mses.append(val_mse)

    print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Dev MSE = {val_mse:.4f}")
    if val_mse < best['mse']:
        best = {'mse': val_mse, 'epoch': epoch}
        torch.save(model.state_dict(), f'{dataset}/LLM/{mode}_best_llm_model.pt')
    elif epoch - best['epoch'] >= 5:
        print("Early stopping.")
        break

# --- Final Test Evaluation ---
model.load_state_dict(torch.load(f'{dataset}/LLM/{mode}_best_llm_model.pt'))
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for tokens, labels in test_loader:
        tokens = {k: v.cuda() for k, v in tokens.items()}
        preds = model(tokens).cpu()
        test_preds.extend(preds.numpy())
        test_targets.extend(labels.numpy())

test_mse = np.mean((np.array(test_preds) - np.array(test_targets)) ** 2)
print(f"Best model on test set has MSE: {test_mse:.4f}")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_mses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss / MSE')
plt.title('Training Loss and Validation MSE over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'{dataset}/LLM/{mode}_train_llm.png')
plt.close()

# Save logs
log_df = pd.DataFrame({
    'epoch': list(range(len(train_losses))),
    'train_loss': train_losses,
    'val_mse': val_mses
})
log_df.to_csv(f'{dataset}/LLM/{mode}_training_log_llm.csv', index=False)
print("Training logs saved to training_log_llm.csv and plot to train_llm.png.")
