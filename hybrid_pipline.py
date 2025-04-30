import torch
import random
import numpy as np
import csv
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd


from torch import nn
from lambeq import (
    BobcatParser, RemoveCupsRewriter, PennyLaneModel, Dataset, AtomicType, IQPAnsatz
)
from discopy.rigid import Ty

# Set seed for reproducibility
SEED = 12
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- Data Loading Functions ---
def read_data(filename):
    labels, sentences = [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            sentence1, sentence2 = row[0], row[1]
            label = float(row[2])
            labels.append(label)
            sentences.append((sentence1, sentence2))
    return labels, sentences

# --- Load datasets ---
train_labels, train_data = read_data('mc_pair_train_data.csv')
dev_labels, dev_data = read_data('mc_pair_dev_data.csv')
test_labels, test_data = read_data('mc_pair_test_data.csv')

# Unique sentences
train_data_unpaired = [s for pair in train_data for s in pair]
dev_data_unpaired = [s for pair in dev_data for s in pair]
test_data_unpaired = [s for pair in test_data for s in pair]
unique_sentences = set(train_data_unpaired + dev_data_unpaired + test_data_unpaired)

# --- Parser and Rewriter ---
reader = BobcatParser(verbose='text')
rewriter = RemoveCupsRewriter()

# --- Sentence -> Diagram Mapping ---
if os.path.exists('sentence2diagram.pkl'):
    print('Loading cached sentence diagrams...', flush=True)
    with open('sentence2diagram.pkl', 'rb') as f:
        sentence2diagram = pickle.load(f)
else:
    print('Generating diagrams for unique sentences...', flush=True)
    sentence2diagram = {}
    for sentence in unique_sentences:
        if len(sentence.split()) > 10:
            print(f"[Skip] Too long: {sentence[:50]}...", flush=True)
            continue
        try:
            diagram = reader.sentence2diagram(sentence)
            rewritten = rewriter(diagram)
            sentence2diagram[sentence] = rewritten
        except Exception as e:
            print(f"[Warning] Skipping sentence: {sentence} -- {e}", flush=True)
    with open('sentence2diagram.pkl', 'wb') as f:
        pickle.dump(sentence2diagram, f)
    print('Saved sentence diagram cache.', flush=True)

print('All diagrams ready.', flush=True)

# --- Diagram checking (Optional: can comment out after first run) ---
base_types = {'n', 's', 'p'}

def has_unexpected_atomic_type(diagram):
    for box in diagram.boxes:
        all_objects = box.dom @ box.cod
        for t in all_objects:
            if isinstance(t, Ty):
                for atom in t.objects:
                    if not (str(atom).startswith(('n', 's', 'p'))):
                        return True
            else:
                if not (str(t).startswith(('n', 's', 'p'))):
                    return True
    return False

problematic_sentences = [s for s, d in sentence2diagram.items() if has_unexpected_atomic_type(d)]
print(f"Found {len(problematic_sentences)} problematic sentences.", flush=True)

# --- Diagram -> Circuit Mapping ---
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=3)

if os.path.exists('sentence2circuit.pkl'):
    print('Loading cached sentence circuits...', flush=True)
    with open('sentence2circuit.pkl', 'rb') as f:
        sentence2circuit = pickle.load(f)
else:
    print('Generating circuits for unique sentences...', flush=True)
    sentence2circuit = {}
    for sentence, diagram in sentence2diagram.items():
        try:
            # Filter by diagram size
            if len(diagram.dom) > 6 or len(diagram.cod) > 6:
                print(f"[Skip] Too many wires: {sentence}", flush=True)
                continue
            if len(diagram.boxes) > 50:
                print(f"[Skip] Too many boxes: {sentence}", flush=True)
                continue

            # Build circuit only if diagram is manageable
            circuit = ansatz(diagram)
            sentence2circuit[sentence] = circuit
        except Exception as e:
            print(f"[Warning] Skipping sentence for circuit: {sentence} -- {e}", flush=True)
    with open('sentence2circuit.pkl', 'wb') as f:
        pickle.dump(sentence2circuit, f)
    print('Saved sentence circuit cache.', flush=True)

print('Circuits generated.', flush=True)

# --- Model Definition ---
class XORSentenceModel(PennyLaneModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xor_net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, diagram_pairs):
        first_d, second_d = zip(*diagram_pairs)
        evaluated_pairs = torch.cat((
            self.get_diagram_output(first_d),
            self.get_diagram_output(second_d)
        ), dim=1)
        evaluated_pairs = 2 * (evaluated_pairs - 0.5)
        return self.xor_net(evaluated_pairs)

def count_parameters(model):
    xor_net_total = sum(p.numel() for p in model.xor_net.parameters() if p.requires_grad)
    circuit_total = sum(p.numel() for p in model.weights if p.requires_grad)
    total = xor_net_total + circuit_total
    print(f"Total Parameters: {total}", flush=True)
    print(f" - XOR Net (Neural) Parameters: {xor_net_total}", flush=True)
    print(f" - Circuit (Quantum) Parameters: {circuit_total}", flush=True)
    return xor_net_total, circuit_total

# --- Build model over unique circuits only ---
unique_circuits = list(sentence2circuit.values())
print(f"Building model with {len(unique_circuits)} unique circuits.", flush=True)

model = XORSentenceModel.from_diagrams(unique_circuits, probabilities=True, normalize=True)
model.initialise_weights()
model = model.double()

print(f"Model loaded.", flush=True)
count_parameters(model)

# --- Pairing on-the-fly function ---
def pair_sentences_to_circuits(sentence_pairs, labels):
    new_pairs = []
    new_labels = []
    for (s1, s2), label in zip(sentence_pairs, labels):
        if s1 in sentence2circuit and s2 in sentence2circuit:
            new_pairs.append((sentence2circuit[s1], sentence2circuit[s2]))
            new_labels.append(label)
    return new_pairs, new_labels

def log_gradient_norms(model, xor_storage, circuit_storage):
    xor_grad_norm = 0.0
    circ_grad_norm = 0.0

    for p in model.xor_net.parameters():
        if p.grad is not None:
            xor_grad_norm += p.grad.norm().item() ** 2

    for p in model.weights:
        if p.grad is not None:
            circ_grad_norm += p.grad.norm().item() ** 2

    xor_grad_norm = xor_grad_norm ** 0.5
    circ_grad_norm = circ_grad_norm ** 0.5

    xor_storage.append(xor_grad_norm)
    circuit_storage.append(circ_grad_norm)

    # Optional: still print if you want real-time monitoring
    print(f"Gradient Norms -- XOR Net: {xor_grad_norm:.4f}, Circuit Params: {circ_grad_norm:.4f}")

# --- Initialize dataset and optimizer ---
BATCH_SIZE = 50
EPOCHS = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# --- Mse_loss ---
def mse_loss(circs, labels):
    predicted = model(circs)
    labels = torch.DoubleTensor(labels)
    return torch.mean((torch.flatten(predicted) - labels)**2).item()

def accuracy(circs, labels):
    predicted = model(circs)
    return (torch.round(torch.flatten(predicted)) == torch.DoubleTensor(labels)).sum().item() / len(circs)

# --- Training Loop ---
print('Start training...', flush=True)

# Pair sentences and filter labels
train_pairs, train_labels_filtered = pair_sentences_to_circuits(train_data, train_labels)
dev_pairs, dev_labels_filtered = pair_sentences_to_circuits(dev_data, dev_labels)
test_pairs, test_labels_filtered = pair_sentences_to_circuits(test_data, test_labels)

# train_labels_filtered = [float(1.0 if lbl >= 0.5 else 0.0) for lbl in train_labels_filtered]
# dev_labels_filtered = [float(1.0 if lbl >= 0.5 else 0.0) for lbl in dev_labels_filtered]
# test_labels_filtered = [float(1.0 if lbl >= 0.5 else 0.0) for lbl in test_labels_filtered]

# Dataset creation
train_pair_dataset = Dataset(train_pairs, train_labels_filtered, batch_size=BATCH_SIZE)

# --- Tracking Losses ---
train_losses = []
val_mses = []
xor_grad_norms = []
circuit_grad_norms = []


best = {'mse': float('inf'), 'epoch': 0}

for i in range(EPOCHS):
    epoch_loss = 0
    for circuits, labels in train_pair_dataset:
        optimizer.zero_grad()
        predicted = model(circuits)
        loss = torch.nn.functional.mse_loss(
            torch.flatten(predicted), torch.DoubleTensor(labels))
        epoch_loss += loss.item()
        loss.backward()
        log_gradient_norms(model, xor_grad_norms, circuit_grad_norms)
        optimizer.step()


    train_losses.append(epoch_loss)

    dev_mse = mse_loss(dev_pairs, dev_labels_filtered)
    val_mses.append(dev_mse)

    if i % 5 == 0:
        print(f'Epoch {i}: Train Loss = {epoch_loss:.4f}, Dev MSE = {dev_mse:.4f}', flush=True)

    if dev_mse < best['mse']:
        best = {'mse': dev_mse, 'epoch': i}
        model.save('xor_model.lt')
    elif i - best['epoch'] >= 10:
        print('Early stopping.', flush=True)
        model.save('xor_model.lt')
        break

model.load('xor_model.lt')
print(f"best model on test dataset has MSE {mse_loss(test_pairs, test_labels_filtered)}")

# --- Plotting Loss and MSE Curves ---
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_mses)), val_mses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss / MSE')
plt.title('Training Loss and Validation MSE over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('train.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(xor_grad_norms, label='XOR Net Gradient Norm')
plt.plot(circuit_grad_norms, label='Circuit Params Gradient Norm')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Flow: XOR Net vs Circuit Params')
plt.legend()
plt.grid(True)
plt.savefig('gradient_norms.png')
plt.show()

print('Training curve saved as train.png.', flush=True)

# Align step-wise data: number of training steps = number of gradient norm entries
steps = list(range(len(xor_grad_norms)))

# Create DataFrame
# Build step-wise values from epoch-level logs
step_epochs = [i // len(train_pair_dataset) for i in range(len(xor_grad_norms))]
step_train_loss = [
    train_losses[i // len(train_pair_dataset)] if i // len(train_pair_dataset) < len(train_losses) else None
    for i in range(len(xor_grad_norms))
]
step_val_mse = [
    val_mses[i // len(train_pair_dataset)] if i // len(train_pair_dataset) < len(val_mses) else None
    for i in range(len(xor_grad_norms))
]

# Now construct the DataFrame
log_df = pd.DataFrame({
    'step': list(range(len(xor_grad_norms))),
    'epoch': step_epochs,
    'train_loss': step_train_loss,
    'val_mse': step_val_mse,
    'xor_grad_norm': xor_grad_norms,
    'circuit_grad_norm': circuit_grad_norms
})

# Save to CSV
log_df.to_csv('training_log.csv', index=False)
print("Training logs saved to training_log.csv.")
