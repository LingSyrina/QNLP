import torch
import random
import numpy as np
import csv
import pickle
import os

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
            if len(row) < 4:
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
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=2)

if os.path.exists('sentence2circuit.pkl'):
    print('Loading cached sentence circuits...', flush=True)
    with open('sentence2circuit.pkl', 'rb') as f:
        sentence2circuit = pickle.load(f)
else:
    print('Generating circuits for unique sentences...', flush=True)
    sentence2circuit = {}
    for sentence, diagram in sentence2diagram.items():
        try:
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

# --- Build model over unique circuits only ---
unique_circuits = list(sentence2circuit.values())
print(f"Building model with {len(unique_circuits)} unique circuits.", flush=True)

model = XORSentenceModel.from_diagrams(unique_circuits, probabilities=True, normalize=True)
model.initialise_weights()
model = model.double()

print('Model loaded.', flush=True)

# --- Pairing on-the-fly function ---
def pair_sentences_to_circuits(sentence_pairs, labels):
    new_pairs = []
    new_labels = []
    for (s1, s2), label in zip(sentence_pairs, labels):
        if s1 in sentence2circuit and s2 in sentence2circuit:
            new_pairs.append((sentence2circuit[s1], sentence2circuit[s2]))
            new_labels.append(label)
    return new_pairs, new_labels

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
        optimizer.step()

    if i % 5 == 0:
        dev_mse = mse_loss(dev_pairs, dev_labels_filtered)
        print(f'Epoch {i}: Train Loss = {epoch_loss:.4f}, Dev MSE = {dev_mse:.4f}', flush=True)

        if dev_mse < best['mse']:
            best = {'mse': dev_mse, 'epoch': i}
            torch.save(model.state_dict(), 'xor_model.pth')
        elif i - best['epoch'] >= 10:
            print('Early stopping.', flush=True)
            torch.save(model.state_dict(), 'xor_model.pth')
            break

# --- Load best model and evaluate ---
model.load_state_dict(torch.load('xor_model.pth'))
model = model.double()

print('Best model loaded.', flush=True)
