import torch
import pickle
from lambeq import BobcatParser, RemoveCupsRewriter, PennyLaneModel, AtomicType, IQPAnsatz
from discopy.rigid import Ty, Diagram, Box
from torch import nn
from discopy.monoidal import Box as MonoidalBox
import re

# --- Load saved circuits ---
with open('SICKrelatednessRun/sentence2circuit.pkl', 'rb') as f:
    sentence2circuit = pickle.load(f)

# --- Load saved diagrams ---
with open('SICKrelatednessRun/sentence2diagram.pkl', 'rb') as f:
    sentence2diagram = pickle.load(f)

# --- Build model ---
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

unique_circuits = list(sentence2circuit.values())
model = XORSentenceModel.from_diagrams(unique_circuits, probabilities=True, normalize=True)
model.load('SICKrelatednessRun/xor_model.lt')
model = model.double()
model.eval()

# --- Expand and Save Box Info + Embeddings ---

# Collect all unique (box_name, dom, cod) from the diagrams
box_keys = set()
for diagram in sentence2diagram.values():
    for box in diagram.boxes:
        box_keys.add((box.name, box.dom, box.cod))

output_lines = {}
for symbol, param in zip(model.symbols, model.weights):
    try:
        # Parse components
        full_symbol = str(symbol)
        parts = full_symbol.split('_')
        word = re.sub(r'†', '', parts[0])
        grammar_info = ''.join(parts[1:])  # rest of the symbol string

        embedding = param.detach().numpy().flatten().tolist()

        # Build nested dictionary
        if word not in output_lines:
            output_lines[word] = {}

        output_lines[word][grammar_info] = embedding

    except Exception as e:
        print(f"[Error] Could not extract from {symbol}: {e}")

# --- Save to TXT file ---
with open('SICKrelatednessRun/box_embeddings.txt', 'w') as f:
    for word, grammars in output_lines.items():
        f.write(f"Word: {word}\n")
        for grammar_type, embedding in grammars.items():
            f.write(f"  Grammar: {grammar_type}\n")
            f.write(f"  Embedding: {embedding}\n")
        f.write("\n")

print("Saved all box embeddings to box_embeddings.txt.")
