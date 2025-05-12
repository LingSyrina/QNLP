# QNLP

## General goal (old git)
```
1. A tangible task (around 100 sentences according to the literature);
2. A task that can be used to compare classical NLP models and QNLP models;
```
## Our dataset
[SICK dataset](https://huggingface.co/datasets/RobZamp/sick): Sharedtask with benchmark dataset, we picked 100 pairs of sentences (length <= 10). See `SICK100` folder for our dataset. See [relative work](https://arxiv.org/pdf/1910.08772).  

## Our task
1. Sentence inference relation: Entail:1, Neutral:0.5, Contradict: 0. For entailment, the directionality matters. 
2. Sentence relatedness: SICK relatedness score clipped to [0,1] interval.  
The data sampling code can be found in `SICK100/SICK_sampling.ipynb`.

## Model architecture
1. Quantum model: Hybrid pipeline with Lambeq circuit + XORNet (Feedforward NN). See `hybrid_pipline.py`
2. Transformer model: MiniLM-L6, 6 Transformer layers (Hidden size: 384, Max position embeddings: 512, Heads: 8). See `classical_run.py`.
   
## Model results
1. Inference task is harder than sentence relatedness for both quantum model and transformer.
<img width="705" alt="image" src="https://github.com/user-attachments/assets/1c76942e-dd73-42b8-9986-8612e80ecd5e" />

<img width="699" alt="image" src="https://github.com/user-attachments/assets/cd864ff7-2367-46d2-8057-430a60b72bd9" />

3. Quantum model is more efficient than transformer model.
   <img width="676" alt="image" src="https://github.com/user-attachments/assets/4975e040-5f23-4147-a0de-8ad9d3ffc7cc" />

## Model generality
By word circuit parameter (RX gate theta) can be found as `box_embedding.txt` in each Task folder (i.e., SICKinferenceRun, SICKrelatednessRun).

1. Word/Phrase level generality: Can the model be trained on sentence relation but learn word relation?
2. Composition level generality: Can the model be trained on set of sentence but generalize to all sentence with similar structures?

## Future research ideas
1. [SemEval-2023 Task](https://raganato.github.io/vwsd/): Ambiguous visual labeling (harder but more interesting?). Baseline can be found [here](https://aclanthology.org/2023.semeval-1.308.pdf) with git [repo](https://github.com/asahi417/visual-wsd-baseline).
