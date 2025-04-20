
import numpy as np
from lambeq import (
    BobcatParser, RemoveCupsRewriter, IQPAnsatz, AtomicType,
    TketModel, QuantumTrainer, SPSAOptimizer,
    Dataset
)
from pytket.extensions.qiskit import AerBackend




def accuracy(y_pred, y_true):
    return (np.round(y_pred) == y_true).mean()

if __name__ == '__main__':
    sentence_A = "A group of kids is playing in a yard and an old man is standing in the background"
    sentence_B = "A group of boys in a yard is playing and a man is standing in the background"
    label = 1  # entailment

    parser = BobcatParser()
    rewriter = RemoveCupsRewriter()

    raw_diagrams = parser.sentences2diagrams([sentence_A, sentence_B])
    rewritten_diagrams = [rewriter(d) for d in raw_diagrams]

    ansatz = IQPAnsatz(
        {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},  # <--- fix here
        n_layers=2,
        n_single_qubit_params=3
    )

    train_circuits = [ansatz(d) for d in rewritten_diagrams]
    train_labels = np.array([[label], [label]])  # shape (2, 1)

    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 8192
    }

    model = TketModel.from_diagrams(train_circuits, backend_config=backend_config)

    BATCH_SIZE = 2
    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)

    bce = lambda y_pred, y_true: -(
        y_true * np.log(y_pred + 1e-10) +
        (1 - y_true) * np.log(1 - y_pred + 1e-10)
    ).mean()



    eval_metrics = {'accuracy': accuracy}

    EPOCHS = 10

    trainer = QuantumTrainer(
        model=model,
        loss_function=bce,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.001 * EPOCHS},
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        log_dir='entailment/logs',
        seed=0
    )

    trainer.fit(train_dataset)