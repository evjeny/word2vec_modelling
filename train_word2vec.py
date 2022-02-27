import os
import time
from typing import Iterator

from config_manager.config import Config
import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import tqdm

from word2vec_model import Word2Vec, optimization_step


class Word2VecConfig(Config):
    texts_folder: str = "dataset_texts"
    word_separator: str = ","
    fold_list_file: str = "folds.txt"

    embedding_dim: int = 300
    context: int = 2
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 5
    device: str = "cpu"

    models_folder: str = "word2vec_models"


def batch_iterator(
        fold_word_indices: list[list[int]], context: int, batch_size: int
) -> Iterator[tuple[list[list[int]], list[int]]]:
    for document in fold_word_indices:
        n = len(document)
        n_samples = n - 2 * context
        n_batches = n_samples // batch_size

        current_position = context
        for _ in range(n_batches):
            x_batch = []
            y_batch = []
            for i in range(batch_size):
                x_batch.append(
                    document[current_position - context: current_position] + \
                    document[current_position + 1: current_position + context + 1]
                )
                y_batch.append(document[current_position])
                current_position += 1
            yield x_batch, y_batch


def train_single_model(fold_words: list[list[str]], config: Word2VecConfig) -> Word2Vec:
    all_words = []
    for words in fold_words:
        all_words.extend(words)
    all_words = set(all_words)

    device = torch.device(config.device)

    model = Word2Vec(all_words, embedding_dim=config.embedding_dim).to(device)
    fold_word_indices = [
        model.sequence_to_indices(document)
        for document in fold_words
    ]
    optimizer = Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        t1 = time.time()
        epoch_loss = 0

        batches = list(batch_iterator(
            fold_word_indices, config.context, config.batch_size
        ))

        for x_batch, y_batch in tqdm.tqdm(batches, desc=f"Epoch {epoch}"):
            x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
            y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            loss = optimization_step(
                model, optimizer, cross_entropy, x_batch, y_batch
            )
            epoch_loss += float(loss)

        t2 = time.time()
        print(f"[Epoch {epoch}/{config.epochs}]: loss={epoch_loss:.4f}, time={t2 - t1} s")

    return model


def read_fold_words(paths: list[str], sep: str) -> list[list[str]]:
    result = []
    for path in paths:
        with open(path) as f:
            result.append(f.read().strip().split(sep))
    return result


def main(config: Word2VecConfig):
    folds_paths = []
    with open(config.fold_list_file) as f:
        for line in f.readlines():
            folds_paths.append([
                os.path.join(config.texts_folder, filename)
                for filename in line.strip().split(",")
            ])

    os.makedirs(config.models_folder, exist_ok=True)
    for model_num, fold in enumerate(folds_paths):
        fold_words = read_fold_words(fold, config.word_separator)
        model = train_single_model(fold_words, config)
        torch.save(model.state_dict(), os.path.join(config.models_folder, f"word2vec_{model_num}"))


if __name__ == '__main__':
    main(Word2VecConfig().parse_arguments("Train Word2Vec on several folds"))
