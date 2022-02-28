import os
import time
from typing import Iterator

from config_manager.config import Config
from config_manager.variable_parsers import ListType
import torch
from torch.optim import Adam
from torch.nn.functional import l1_loss
import tqdm

from word2vec_model import Word2Vec, Word2VecMapping, load_word2vec
from mapper_model import Mapper, optimization_step, dump_mapper


class MapperConfig(Config):
    folder: str = "word2vec_models"
    dump_names: ListType[str]
    print_common_parts: bool = True

    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 5
    device: str = "cpu"

    models_folder: str = "mapping_models"


def find_common_words(
        words_a: list[str], words_b: list[str],
        print_part: bool
) -> list[str]:
    words_a = set(words_a)
    words_b = set(words_b)
    common_words = words_a.intersection(words_b)

    if print_part:
        all_words = words_a.union(words_b)
        print(f"Common words: {len(common_words)} ({len(common_words) / len(all_words) * 100:.2f}%)")

    return list(common_words)


def batch_iterator(
        indices_a: list[int],
        indices_b: list[int],
        batch_size: int
) -> Iterator[tuple[list[int], list[int]]]:
    min_size = min(len(indices_a), len(indices_b))
    for i in range(0, min_size, batch_size):
        yield indices_a[i: i + batch_size], indices_b[i: i + batch_size]


def train_single_mapper(
        model_a: Word2Vec, model_b: Word2Vec,
        word_indices_a: list[int], word_indices_b: list[int],
        config: MapperConfig
) -> Mapper:
    device = torch.device(config.device)
    model = Mapper(
        model_a.embedding_dim,
        model_b.embedding_dim
    ).to(device)
    model_a.to(device)
    model_b.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        t1 = time.time()
        epoch_loss = 0

        batches = list(batch_iterator(
            word_indices_a, word_indices_b, config.batch_size
        ))
        for indices_batch_a, indices_batch_b in tqdm.tqdm(batches, desc=f"Epoch {epoch}"):
            embeddings_a = model_a.get_embedding(
                torch.reshape(
                    torch.tensor(indices_batch_a),
                    (-1, 1)
                ).to(device)
            )
            embeddings_b = model_b.get_embedding(
                torch.reshape(
                    torch.tensor(indices_batch_b),
                    (-1, 1)
                ).to(device)
            )
            loss = optimization_step(
                model, optimizer, l1_loss, embeddings_a, embeddings_b
            )

            epoch_loss += float(loss)

        t2 = time.time()
        mean_loss = epoch_loss / len(batches)
        print(f"[Epoch {epoch}/{config.epochs}]: loss={mean_loss:.4f}, time={t2 - t1} s")

    return model


def main(config: MapperConfig):
    models: list[tuple[Word2Vec, Word2VecMapping]] = [
        load_word2vec(config.folder, dump_name)
        for dump_name in config.dump_names
    ]

    os.makedirs(config.models_folder, exist_ok=True)
    index_pairs = [
        (i, j) for i in range(len(models))
        for j in range(i + 1, len(models))
    ]
    for index_a, index_b in index_pairs:
        name_a = config.dump_names[index_a]
        name_b = config.dump_names[index_b]
        model_a, mapping_a = models[index_a]
        model_b, mapping_b = models[index_b]

        if config.print_common_parts:
            print(f"Dumps {name_a} and {name_b}")

        common_words = find_common_words(mapping_a.words, mapping_b.words, config.print_common_parts)
        indices_a = mapping_a.sequence_to_indices(common_words)
        indices_b = mapping_b.sequence_to_indices(common_words)

        model = train_single_mapper(model_a, model_b, indices_a, indices_b, config)
        dump_mapper(config.models_folder, f"mapper_{name_a}_{name_b}", model)


if __name__ == "__main__":
    main(MapperConfig().parse_arguments("Find mappings between different Word2Vec models"))
