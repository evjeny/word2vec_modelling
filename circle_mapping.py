from config_manager.config import Config
from config_manager.variable_parsers import ListType
import numpy as np
import torch
import tqdm

from mapper_model import Mapper, load_mapper
from word2vec_model import Word2Vec, Word2VecMapping, load_word2vec


class CircleMappingConfig(Config):
    mapping_folder: str = "mapping_models"
    mapping_sequence: ListType[str]
    word2vec_folder: str = "word2vec_models"
    word2vec_dump: str

    test_all: bool = False
    device: str = "cpu"
    hist_width: int = 40
    hist_segments: int = 11


class MappingSequence:
    def __init__(self):
        self.sequence: list[tuple[Mapper, str]] = []

    def add(self, mapper: Mapper, method: str):
        if method not in ["a2b", "b2a"]:
            raise Exception("Unknown mapping method, should be one of: a2b, b2a")
        self.sequence.append((mapper, method))
        return self

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        for mapper, method in self.sequence:
            if method == "a2b":
                embedding = mapper.forward_a2b(embedding)
            else:
                embedding = mapper.forward_b2a(embedding)
        return embedding


def load_mappers(sequence: list[str], mapping_folder: str, device: torch.device) -> MappingSequence:
    result_sequence = MappingSequence()
    for description in sequence:
        dump_name, method = description.split(":")
        result_sequence.add(
            load_mapper(mapping_folder, dump_name).to(device),
            method
        )
    return result_sequence


def get_word_similarity(
        word: str,
        word2vec_model: Word2Vec,
        word2vec_mapping: Word2VecMapping,
        mapping_sequence: MappingSequence,
        device: torch.device
) -> float:
    input_tensor = torch.tensor([[word2vec_mapping.word2index[word]]], dtype=torch.long, device=device)
    source_embedding = word2vec_model.get_embedding(input_tensor)
    result_embedding = mapping_sequence.forward(source_embedding)

    source_embedding = source_embedding.detach().cpu().numpy().reshape(-1)
    result_embedding = result_embedding.detach().cpu().numpy().reshape(-1)

    return np.dot(source_embedding, result_embedding) / (
            np.linalg.norm(source_embedding) * np.linalg.norm(result_embedding)
    )


def test_single_word(
        word2vec_model: Word2Vec, word2vec_mapping: Word2VecMapping,
        mapping_sequence: MappingSequence,
        device: torch.device
):
    while True:
        word = input("Enter word: ")
        if word not in word2vec_mapping.word2index:
            print("Unknown word!")
            continue

        similarity = get_word_similarity(
            word,
            word2vec_model, word2vec_mapping,
            mapping_sequence, device
        )
        print(f"Similarity: {similarity:.4f}")


def test_all_words(
        word2vec_model: Word2Vec, word2vec_mapping: Word2VecMapping,
        mapping_sequence: MappingSequence,
        device: torch.device,
        hist_width: int,
        hist_segments: int
):
    similarities = np.array([
        get_word_similarity(
            word, word2vec_model, word2vec_mapping, mapping_sequence, device
        )
        for word in tqdm.tqdm(word2vec_mapping.word2index.keys(), desc="converting embeddings")
    ])

    threshes = np.linspace(-1, 1, hist_segments)
    representation: list[tuple[str, int]] = []
    for begin, end in zip(threshes[:-1], threshes[1:]):
        representation.append((
            f"[{begin:5.2f}; {end:5.2f}]",
            ((similarities >= begin) & (similarities <= end)).sum()
        ))
    max_value = max([count for _, count in representation])
    for range_name, count in representation:
        print(f"{range_name}\t{'#' * int(count * hist_width / max_value)} ({count})")


def main(config: CircleMappingConfig):
    device = torch.device(config.device)

    model, word_mapper = load_word2vec(config.word2vec_folder, config.word2vec_dump)
    model.to(device)

    mapping_sequence = load_mappers(config.mapping_sequence, config.mapping_folder, device)

    if config.test_all:
        test_all_words(model, word_mapper, mapping_sequence, device, config.hist_width, config.hist_segments)
    else:
        test_single_word(model, word_mapper, mapping_sequence, device)


if __name__ == '__main__':
    main(CircleMappingConfig().parse_arguments("Test embedding combination"))
