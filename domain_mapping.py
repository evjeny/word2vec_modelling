from config_manager.config import Config
from config_manager.variable_parsers import ListType
import numpy as np
import torch
import tqdm

from mapper_model import Mapper, load_mapper
from word2vec_model import Word2Vec, Word2VecMapping, load_word2vec


class DomainMappingConfig(Config):
    mapping_folder: str = "mapping_models"
    mapping_dump: str
    mapping_mode: str
    word2vec_folder: str = "word2vec_models"
    word2vec_dump_a: str
    word2vec_dump_b: str

    device: str = "cpu"


def get_word_similarity(
        word: str,
        word2vec_model_a: Word2Vec,
        word2vec_mapping_a: Word2VecMapping,
        word2vec_model_b: Word2Vec,
        word2vec_mapping_b: Word2VecMapping,
        mapping_model: Mapper,
        mapping_mode: str,
        device: torch.device
) -> float:
    source_embedding = word2vec_model_a.get_embedding(
        torch.tensor([[word2vec_mapping_a.word2index[word]]], dtype=torch.long, device=device)
    )
    target_embedding = word2vec_model_b.get_embedding(
        torch.tensor([[word2vec_mapping_b.word2index[word]]], dtype=torch.long, device=device)
    )
    if mapping_mode == "a2b":
        mapped_embedding = mapping_model.forward_a2b(source_embedding)
    elif mapping_mode == "b2a":
        mapped_embedding = mapping_model.forward_b2a(source_embedding)
    else:
        raise Exception("mapping_mode should be one of: a2b, b2a")

    mapped_embedding = mapped_embedding.detach().cpu().numpy().reshape(-1)
    target_embedding = target_embedding.detach().cpu().numpy().reshape(-1)

    return np.dot(mapped_embedding, target_embedding) / (
            np.linalg.norm(mapped_embedding) * np.linalg.norm(target_embedding)
    )


def plot_ascii_hist(
        values: np.ndarray,
        data_range: tuple[float, float],
        n_segments: int,
        bar_width: int,
        label_width: int,
        float_width: int = 2
):
    empty_label_width = len("[; ]")
    begin_width = (label_width - empty_label_width) // 2
    end_width = label_width - empty_label_width - begin_width

    threshes = np.linspace(data_range[0], data_range[1], n_segments)
    labels: list[str] = []
    counts: list[int] = []

    for begin, end in zip(threshes[:-1], threshes[1:]):
        labels.append(
            f"[{begin:{begin_width}.{float_width}f}; {end:{end_width}.{float_width}f}]"
        )
        counts.append(
            ((values >= begin) & (values < end)).sum()
        )

    max_count = max(counts)
    for label, count in zip(labels, counts):
        print(f"{label} {'#' * int(count * bar_width / max_count)} ({count})")


def test_embedding_similarities(
        words: list[str],
        word2vec_model_a: Word2Vec,
        word2vec_mapping_a: Word2VecMapping,
        word2vec_model_b: Word2Vec,
        word2vec_mapping_b: Word2VecMapping,
        mapping_model: Mapper,
        mapping_mode: str,
        device: torch.device
):
    similarities = np.array([
        get_word_similarity(
            word,
            word2vec_model_a, word2vec_mapping_a,
            word2vec_model_b, word2vec_mapping_b,
            mapping_model, mapping_mode, device
        )
        for word in tqdm.tqdm(words, desc="processing words")
    ])

    plot_ascii_hist(similarities, data_range=(-1, 1), n_segments=10, bar_width=40, label_width=5)


def get_common_words(word2vec_mapping_a: Word2VecMapping, word2vec_mapping_b: Word2VecMapping) -> list[str]:
    return list(set(word2vec_mapping_a.words).intersection(set(word2vec_mapping_b.words)))


def main(config: DomainMappingConfig):
    if config.mapping_mode not in ["a2b", "b2a"]:
        raise Exception("Mapping mode should be one of: a2b, b2a")

    device = torch.device(config.device)

    model_a, word_mapping_a = load_word2vec(config.word2vec_folder, config.word2vec_dump_a)
    model_a.to(device)
    model_b, word_mapping_b = load_word2vec(config.word2vec_folder, config.word2vec_dump_b)
    model_b.to(device)

    mapper = load_mapper(config.mapping_folder, config.mapping_dump)
    mapper.to(device)

    test_embedding_similarities(
        get_common_words(word_mapping_a, word_mapping_b),
        model_a, word_mapping_a,
        model_b, word_mapping_b,
        mapper, config.mapping_mode,
        device
    )


if __name__ == '__main__':
    main(DomainMappingConfig().parse_arguments("Test embedding combination"))
