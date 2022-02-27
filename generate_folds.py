import os
import random

from config_manager.config import Config


class FoldsConfig(Config):
    texts_folder: str = "dataset_texts"
    n_folds: int
    document_overlap: float = 0.3
    fold_list_file: str = "folds.txt"


def main(config: FoldsConfig):
    filenames = os.listdir(config.texts_folder)
    indices = list(range(len(filenames)))
    random.shuffle(indices)

    fold_size = int(
        len(filenames) / (config.n_folds - config.document_overlap * (config.n_folds - 1))
    )
    overlap_size = int(fold_size * config.document_overlap)
    print(f"Fold size = {fold_size} documents")
    print(f"Overlap size = {overlap_size} documents")

    begin_index = 0
    fold_indices: list[list[int]] = []
    for _ in range(config.n_folds):
        fold_indices.append(indices[begin_index: begin_index + fold_size])
        begin_index += (fold_size - overlap_size)

    with open(config.fold_list_file, "w+") as f:
        for fold in fold_indices:
            fold_filenames = [filenames[index] for index in fold]
            print(*fold_filenames, sep=",", end="\n", file=f)


if __name__ == '__main__':
    main(FoldsConfig().parse_arguments("Fold generator"))
