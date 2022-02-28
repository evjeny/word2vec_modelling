from config_manager.config import Config
import torch

from word2vec_model import load_word2vec


class InferenceConfig(Config):
    models_folder: str = "word2vec_models"
    dump_name: str
    top_n: int = 3


def main(config: InferenceConfig):
    model, mapping = load_word2vec(config.models_folder, config.dump_name)

    while True:
        words = input("Enter words: ").split()
        indices = mapping.sequence_to_indices(words)

        input_tensor = torch.tensor([indices], dtype=torch.long)
        output_probas = model(input_tensor)

        top_indices = torch.argsort(
            output_probas[0], descending=True
        )[:config.top_n].numpy().astype(int)
        closest_words = mapping.indices_to_sequence(top_indices)
        print(closest_words)


if __name__ == "__main__":
    main(InferenceConfig().parse_arguments("Load inference model"))
