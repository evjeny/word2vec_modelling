import os

from config_manager.config import Config
from sklearn.datasets import fetch_20newsgroups
import tqdm


class SplitterConfig(Config):
    data_folder: str = "dataset_texts"
    word_separator: str = ","
    to_lower: bool = True
    cache_folder: str = "dataset_cache"


def text2words(text: str, to_lower: bool) -> list[str]:
    words = []
    last_word = ""

    def _add_last_word():
        if len(last_word) > 0:
            words.append(last_word.lower() if to_lower else last_word)

    for character in text:
        if character.isalnum():
            last_word += character
            continue
        _add_last_word()
        last_word = ""
    _add_last_word()

    return words


def main(config: SplitterConfig):
    os.makedirs(config.data_folder, exist_ok=True)

    texts: list[str] = fetch_20newsgroups(subset="all", data_home=config.cache_folder).data
    total_words = 0
    for i, text in enumerate(tqdm.tqdm(texts, desc="processing texts")):
        words = text2words(text, config.to_lower)
        total_words += len(words)
        with open(os.path.join(config.data_folder, str(i)), "w+") as f:
            print(*words, sep=config.word_separator, file=f)

    print(f"Handled {total_words} words")


if __name__ == "__main__":
    application_config = SplitterConfig() \
        .parse_arguments("Prepare datasets for Word2Vec")
    main(application_config)
