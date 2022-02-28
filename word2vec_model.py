import json
import os

import torch
import torch.nn as nn
import torch.optim as optim


class Word2VecMapping:
    def __init__(self, words: list[str], unknown_token: str = "<UNK>"):
        self.words = words
        self.unknown_token = unknown_token

        self.word2index = {
            word: index + 1
            for index, word in enumerate(self.words)
        } | {unknown_token: 0}

        self.index2word = {
            index: word
            for word, index in self.word2index.items()
        } | {0: unknown_token}
    
    def get_dict_size(self) -> int:
        return len(self.word2index)
    
    def sequence_to_indices(self, words: list[str]) -> list[int]:
        return [self.word2index.get(word, 0) for word in words]
    
    def indices_to_sequence(self, indices: list[int]) -> list[str]:
        return [self.index2word.get(index, self.unknown_token) for index in indices]
    
    def get_json(self) -> dict:
        return {
            "words": self.words,
            "unknown_token": self.unknown_token
        }
    
    @staticmethod
    def load_json(data: dict):
        return Word2VecMapping(**data)


class Word2Vec(nn.Module):
    def __init__(self, dict_size: int, embedding_dim: int):
        super().__init__()

        self.dict_size = dict_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.dict_size+1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.embedding2word = nn.Linear(
            embedding_dim,
            self.dict_size+1
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(axis=1)
        x = self.embedding2word(x)
        return x
    
    def get_json(self) -> dict:
        return {
            "dict_size": self.dict_size,
            "embedding_dim": self.embedding_dim
        }

    @staticmethod
    def load_json(data: dict):
        return Word2Vec(**data)


def optimization_step(
        model: Word2Vec,
        optimizer: optim.Optimizer,
        loss_function,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
):
    optimizer.zero_grad()

    prediction = model(x_batch)
    loss = loss_function(prediction, y_batch)

    loss.backward()
    optimizer.step()

    return loss


def dump_word2vec(
    folder: str, dump_name: str,
    model: Word2Vec,
    mapping: Word2VecMapping
):
    _get_dump_path = lambda suffix: os.path.join(folder, f"{dump_name}{suffix}")
    mapping_path = _get_dump_path("_mapping.json")
    params_path = _get_dump_path("_params.json")
    weights_path = _get_dump_path("_weights.pth")

    with open(mapping_path, "w+") as f:
        json.dump(mapping.get_json(), f)
    
    with open(params_path, "w+") as f:
        json.dump(model.get_json(), f)
    
    torch.save(model.state_dict(), weights_path)


def load_word2vec(folder: str, dump_name: str) -> tuple[Word2Vec, Word2VecMapping]:
    _get_dump_path = lambda suffix: os.path.join(folder, f"{dump_name}{suffix}")
    mapping_path = _get_dump_path("_mapping.json")
    params_path = _get_dump_path("_params.json")
    weights_path = _get_dump_path("_weights.pth")

    with open(mapping_path) as f:
        mapping = Word2VecMapping.load_json(json.load(f))
    
    with open(params_path) as f:
        model = Word2Vec.load_json(json.load(f))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    return model, mapping
