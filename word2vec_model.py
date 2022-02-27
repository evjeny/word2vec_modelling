import torch
import torch.nn as nn
import torch.optim as optim


class Word2Vec(nn.Module):
    def __init__(self, words: set[str], embedding_dim: int):
        super().__init__()

        self.word2index = {
            word: i + 1 for i, word in enumerate(words)
        }
        self.index2word = {
            index: word for word, index in self.word2index.items()
        }

        self.embedding = nn.Embedding(
            num_embeddings=len(words) + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.embedding2word = nn.Linear(
            embedding_dim,
            len(words) + 1
        )

    def sequence_to_indices(self, words: list[str]) -> list[int]:
        return [self.word2index.get(word, 0) for word in words]

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(axis=1)
        x = self.embedding2word(x)
        return x


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
