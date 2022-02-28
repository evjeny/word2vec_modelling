import json
import os

import torch
import torch.nn as nn
import torch.optim as optim


class Mapper(nn.Module):
    def __init__(self, embedding_dim_a: int, embedding_dim_b: int):
        super().__init__()

        self.embedding_dim_a = embedding_dim_a
        self.embedding_dim_b = embedding_dim_b

        self.a2b = nn.Linear(
            self.embedding_dim_a,
            self.embedding_dim_b,
            bias=False
        )
        self.b2a = nn.Linear(
            self.embedding_dim_b,
            self.embedding_dim_a,
            bias=False
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_back = self.b2a(self.a2b(a))
        b_back = self.a2b(self.b2a(b))
        return a_back, b_back

    def get_mapping_a2b(self) -> torch.Tensor:
        return self.a2b.weight

    def get_mapping_b2a(self) -> torch.Tensor:
        return self.b2a.weight

    def get_json(self) -> dict:
        return {
            "embedding_dim_a": self.embedding_dim_a,
            "embedding_dim_b": self.embedding_dim_b
        }

    @staticmethod
    def load_json(data: dict):
        return Mapper(**data)


def optimization_step(
        model: Mapper,
        optimizer: optim.Optimizer,
        loss_function,
        a_batch: torch.Tensor,
        b_batch: torch.Tensor,
        gamma: float = 0.5
):
    optimizer.zero_grad()

    a_back_batch, b_back_batch = model(a_batch, b_batch)
    loss_a2b = loss_function(a_back_batch, a_batch)
    loss_b2a = loss_function(b_back_batch, b_batch)
    loss = loss_a2b * (1 - gamma) + loss_b2a * gamma

    loss.backward()
    optimizer.step()

    return loss


def dump_mapper(
        folder: str, dump_name: str,
        model: Mapper
):
    params_path = os.path.join(folder, f"{dump_name}_params.json")
    weights_path = os.path.join(folder, f"{dump_name}_weights.pth")

    with open(params_path, "w+") as f:
        json.dump(model.get_json(), f)

    torch.save(model.state_dict(), weights_path)


def load_mapper(folder: str, dump_name: str) -> Mapper:
    params_path = os.path.join(folder, f"{dump_name}_params.json")
    weights_path = os.path.join(folder, f"{dump_name}_weights.pth")

    with open(params_path) as f:
        model = Mapper.load_json(json.load(f))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    return model
