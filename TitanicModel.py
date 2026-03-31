from flautim.pytorch.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanicModel(Model):
    """
    MLP simples para classificação binária no dataset Titanic.

    Entrada:
        features tabulares já pré-processadas

    Saída:
        2 classes -> 0 (não sobreviveu), 1 (sobreviveu)
    """

    def __init__(self, context, input_dim: int, **kwargs) -> None:
        super(TitanicModel, self).__init__(
            context,
            name="TitanicMLP",
            version=1,
            id=1,
            **kwargs
        )

        self.hidden1 = nn.Linear(input_dim, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output_layer(x)
        return x