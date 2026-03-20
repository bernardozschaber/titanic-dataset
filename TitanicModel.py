from flautim.pytorch.Model import Model
import torch

class TitanicModel(Model):
    def __init__(self, context, num_classes=2, **kwargs):
        super(TitanicModel, self).__init__(context, name="TITANIC-NN", **kwargs)

        # 11 entradas agora
        self.c1 = torch.nn.Linear(11, 32)
        self.c2 = torch.nn.Linear(32, 16)
        self.c3 = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = self.c3(x)
        return x