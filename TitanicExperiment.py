from flautim.pytorch.federated.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
from collections import OrderedDict
from math import inf


# ------------------------------------------------------------------
# Funções auxiliares no escopo do módulo (importáveis via from ... import)
# ------------------------------------------------------------------
def set_params(model, parameters):
    """Substitui os parâmetros do modelo pelos parâmetros recebidos."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extrai os parâmetros do modelo como lista de arrays NumPy."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# ------------------------------------------------------------------
# Classe do experimento
# ------------------------------------------------------------------
class TitanicExperiment(Experiment):

    def __init__(self, model, dataset, context, **kwargs):
        super(TitanicExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epochs    = kwargs.get('epochs', 5)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_loss = inf

        # Compatível com TitanicDataset centralizado (usa .features via .train())
        self.data_size = len(dataset.train().features)

    # ------------------------------------------------------------------
    # Power of Choice: epochs=-1 (data_size), 0 (local_loss), N (treino)
    # ------------------------------------------------------------------
    def fit(self, parameters, config):
        set_params(self.model, parameters)
        epochs = config.get("epochs", self.epochs)

        if epochs == -1:
            return parameters, 0, {"data_size": self.data_size}

        if epochs == 0:
            local_loss, _ = self.validation_loop(self.dataset.dataloader(validation=True))
            local_loss += np.random.uniform(low=1e-10, high=1e-9)
            return parameters, 0, {"local_loss": local_loss}

        loss, metrics = self.training_loop(self.dataset.dataloader())
        return get_params(self.model), self.data_size, metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def training_loop(self, data_loader):
        self.model.to(self.device)
        self.model.train()

        running_loss, correct, total = 0.0, 0, 0

        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss    = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total   += y.size(0)

        avg_loss       = running_loss / len(data_loader)
        accuracy       = correct / total if total > 0 else 0.0
        self.last_loss = avg_loss

        return float(avg_loss), {'ACCURACY': accuracy}

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    def validation_loop(self, data_loader):
        self.model.to(self.device)
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in data_loader:
                X, y    = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                running_loss += self.criterion(outputs, y).item()
                _, predicted  = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total   += y.size(0)

        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0.0

        return float(avg_loss), {'ACCURACY': accuracy}