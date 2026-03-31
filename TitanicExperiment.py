from flautim.pytorch.federated.Experiment import Experiment
import torch
import numpy as np

from collections import OrderedDict
from sklearn.metrics import f1_score


def set_params(model, parameters):
    """Substitui os parâmetros do modelo pelos recebidos do servidor."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extrai os parâmetros do modelo como lista de NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


class TitanicExperiment(Experiment):
    """
    Experimento federado simples para o dataset Titanic.
    """

    def __init__(self, model, dataset, context, **kwargs):
        super(TitanicExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=kwargs.get("lr", 0.01),
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
        self.epochs = kwargs.get("epochs", 1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_size = len(dataset.train_partition)

    def fit(self, parameters, config):
        """
        Método chamado pelo servidor em cada rodada federada.
        """
        set_params(self.model, parameters)

        epochs = config.get("epochs", self.epochs)

        self.model.to(self.device)

        final_loss = 0.0
        final_metrics = {}

        for _ in range(epochs):
            final_loss, final_metrics = self.training_loop(self.dataset.dataloader())

        return get_params(self.model), self.data_size, final_metrics

    def training_loop(self, data_loader):
        """
        Treinamento local no cliente.
        """
        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in data_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(avg_loss), {"ACCURACY": accuracy}

    def validation_loop(self, data_loader):
        """
        Avaliação local no conjunto de validação do cliente.
        """
        self.model.to(self.device)
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        if len(set(all_labels)) > 1 and len(set(all_predictions)) > 1:
            f1 = f1_score(all_labels, all_predictions, average="macro")
        else:
            f1 = 0.0

        return float(avg_loss), {"ACCURACY": accuracy, "F1_SCORE": f1}