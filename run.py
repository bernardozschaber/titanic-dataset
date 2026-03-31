from flautim.pytorch.common import run_federated, weighted_average
import flautim as fl

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch

import TitanicDataset
import TitanicModel
import TitanicExperiment
from TitanicExperiment import get_params

# -----------------------------
# Configurações do experimento
# -----------------------------
NUM_CLIENTS = 4
CSV_PATH = "./data/titanic.csv"
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
NUM_ROUNDS = 20


class CustomFedAvg(FedAvg):
    """FedAvg extended to save the best global model per round based on accuracy."""

    def __init__(self, context, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_acc_so_far = 0.0
        self.round_results = []

        # Create timestamped output directory
        run_dir = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.save_path = Path.cwd() / f"outputs/{run_dir}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self._context = context
        self._input_dim = input_dim

    def _update_best_acc(self, server_round, accuracy, parameters):
        """Save model checkpoint when a new best accuracy is found."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            fl.log(f"New best global model found: {accuracy:.6f}")

            ndarrays = parameters_to_ndarrays(parameters)
            params_dict = zip(
                TitanicModel.TitanicModel(
                    self._context, input_dim=self._input_dim, suffix="tmp"
                ).state_dict().keys(),
                ndarrays,
            )
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            file_name = f"model_acc_{accuracy:.6f}_round_{server_round}.pth"
            torch.save(state_dict, self.save_path / file_name)
            fl.log(f"Saved: {file_name}")

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation, track best, and print round summary."""
        loss, metrics = super().evaluate(server_round, parameters)

        accuracy = metrics.get("ACCURACY", 0.0)
        f1 = metrics.get("F1_SCORE", 0.0)

        self.round_results.append({"round": server_round, "accuracy": accuracy, "F1_SCORE": f1})
        self._update_best_acc(server_round, accuracy, parameters)

        # Print per-round table sorted by accuracy
        sorted_results = sorted(self.round_results, key=lambda x: x["accuracy"], reverse=True)
        header = f"{'Round':>6}    {'Accuracy':>10}    {'F1':>10}"
        fl.log(header)
        fl.log("-" * len(header))
        for r in sorted_results:
            fl.log(f"{r['round']:>6}    {r['accuracy']:>10.6f}    {r['f1']:>10.6f}")

        return loss, metrics


def fit_config(server_round: int):
    """
    Configuração enviada para os clientes em cada rodada.
    """
    return {
        "server_round": server_round,
        "epochs": LOCAL_EPOCHS,
    }


def generate_client_fn(context):
    """
    Cria a função que instancia cada cliente federado.
    """

    def create_client_fn(context_flwr: Context):
        cid = int(context_flwr.node_config["partition-id"])

        dataset = TitanicDataset.TitanicDataset(
            csv_path=CSV_PATH,
            client_id=cid,
            num_clients=NUM_CLIENTS,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = TitanicModel.TitanicModel(
            context,
            input_dim=dataset.input_dim,
            suffix=cid,
        )

        experiment = TitanicExperiment.TitanicExperiment(
            model,
            dataset,
            context,
            epochs=LOCAL_EPOCHS,
            lr=0.01,
            momentum=0.9,
        )

        return experiment.to_client()

    return create_client_fn


def evaluate_fn(context):
    """
    Avalia o modelo global em todas as partições de validação dos clientes
    e agrega as métricas de forma ponderada pelo número de amostras.
    """

    def fn(server_round, parameters, config):
        total_examples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0

        per_client_metrics = {}

        for cid in range(NUM_CLIENTS):
            dataset = TitanicDataset.TitanicDataset(
                csv_path=CSV_PATH,
                client_id=cid,
                num_clients=NUM_CLIENTS,
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            model = TitanicModel.TitanicModel(
                context,
                input_dim=dataset.input_dim,
                suffix=cid,
            )
            model.set_parameters(parameters)

            experiment = TitanicExperiment.TitanicExperiment(
                model,
                dataset,
                context,
                epochs=LOCAL_EPOCHS,
                lr=1e-3,
            )

            local_config = dict(config)
            local_config["server_round"] = server_round

            loss, num_examples, metrics = experiment.evaluate(parameters, local_config)

            accuracy = metrics.get("ACCURACY", 0.0)
            f1 = metrics.get("F1_SCORE", 0.0)

            weighted_loss += loss * num_examples
            weighted_accuracy += accuracy * num_examples
            weighted_f1 += f1 * num_examples
            total_examples += num_examples

            per_client_metrics[f"client_{cid}_loss"] = float(loss)
            per_client_metrics[f"client_{cid}_accuracy"] = float(accuracy)
            per_client_metrics[f"client_{cid}_f1"] = float(f1)
            per_client_metrics[f"client_{cid}_examples"] = int(num_examples)

        if total_examples == 0:
            return 0.0, {
                "ACCURACY": 0.0,
                "F1_SCORE": 0.0,
                "total_examples": 0,
            }

        global_loss = weighted_loss / total_examples
        global_accuracy = weighted_accuracy / total_examples
        global_f1 = weighted_f1 / total_examples

        global_metrics = {
            "ACCURACY": float(global_accuracy),
            "F1_SCORE": float(global_f1),
            "total_examples": int(total_examples),
        }

        global_metrics.update(per_client_metrics)

        return float(global_loss), global_metrics

    return fn

def generate_server_fn(context, eval_fn, **kwargs):
    """
    Cria o servidor federado com estratégia FedAvg.
    """

    def create_server_fn(context_flwr: Context):
        # Criamos um dataset temporário só para descobrir input_dim
        temp_dataset = TitanicDataset.TitanicDataset(
            csv_path=CSV_PATH,
            client_id=0,
            num_clients=NUM_CLIENTS,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        net = TitanicModel.TitanicModel(
            context,
            input_dim=temp_dataset.input_dim,
            suffix="server_init",
        )

        ndarrays = get_params(net)
        initial_parameters = ndarrays_to_parameters(ndarrays)

        strategy = CustomFedAvg(
            context=context,
            input_dim=temp_dataset.input_dim,
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            fraction_fit=1.0,         # todos os clientes treinam
            fraction_evaluate=1.0,    # todos os clientes podem avaliar
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )

        config = ServerConfig(num_rounds=NUM_ROUNDS)

        return ServerAppComponents(config=config, strategy=strategy)

    return create_server_fn


if __name__ == "__main__":
    context = fl.init()
    fl.log("Flautim inicializado!")

    client_fn_callback = generate_client_fn(context)
    evaluate_fn_callback = evaluate_fn(context)
    server_fn_callback = generate_server_fn(context, eval_fn=evaluate_fn_callback)

    fl.log("Experimento Titanic federado criado!")

    run_federated(
        client_fn_callback,
        server_fn_callback,
        num_clients=NUM_CLIENTS,
    )