from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch.federated import Experiment
import TitanicDataset, TitanicModel, TitanicExperiment
from TitanicExperiment import get_params

import flautim as fl
import pandas as pd
import numpy as np
import torch

from flwr.common import Context, ndarrays_to_parameters, Parameters, FitIns
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, DifferentialPrivacyServerSideFixedClipping

NUM_PARTITIONS = 10

def load_and_preprocess():
    titanic = pd.read_csv("./data/Titanic-Dataset.csv")
    titanic['Sex']        = titanic['Sex'].map({'male': 0, 'female': 1})
    titanic['Age']        = titanic['Age'].fillna(titanic['Age'].median())
    titanic['Fare']       = titanic['Fare'].fillna(0.0)
    titanic['Title']      = titanic['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    titanic['Title']      = titanic['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}).fillna(4)
    titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic['IsAlone']    = (titanic['FamilySize'] == 1).astype(int)
    titanic['Embarked']   = titanic['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    titanic['Deck']       = titanic['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    titanic['Deck']       = pd.factorize(titanic['Deck'])[0]
    titanic = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                        'Title', 'FamilySize', 'IsAlone', 'Embarked', 'Deck']]
    return titanic.sample(frac=1, random_state=42).reset_index(drop=True)

def dirichlet_partition(df, n_clients, alpha=0.3, seed=42):
    rng    = np.random.default_rng(seed)
    labels = df['Survived'].values
    classes = np.unique(labels)
    client_indices = [[] for _ in range(n_clients)]
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
        splits = (proportions * len(cls_indices)).astype(int)
        splits[-1] = len(cls_indices) - splits[:-1].sum()
        start = 0
        for cid, count in enumerate(splits):
            client_indices[cid].extend(cls_indices[start:start + count].tolist())
            start += count
    return [df.iloc[idx].reset_index(drop=True) for idx in client_indices]

class MyStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = []
        self.d = 5
        self.m = self.min_fit_clients

    def configure_fit(self, server_round, parameters, client_manager):
        client_manager.wait_for(self.m)
        available_clients = list(client_manager.clients.values())

        print(f"Round {server_round} — clientes: {[c.cid for c in available_clients]}")

        if not available_clients:
            return []

        if self.p == [] or len(self.p) != len(available_clients):
            data_sizes = {
                client.cid: client.fit(
                    FitIns(parameters, {"epochs": -1}), timeout=60, group_id=str(client.cid)
                ).metrics.get("data_size", 1)
                for client in available_clients
            }
            total = sum(data_sizes.values()) or 1
            p_np  = np.array([size / total for size in data_sizes.values()], dtype=float)
            p_np  = np.clip(p_np, 1e-10, None)
            p_np /= p_np.sum()
            self.p = p_np.tolist()

        n_candidates = min(self.d, len(available_clients))
        candidate_clients = np.random.choice(
            available_clients, size=n_candidates, p=self.p, replace=False
        )
        local_losses = {
            client.cid: client.fit(
                FitIns(parameters, {"epochs": 0}), timeout=60, group_id=str(client.cid)
            ).metrics.get("local_loss", float("inf"))
            for client in candidate_clients
        }
        selected_cids = sorted(local_losses, key=local_losses.get, reverse=True)[:self.m]
        print(f"  Selecionados: {selected_cids}")
        return [(client_manager.clients.get(cid), FitIns(parameters, {})) for cid in selected_cids]

def fit_config(server_round):
    return {"server_round": server_round}

def generate_server_fn(context, eval_fn):
    def create_server_fn(context_flwr: Context):
        net               = TitanicModel.TitanicModel(context, num_classes=2, suffix=0)
        global_model_init = ndarrays_to_parameters(get_params(net))
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy=MyStrategy(
                evaluate_fn=eval_fn,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average,
                initial_parameters=global_model_init,
                fraction_fit=0.3,
                min_fit_clients=3,
                min_available_clients=3,  # <-- fix: aguarda clientes disponíveis
            ),
            noise_multiplier=0.1,
            clipping_norm=1.0,
            num_sampled_clients=3,
        )
        return ServerAppComponents(config=ServerConfig(num_rounds=50), strategy=strategy)
    return create_server_fn

def generate_client_fn(context):
    def create_client_fn(context_flwr: Context):
        global partitions
        cid     = int(context_flwr.node_config["partition-id"])
        dataset = TitanicDataset.TitanicDataset(partitions[cid])
        model   = TitanicModel.TitanicModel(context, num_classes=2, suffix=cid)
        return TitanicExperiment.TitanicExperiment(model, dataset, context).to_client()
    return create_client_fn

def evaluate_fn(context):
    def fn(server_round, parameters, config):
        global partitions
        model      = TitanicModel.TitanicModel(context, num_classes=2, suffix="FL-Global")
        model.set_parameters(parameters)
        dataset    = TitanicDataset.TitanicDataset(partitions[0])
        experiment = TitanicExperiment.TitanicExperiment(model, dataset, context)
        config["server_round"] = server_round
        loss, _, return_dic    = experiment.evaluate(parameters, config)
        return loss, return_dic
    return fn

titanic_df = load_and_preprocess()
partitions = dirichlet_partition(titanic_df, n_clients=NUM_PARTITIONS, alpha=0.3, seed=42)

if __name__ == '__main__':
    context = fl.init()
    fl.log("Flautim inicializado!!!")
    client_fn_callback   = generate_client_fn(context)
    evaluate_fn_callback = evaluate_fn(context)
    server_fn_callback   = generate_server_fn(context, eval_fn=evaluate_fn_callback)
    fl.log("Experimento federado Titanic com Differential Privacy criado!")
    run_federated(client_fn_callback, server_fn_callback, num_clients=NUM_PARTITIONS)