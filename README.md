<table style="margin: auto; background-color: white;">
  <tr>
    <td style="background-color: white;"><img src='README/flautim.png' alt="Flautim" width="200" /></td>
    <td style="background-color: white;"><img src='README/futurelab.png' alt="FutureLab" width="200" /></td>
    <td style="background-color: white;"><img src='README/flautim.png' alt="Flautim" width="200" /></td>
    <td style="background-color: white;"><img src='README/futurelab.png' alt="FutureLab" width="200" /></td>
    <td style="background-color: white;"><img src='README/flautim.png' alt="Flautim" width="200" /></td>
    <td style="background-color: white;"><img src='README/futurelab.png' alt="FutureLab" width="200" /></td>
    <td style="background-color: white;"><img src='README/flautim.png' alt="Flautim" width="200" /></td>
  </tr>
</table>

# ⚓ TUTORIAL 5 - DATASET TITANIC, SELEÇÃO DE CLIENTES E PRIVACIDADE DIFERENCIAL

Bem-vindo! Neste tutorial você aprenderá sobre a interface de programação da plataforma **Flautim** e também como montar um experimento simples de classificação usando o dataset [Titanic](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) com **seleção de clientes** e **privacidade diferencial**.

É recomendado que você já esteja familiarizado com aprendizado federado e utilização da plataforma Flautim, tendo realizado algum dos outros tutoriais previamente.

O código desse tutorial pode ser acessado em: [clique aqui](./TUTORIAL_4/).

---

## 📋 Tabela de Conteúdos

- [Passo 1 — Criando o Dataset](#-passo-1-criando-o-dataset)
- [Passo 2 — Criando o Modelo](#-passo-2-criando-o-modelo)
- [Passo 3 — Criando o Experimento](#-passo-3-criando-o-experimento)
  - [Passo 3.1 — Experimento Federado](#-passo-31-experimento-federado)
- [Passo 4 — Criando a estratégia](#-passo-4-criando-a-estratégia)
- [Referências](#referências)

---

Vamos começar entendendo a interface de programação da **Flautim**. A **Flautim_api** é uma biblioteca modularizada que facilita a realização de experimentos de aprendizado de máquina, seja convencional/centralizado ou federado.

Todo projeto **Flautim** precisa herdar essa biblioteca, que contém submódulos específicos para diferentes tecnologias (por exemplo, submódulos para PyTorch, TensorFlow, etc). Neste tutorial usaremos o submódulo para PyTorch.

Dentro de cada submódulo existem três componentes principais (classes):

**📊 1. Dataset:** é utilizado para representar os dados do experimento. Esta classe pode ser reutilizada em diversos experimentos e com diferentes modelos, sendo o componente mais versátil e reutilizável. Os usuários podem importar os dados de diversas fontes, como arquivos locais ou bases de dados online, desde que a classe Dataset seja herdada.

**⚙️ 2. Model:** representa qualquer conjunto de parâmetros treináveis dentro do projeto. Ela permite a aplicação de técnicas de aprendizado de máquina por meio de treinamento desses parâmetros. No caso de PyTorch, a classe herda a nn.Module, que define a estrutura e os parâmetros treináveis do modelo.

**🧪 3. Experiment:** define o ciclo de treinamento e validação. Existem dois tipos principais de experimentos: o experimento centralizado, que segue o fluxo convencional de aprendizado de máquina, e o experimento federado, adaptado para aprendizado federado. Esta classe inclui duas funções principais, um loop de treinamento e um loop de validação, que realizam a atualização dos parâmetros e cálculo das métricas de custo, respectivamente.

Além desses três componentes principais, há também um módulo chamado Common. Este módulo fornece acesso a classes essenciais para o gerenciamento de dados e monitoramento do treinamento.

Com essa visão geral, você está pronto para começar montar seus próprios experimentos. Vamos ao passo a passo!

---

## 🚢 Passo 1: Criando o Dataset

Um conjunto de dados no Flautim é acessado por um arquivo .py que deve conter uma classe que herda de Dataset.

**Exemplo: Implementando a Classe TitanicDataset**

O código abaixo implementa uma classe TitanicDataset utilizando o dataset Titanic-Dataset.csv obtido pelo [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data) para resolver um problema de classificação.

```python
from flautim.pytorch.Dataset import Dataset
import torch
import copy

from flautim.pytorch.Dataset import Dataset
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.preprocessing import StandardScaler


class TitanicTorchDataset(TorchDataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "label": self.y[idx]
        }


def dirichlet_partition(X, y, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)

    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(cls_indices)).astype(int)

        diff = len(cls_indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(cls_indices[start:end])
            start = end

    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


class TitanicDataset(Dataset):
    def __init__(self, csv_path, client_id, num_clients=4, test_size=0.2, seed=42, **kwargs):
        name = kwargs.get("name", "Titanic")
        super(TitanicDataset, self).__init__(name, **kwargs)

        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)

        df = pd.read_csv(csv_path)

        df = df[
            [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Survived",
            ]
        ].copy()

        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=False)

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)

        X = df.drop(columns=["Survived"]).astype("float32").values
        y = df["Survived"].astype("int64").values

        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype("float32")

        if (
            not hasattr(TitanicDataset, "_global_partition")
            or not hasattr(TitanicDataset, "_partition_config")
            or TitanicDataset._partition_config != (num_clients, seed)
        ):
            TitanicDataset._global_partition = dirichlet_partition(
                X, y, num_clients=num_clients, alpha=0.5, seed=seed
            )
            TitanicDataset._partition_config = (num_clients, seed)

        indices = TitanicDataset._global_partition[client_id]

        X_client = X[indices]
        y_client = y[indices]

        split_idx = int((1 - test_size) * len(X_client))
        self.X_train = X_client[:split_idx]
        self.y_train = y_client[:split_idx]
        self.X_test = X_client[split_idx:]
        self.y_test = y_client[split_idx:]

        self.train_partition = TitanicTorchDataset(self.X_train, self.y_train)
        self.test_partition = TitanicTorchDataset(self.X_test, self.y_test)

        self.input_dim = self.X_train.shape[1]

    def train(self):
        return self.train_partition

    def validation(self):
        return self.test_partition

    def dataloader(self, validation=False):
        dataset = self.validation() if validation else self.train()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(False if validation else self.shuffle),
            num_workers=0,
        )
```

---

## 🧠 Passo 2: Criando o Modelo

Agora, vamos criar a classe que implementa o modelo. Essa classe deve herdar da classe Model.

**Exemplo: Implementando a Classe TitanicModel**

A classe TitanicModel implementa uma rede neural totalmente conectada (MLP) para classificação binária de sobrevivência no dataset Titanic, com as seguintes camadas:
* Uma camada oculta totalmente conectada, com entrada de `input_dim` neurônios (dimensão determinada dinamicamente pelo pré-processamento do dataset — por padrão 9 features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S) e saída de 32 neurônios. Ativação ReLU.
* Uma camada oculta totalmente conectada, com entrada de 32 neurônios e saída de 16 neurônios. Ativação ReLU.
* Uma camada de saída totalmente conectada com 2 neurônios (0: não sobreviveu / 1: sobreviveu). Sem ativação — a função CrossEntropyLoss aplica softmax internamente durante o treinamento.

Essa classe deve ser incluída em um arquivo TitanicModel.py.

```from flautim.pytorch.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanicModel(Model):
    """
    MLP simples para classificação binária 

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
```

---

## 🔬 Passo 3: Criando o Experimento

Por fim, será criado o experimento, isto é, uma classe que implementa os loops de treinamento e validação do modelo TitanicModel no dataset TitanicDataset. Para isso, precisamos criar dois arquivos .py, o run.py (que deve ter obrigatoriamente esse nome) e o TitanicExperiment.py responsável por implementar o experimento, descritos a seguir:

**1. Arquivo run.py:**

* Esse arquivo é o ponto de entrada de todo experimento Flautim, pois é ele que deve iniciar a classe do experimento, um modelo e um Dataset.

**2. Arquivo TitanicExperiment.py:**

* Esse arquivo deve conter uma classe que implemente os métodos de treinamento (`training_loop`) e validação (`validation_loop`) do modelo. Essa classe deve herdar da classe `Experiment`.

Esse tutorial cobre o experimento federado com monitoramento de métricas por rodada e salvamento automático do melhor modelo.

#### **Monitoramento do Melhor Modelo por Rodada**

Em experimentos federados, é útil acompanhar a evolução do modelo global ao longo das rodadas de treinamento para identificar quais épocas produziram os melhores resultados. A estratégia `CustomFedAvg`, definida no `run.py`, estende a classe `FedAvg` do Flower para adicionar essa funcionalidade.

A cada rodada, o servidor avalia o modelo global centralizadamente e registra as métricas de desempenho. Quando um novo melhor modelo é encontrado, o checkpoint é salvo automaticamente em disco.

##### Estratégia CustomFedAvg

A `CustomFedAvg` é responsável por:
1. Registrar `ACCURACY` e `F1_SCORE` do modelo global a cada rodada em `round_results`
2. Salvar um checkpoint `.pth` em `outputs/<data>/<hora>/` sempre que uma nova melhor acurácia for encontrada (`_update_best_acc`)
3. Exibir após cada rodada uma tabela classificada por acurácia (decrescente), correlacionando cada rodada (época) com o desempenho do modelo global

##### Configuração do experimento

O experimento é configurado no `run.py` com os seguintes parâmetros principais:
- `NUM_CLIENTS = 10` — número de clientes federados
- `NUM_ROUNDS = 20` — número total de rodadas federadas
- `LOCAL_EPOCHS = 2` — épocas de treinamento local por rodada
- `fraction_fit = 1.0` — todos os clientes participam do treinamento em cada rodada
- `fraction_evaluate = 1.0` — todos os clientes participam da avaliação

---

### 🔒 Passo 3.1: Experimento Federado

**Implementando a Classe TitanicExperiment**

No código abaixo, criamos a classe TitanicExperiment no modo federado com seus métodos `training_loop` e `validation_loop` para treinar e testar a rede neural. Esses métodos retornam o valor da função de perda e as métricas de treinamento e de validação.

```from flautim.pytorch.federated.Experiment import Experiment
import torch
import numpy as np

from collections import OrderedDict
from sklearn.metrics import f1_score


def set_params(model, parameters):
    # Substitui os parâmetros do modelo pelos recebidos do servidor.
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    # Extrai os parâmetros do modelo como lista de NumPy arrays.
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


class TitanicExperiment(Experiment):

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

        # Método chamado pelo servidor em cada rodada federada.
        set_params(self.model, parameters)

        epochs = config.get("epochs", self.epochs)

        self.model.to(self.device)

        final_loss = 0.0
        final_metrics = {}

        for _ in range(epochs):
            final_loss, final_metrics = self.training_loop(self.dataset.dataloader())

        return get_params(self.model), self.data_size, final_metrics

    def training_loop(self, data_loader):

        # Treinamento local no cliente.

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

        # Avaliação local no conjunto de validação do cliente.
        
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
```

## 🎯 Passo 4: Criando a estratégia

**Implementando o run.py para realização de um experimento federado**

**1. Upload do Conjunto de Dados:**

* *Arquivo Local:* Se o seu conjunto de dados for um arquivo (por exemplo, CSV, NPZ, etc.), faça o upload para a plataforma e carregue-o usando o caminho `./data/nomedoarquivo`.
* *URL:* Se o conjunto de dados estiver disponível em uma URL, inclua a URL no seu código e carregue-o diretamente.

**2. Separação dos dados por cliente:**

* Para simular 10 clientes, os dados são divididos via particionamento Dirichlet.

**3. Crie uma instância para TitanicDataset, TitanicModel, TitanicExperiment.**

**4. Execute as funções:**
* ***generate_server_fn:*** Cria a estratégia para o aprendizado federado
* ***generate_client_fn:*** Gera o modelo e o dataset de cada cliente.
* ***evaluate_fn:*** Avalia o modelo global agregando as métricas de todos os clientes.
* ***run_federated:*** Executa o experimento federado.

#### **Implementação da CustomFedAvg**

A estratégia `CustomFedAvg` herda de `FedAvg` e é declarada no `run.py`. Ela sobrescreve o método `evaluate` para registrar e exibir o desempenho do modelo global a cada rodada.

Ao definirmos o servidor utilizamos `strategy = CustomFedAvg(...)` e todo o monitoramento de desempenho é gerenciado por ela. O funcionamento base é:
1. Chama a avaliação centralizada do `FedAvg` para obter `loss` e `metrics` da rodada atual
2. Extrai `ACCURACY` e `F1_SCORE` das métricas e registra em `round_results`:
   - **ACCURACY**: proporção de amostras classificadas corretamente pelo modelo — `corretas / total`. Varia de 0 a 1, onde 1 significa que todas as predições estão corretas.
   - **F1_SCORE**: média harmônica entre precisão e recall, calculada com `average="macro"` (sklearn) — ou seja, a média é feita igualmente entre as duas classes (sobreviveu / não sobreviveu), independente do número de amostras de cada uma. É uma métrica mais robusta que a acurácia em datasets desbalanceados, pois penaliza modelos que ignoram a classe minoritária.
3. Verifica se a acurácia atual supera o melhor valor encontrado até o momento (`best_acc_so_far`)
4. Se sim, constrói o `state_dict` do modelo a partir dos parâmetros recebidos e salva o checkpoint como `model_acc_<valor>_round_<N>.pth` em `outputs/<data>/<hora>/`
5. Exibe uma tabela com todas as rodadas ordenadas por acurácia decrescente, permitindo identificar quais épocas produziram os melhores modelos globais

```from flautim.pytorch.common import run_federated, weighted_average
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


# Configurações do experimento

NUM_CLIENTS = 4
CSV_PATH = "./data/titanic.csv"
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
NUM_ROUNDS = 20


class CustomFedAvg(FedAvg):

    def __init__(self, context, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_acc_so_far = 0.0
        self.round_results = []

        run_dir = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.save_path = Path.cwd() / f"outputs/{run_dir}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self._context = context
        self._input_dim = input_dim

    def _update_best_acc(self, server_round, accuracy, parameters):

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
        loss, metrics = super().evaluate(server_round, parameters)

        accuracy = metrics.get("ACCURACY", 0.0)
        f1 = metrics.get("F1_SCORE", 0.0)

        self.round_results.append({"round": server_round, "accuracy": accuracy, "F1_SCORE": f1})
        self._update_best_acc(server_round, accuracy, parameters)

        sorted_results = sorted(self.round_results, key=lambda x: x["accuracy"], reverse=True)
        header = f"{'Round':>6}    {'Accuracy':>10}    {'F1':>10}"
        fl.log(header)
        fl.log("-" * len(header))
        for r in sorted_results:
            fl.log(f"{r['round']:>6}    {r['accuracy']:>10.6f}    {r['F1_SCORE']:>10.6f}")

        return loss, metrics


def fit_config(server_round: int):
    
    # Configuração enviada para os clientes em cada rodada.

    return {
        "server_round": server_round,
        "epochs": LOCAL_EPOCHS,
    }


def generate_client_fn(context):
    
    # Cria a função que instancia cada cliente federado.
    

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
  
    # Avalia o modelo global em todas as partições de validação dos clientes
    # e agrega as métricas de forma ponderada pelo número de amostras.
  

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
   
    # Cria o servidor federado com estratégia FedAvg.

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
```

---

## 📚 Referências

- [1] FUTURELAB-DCC. flautim_tutoriais. Disponível em: https://github.com/FutureLab-DCC/flautim_tutoriais/tree/main.

- [2] FLOWER LABS GMBH. Vertical Federated Learning with Flower. Disponível em: https://flower.ai/docs/examples/vertical-fl.html.

- [3] FLOWER LABS GMBH. Use Differential Privacy. Disponível em: https://flower.ai/docs/framework/how-to-use-differential-privacy.html.

- [4] FLOWER COMMUNITY. How do I write a custom client selection protocol? Fórum Flower. Disponível em: https://discuss.flower.ai/t/how-do-i-write-a-custom-client-selection-protocol/74.

- [5] FLOWER COMMUNITY. Custom client selection strategy. Fórum Flower. Disponível em: https://discuss.flower.ai/t/custom-client-selection-strategy/63.

- [6] BARROS, P. SaveBestModelPerEpoch.ipynb. Exemplo de estratégia para salvar o melhor modelo por época em experimentos federados com Flower. Disponível em: https://colab.research.google.com/drive/1ZsVqPDwYg3xewaWsVt3twmtGQNypuYJx.

- [7] BARROS, P. Flower+DP.ipynb. Exemplo de experimento federado com Privacidade Diferencial utilizando o framework Flower. Disponível em: https://colab.research.google.com/drive/1NsDr39q0VbmKZcwKwrFo9G1vyMPtoW6u.
