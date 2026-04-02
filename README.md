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

## : 🚢 Passo 1: Criando o Dataset

Um conjunto de dados no Flautim é acessado por um arquivo .py que deve conter uma classe que herda de Dataset.

**Exemplo: Implementando a Classe TitanicDataset**

O código abaixo implementa uma classe TitanicDataset utilizando o dataset Titanic-Dataset.csv obtido pelo [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data) para resolver um problema de classificação.

```python
from flautim.pytorch.Dataset import Dataset
import torch
import copy

class TitanicDataset(Dataset):
    def __init__(self, file, **kwargs):
        super(TitanicDataset, self).__init__(name="TITANIC", **kwargs)

        self.features = file[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                               'Title', 'FamilySize', 'IsAlone', 'Embarked', 'Deck']].values
        self.target = file['Survived'].values

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.batch_size = 32
        self.shuffle = True
        self.num_workers = 1
        self.test_size = int(0.2 * len(self.features))

    def train(self):
        train = copy.deepcopy(self)
        train.features = self.features[:-self.test_size]
        train.target = self.target[:-self.test_size]
        return copy.deepcopy(train)

    def validation(self):
        test = copy.deepcopy(self)
        test.features = self.features[-self.test_size:]
        test.target = self.target[-self.test_size:]
        return copy.deepcopy(test)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.target[idx], dtype=torch.long))
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

```python
from flautim.pytorch.Model import Model
import torch

class TitanicModel(Model):
    def __init__(self, context, num_classes=2, **kwargs):
        super(TitanicModel, self).__init__(context, name="TITANIC-NN", **kwargs)

        self.c1    = torch.nn.Linear(11, 64)
        self.bn1   = torch.nn.BatchNorm1d(64)
        self.drop1 = torch.nn.Dropout(0.3)
        self.c2    = torch.nn.Linear(64, 32)
        self.bn2   = torch.nn.BatchNorm1d(32)
        self.drop2 = torch.nn.Dropout(0.2)
        self.c3    = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.drop1(torch.relu(self.bn1(self.c1(x))))
        x = self.drop2(torch.relu(self.bn2(self.c2(x))))
        x = self.c3(x)
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

```python
from flautim.pytorch.federated.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
from collections import OrderedDict
from math import inf

def set_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class TitanicExperiment(Experiment):

    def __init__(self, model, dataset, context, **kwargs):
        super(TitanicExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs    = kwargs.get('epochs', 10)
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_loss = inf
        self.data_size = len(dataset.train().features)

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

    def training_loop(self, data_loader):
        self.model.to(self.device)
        self.model.train()

        running_loss, correct, total = 0.0, 0, 0

        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device).view(-1)

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

    def validation_loop(self, data_loader):
        self.model.to(self.device)
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in data_loader:
                X, y    = X.to(self.device), y.to(self.device).view(-1)
                outputs = self.model(X)
                running_loss += self.criterion(outputs, y).item()
                _, predicted  = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total   += y.size(0)

        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0.0

        return float(avg_loss), {'ACCURACY': accuracy}
```

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

```python
from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch.federated import Experiment
import TitanicDataset, TitanicModel, TitanicExperiment
from TitanicExperiment import get_params

import flautim as fl
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Parameters, FitIns
from flwr.server import ServerConfig, ServerAppComponents
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
    titanic = titanic.sample(frac=1, random_state=42).reset_index(drop=True)
    scaler = StandardScaler()
    numeric_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'FamilySize']
    titanic[numeric_cols] = scaler.fit_transform(titanic[numeric_cols].astype(float))
    return titanic

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
        self.round_accuracies = {}
        self.best_acc_so_far  = 0.0
        self.best_round       = None

    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        accuracy = metrics.get("ACCURACY", 0.0)
        self.round_accuracies[server_round] = accuracy

        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            self.best_round      = server_round
            print(f"  [Round {server_round}] New best global model: accuracy={accuracy:.6f}")
            ndarrays  = parameters_to_ndarrays(parameters)
            save_path = f"best_model_round_{server_round}_acc_{accuracy:.4f}.pth"
            torch.save({"round": server_round, "accuracy": accuracy, "ndarrays": ndarrays}, save_path)
            print(f"  Checkpoint saved -> {save_path}")

        self._print_ranking()
        return loss, metrics

    def _print_ranking(self):
        df = (
            pd.DataFrame(
                [(f"Round-{r}", acc) for r, acc in self.round_accuracies.items()],
                columns=["Model", "Accuracy"],
            )
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True)
        )
        print("\n--- Model (Round) x Accuracy ---")
        print(df.to_string())
        print()

    def configure_fit(self, server_round, parameters, client_manager):
        client_manager.wait_for(self.m)
        available_clients = list(client_manager.clients.values())

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
                min_available_clients=3,
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
