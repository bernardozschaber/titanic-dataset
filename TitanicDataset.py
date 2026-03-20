from flautim.pytorch.Dataset import Dataset
import torch
import copy

class TitanicDataset(Dataset):
    def __init__(self, file, **kwargs):
        super(TitanicDataset, self).__init__(name="TITANIC", **kwargs)

        # 11 features agora
        self.features = file[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                               'Title', 'FamilySize', 'IsAlone', 'Embarked', 'Deck']].values
        self.target = file['Survived'].values

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.batch_size = 10
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
                torch.LongTensor([self.target[idx]]))