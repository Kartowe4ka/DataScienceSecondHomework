import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CustomCSVLoader(Dataset):
    def __init__(self, path, target_col):
        df = pd.read_csv(path)
        self.y = torch.tensor(df[target_col].values).float().unsqueeze(1)
        self.X = df.drop(columns=[target_col])

        # Кодирование и нормализация
        for col in self.X.columns:
            if self.X[col].dtype == object:
                self.X[col] = LabelEncoder().fit_transform(self.X[col])

        self.X = StandardScaler().fit_transform(self.X)
        self.X = torch.tensor(self.X).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



