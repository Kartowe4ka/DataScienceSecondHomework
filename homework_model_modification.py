import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset


# Линейная регрессия
class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

    # L1 регуляризация
    def l1_regularization(self, ratio):
        """
        Высчитывает значение L1-регуляризации
        :param ratio: коэффициент L1-регуляризации
        :return: сумма модулей весов, умноженных на коэффициент
        """
        return sum(param.abs().sum() for param in self.parameters()) * ratio

    # L2 регуляризация
    def l2_regularization(self, ratio):
        """
        Высчитывает значение L2-регуляризации
        :param ratio: коэффициент L2-регуляризации
        :return: сумма квадратов весов, умноженных на коэффициент
        """
        return sum(p.pow(2.0).sum() for p in self.parameters()) * ratio


# Логистическая регрессия
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100
    bestLoss = float("inf")
    noImprove = 0
    waitingEpoch = 10

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y) + model.l1_regularization(0.001) + model.l2_regularization(0.001)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)


        # Реализация earlyStopping
        if (bestLoss - avg_loss) > 0.001:
            best_loss = avg_loss
            no_improve = 0
        else:
            noImprove += 1
            if waitingEpoch < noImprove:
                print(epoch, bestLoss)
                break

    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch.pth')

    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval()
