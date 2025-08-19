import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import pandas as pd  # NEW

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, *_ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []

            for batch_x, batch_y, *_ in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

            train_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch {epoch+1} | Train Loss: {train_avg:.6f}, Vali Loss: {vali_loss:.6f}, Test Loss: {test_loss:.6f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

# ------------------- MODIFIED TEST -------------------
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, *_ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                output = self.model(batch_x)  # shape: [B, 2]
                preds.append(output.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print(f"Test shape: preds: {preds.shape}, trues: {trues.shape}")

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        correctness = 100 - mape

        print(f"mse: {mse:.4f}, mae: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%, Correctness: {correctness:.2f}%")

        folder_path = f'./results/{setting}/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        # === NEW: Save CSVs ===
        test_csv_path = os.path.join(folder_path, 'testset_true.csv')
        pred_csv_path = os.path.join(folder_path, 'predictions.csv')

        pd.DataFrame(trues, columns=['SBP_true', 'DBP_true']).to_csv(test_csv_path, index=False)
        pd.DataFrame(preds, columns=['SBP_pred', 'DBP_pred']).to_csv(pred_csv_path, index=False)

        print(f"Ground truth test BP saved at {test_csv_path}")
        print(f"Predictions saved at {pred_csv_path}")

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"mse: {mse:.4f}, mae: {mae:.4f}\n")
            f.write(f"MAPE: {mape:.2f}%, Correctness: {correctness:.2f}%\n\n")

        return
# -----------------------------------------------------

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data('pred')
        if load:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _, *_ in pred_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        folder_path = f'./results/{setting}/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'real_prediction.npy', preds)
        return
