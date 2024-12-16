import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from utils.output import *
from utils.compute_corr import *
from modal.FIE_Net import FIENet
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature")
torch.autograd.set_detect_anomaly(True)

# DATA
eeg_data_array = np.load('data/alldata/eeg_data.npy')
eog_data_array = np.load('data/alldata/eog_data.npy')
label_array = np.load('data/alldata/labels.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataloader(eeg_data, eog_data, labels, batch_size=32):
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    eog_tensor = torch.tensor(eog_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(eeg_tensor, eog_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Train
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss, total_corr = 0.0, 0.0
        for eeg_inputs, eog_inputs, labels in train_loader:
            eeg_inputs, eog_inputs, labels = eeg_inputs.to(device), eog_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(eeg=eeg_inputs, eog=eog_inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            corr = compute_corr(labels, outputs.squeeze()).item()
            running_loss += loss.item()
            total_corr += corr
        avg_loss = running_loss / len(train_loader)
        avg_corr = total_corr / len(train_loader)
        train_rmse = torch.sqrt(torch.tensor(avg_loss))

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, RMSE: {train_rmse:.4f}, CORR: {avg_corr:.4f}")

# Test
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss, total_corr = 0.0, 0.0
    with torch.no_grad():
        for eeg_inputs, eog_inputs, labels in test_loader:
            eeg_inputs, eog_inputs, labels = eeg_inputs.to(device), eog_inputs.to(device), labels.to(device)
            outputs = model(eeg=eeg_inputs, eog=eog_inputs)
            loss = criterion(outputs.squeeze(), labels)
            corr = compute_corr(labels, outputs.squeeze()).item()

            total_loss += loss.item()
            total_corr += corr

    avg_loss = total_loss / len(test_loader)
    avg_corr = total_corr / len(test_loader)
    test_rmse = torch.sqrt(torch.tensor(avg_loss))

    return avg_loss, test_rmse, avg_corr

# 5 Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_model = None
best_val_loss = float('inf')

for fold, (train_idx, test_idx) in enumerate(kfold.split(eeg_data_array)):
    print(f"Fold {fold + 1}/5")

    train_eeg, test_eeg = eeg_data_array[train_idx], eeg_data_array[test_idx]
    train_eog, test_eog = eog_data_array[train_idx], eog_data_array[test_idx]
    train_labels, test_labels = label_array[train_idx], label_array[test_idx]

    train_loader = create_dataloader(train_eeg, train_eog, train_labels)
    test_loader = create_dataloader(test_eeg, test_eog, test_labels)

    model = FIENet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer)

    val_loss, val_rmse, val_corr = test_model(model, test_loader, criterion)
    print(f"Fold {fold + 1} Test MSE: {val_loss:.4f}, Test RMSE: {val_rmse:.4f}, Test CORR: {val_corr:.4f}")

    results.append({"fold": fold + 1, "loss": val_loss, "rmse": val_rmse, "corr": val_corr})


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()


results_df = pd.DataFrame(results)
results_df.to_csv("outputs/mix5fold_results.csv", index=False)

print(results_df)


if best_model:
    torch.save(best_model, "outputs/best_model.pth")
    print("Best model saved to 'outputs/best_model.pth'")
