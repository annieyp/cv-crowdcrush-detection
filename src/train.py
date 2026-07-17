import torch
import torch.nn as nn

#loss is always MSE, optimizer is always SGD

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            pred_count = pred.sum(dim=(1, 2, 3))
            gt_count = y.sum(dim=(1, 2, 3))
            mae = torch.abs(pred_count - gt_count).mean().item()
            mse = torch.mean((pred_count - gt_count) ** 2).item()

            loss_val, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss_val:>7f}  MAE: {mae:>7.2f}  MSE: {mse:>9.2f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    sum_abs_error = 0.0
    sum_sq_error = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred_count = pred.sum(dim=(1, 2, 3))
            gt_count = y.sum(dim=(1, 2, 3))
            sum_abs_error += torch.abs(pred_count - gt_count).sum().item()
            sum_sq_error += ((pred_count - gt_count) ** 2).sum().item()

    test_loss /= num_batches
    mae = sum_abs_error / size
    rmse = (sum_sq_error / size) ** 0.5
    print(f"Test Error: \n MAE: {mae:>0.2f}, RMSE: {rmse:>0.2f}, Avg loss: {test_loss:>8f} \n")