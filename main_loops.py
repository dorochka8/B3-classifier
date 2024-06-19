import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = []
    
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_fn(output.view(-1), data.y)
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
    return total_loss, sum(total_loss) / len(total_loss)


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(output.squeeze().cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_pred_binary = (y_true == (y_pred > 0.5).astype(int))
    
    roc_auc = roc_auc_score(y_true, y_pred)
    
    accuracy = y_pred_binary.sum() / len(y_true)
    
    f1 = f1_score(y_true, y_pred_binary)

    return roc_auc, accuracy, f1


def main_loop(model, optimizer, loss_fn, train_loader, val_loader, test_loader, device, num_epochs, scheduler=None):
    train_losses = []
    epoch_losses = []
    roc_aucs = []
    print('Training ...')
    for epoch in range(num_epochs):
        train_loss, train_loss_int = train(model, train_loader, optimizer, loss_fn, device)
        train_losses.extend(train_loss)
        epoch_losses.append(train_loss_int)

        roc_auc, accuracy, f1 = evaluate(model, val_loader, device)
        roc_aucs.append(roc_auc)

        if scheduler: 
            scheduler.step(train_loss_int)

        if roc_auc >= 0.9:
            if scheduler: 
                current_lr = optimizer.param_groups[0]['lr']
                formatt_lr = f'{current_lr}'.ljust(10)
                print(f'Epoch {epoch+1}/{num_epochs} \tlr: {formatt_lr} \tTrain_loss: {train_loss_int:<22} \tval_acc: {accuracy*100:.2f}% \t***roc_auc: {roc_auc:.2f}*** \tf1_score: {f1:.4f}')

            else: print(f'Epoch {epoch+1}/{num_epochs} \tTrain_loss: {train_loss_int:<22} \tval_acc: {accuracy*100:.2f}% \t***roc_auc: {roc_auc:.2f}*** \tf1_score: {f1:.4f}')

        else:
            if scheduler: 
                current_lr = optimizer.param_groups[0]['lr']
                formatt_lr = f'{current_lr}'.ljust(10)
                print(f'Epoch {epoch+1}/{num_epochs} \tlr: {formatt_lr} \tTrain_loss: {train_loss_int:<22} \tval_acc: {accuracy*100:.2f}% \t   roc_auc: {roc_auc:.2f}   \tf1_score: {f1:.4f}')

            else: print(f'Epoch {epoch+1}/{num_epochs} \tTrain_loss: {train_loss_int:<22} \tval_acc: {accuracy*100:.2f}% \t   roc_auc: {roc_auc:.2f}   \tf1_score: {f1:.4f}')

    print(f'Testing...')
    roc_auc_test, accuracy_test, f1_test = evaluate(model, test_loader, device)

    print(f'TEST RESULTS: \troc_auc_test: {roc_auc_test} \taccuracy_test: {accuracy_test} \tf1_score_test: {f1_test}')

    print('\n')
    plt.plot(train_losses)
    plt.title('losses')
    plt.show()

    plt.plot(epoch_losses)
    plt.title('losses per epoch')
    plt.show()

    plt.plot(roc_aucs)
    plt.title('roc_aucs')
    plt.show()
    print('\n')

    return train_losses, epoch_losses, roc_aucs