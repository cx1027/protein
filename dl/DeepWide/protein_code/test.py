import pandas as pd
import torch
import torch.nn as nn
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
 

class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(wide_input, 1800)
        self.relu = nn.ReLU()
        self.output = nn.Linear(1800, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(wide_input, wide_input)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(wide_input, wide_input)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(wide_input, wide_input)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(wide_input, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 20   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        print("epoch", epoch)
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            # print("epoch",epoch)
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                # print("y_pred", y_pred.size())
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        bcc = pd.DataFrame(y_pred.round() == y_val)
        y_val_df = pd.DataFrame(y_val)
        merge_bcc = pd.concat([bcc,y_val_df], axis = 1)
        y_pred_df = pd.DataFrame(y_pred.detach().numpy().round())
        merge_bcc2 = pd.concat([merge_bcc,y_pred_df], axis = 1)
        merge_bcc2.to_csv("merge_bcc2.csv")
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        print("acc",acc)
        # export result
        # print("y_pred", y_pred.round(), y_pred.shape)
        # if y_pred.round()==0:
        #     print("x\n")
        # print("y_val", y_val)
        
        
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc   
    
df_encoded = pd.read_csv("./dl/DeepWide/protein_code/protein_one_hot.csv")
X = df_encoded.iloc[:, 2:]
y = df_encoded.iloc[:, 1]
y = df_encoded['class']
wide_input = len(X.axes[1])

 
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# model1 = Wide()
# model2 = Deep()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

model = Wide()
acc = model_train(model, X_train, y_train, X_test, y_test)
print("Accuracy (deep): %.2f" % acc)

