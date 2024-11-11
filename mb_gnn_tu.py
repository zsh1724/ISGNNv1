import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool
# from sh_transformb import basisLink
from sh_transformb import basisLink


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

init_wandb(name=f'GIN-{args.dataset}', batch_size=args.batch_size, lr=args.lr,
           epochs=args.epochs, hidden_channels=args.hidden_channels,
           num_layers=args.num_layers, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..','dataset')



class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


dataset = TUDataset(path, name=args.dataset,pre_transform = basisLink()).shuffle()
# dataset = TUDataset(path, name=args.dataset).shuffle()


kf = KFold(n_splits=10, shuffle=True)

accs=[]
for train_index, test_index in kf.split(dataset):
    model = Net(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
    for epoch in range(1, args.epochs + 1):
        


        train_dataset=[dataset[i] for i in train_index]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_dataset=[dataset[i] for i in test_index]
        test_loader = DataLoader(test_dataset, args.batch_size)
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
        epo=[]
        tra_a=[]
        if epoch == 0 :
            test_best = test_acc
        else:
            if test_acc > test_best:
                test_best=test_acc
                epo=epoch
                tra_a=train_acc
        listt=[epoch,loss,train_acc,test_acc]
        

