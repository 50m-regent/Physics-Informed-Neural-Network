import csv
import numpy
import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

class PDEDataset(Dataset):
    def __init__(self, xc:Tensor, xu:Tensor, u:Tensor) -> None:
        assert len(xc) == len(xu) == len(u)
        
        self.xu = xu
        self.xc = xc
        self.u  = u
        
    def __len__(self) -> int:
        return len(self.xu) - 1
    
    def __getitem__(self, i:int) -> tuple[Tensor, Tensor]:
        return \
            torch.cat((self.xu[i + 1], self.xc[i], self.xu[i], self.u[i])), \
            self.xu[i + 1]
            
class W(nn.Module):
    def __init__(
        self,
        in_features:int, out_features:int,
        hidden_features:int=64
    ) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Sigmoid(),

            nn.Linear(hidden_features, out_features)
        )
        
    def forward(self, d:Tensor) -> Tensor:
        return self.net(d)

class Decoy:
    def __enter__(self):
        pass
    
    def __exit__(self, _, __, ___):
        pass

class Trainer:
    def __init__(
        self,
        model:nn.Module,
        train_loader:DataLoader,
        valid_loader:DataLoader,
        device:str
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device       = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def step(self, train:bool=True) -> float:
        if train:
            self.model.train()
            no_grad = Decoy
        else:
            self.model.eval()
            no_grad = torch.no_grad
        
        loss_sum = 0
        with no_grad():
            for x, y in tqdm(
                self.train_loader if train else self.valid_loader,
                desc = 'Training' if train else 'Validating'
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                
                prediction = self.model(x)
                
                loss:Tensor = self.criterion(prediction, y)
                loss_sum += loss.item()
                
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
        data_count = len(self.train_loader if train else self.valid_loader)
        return loss_sum / data_count
        
    def fit(self, num_epoch:int) -> None:
        for epoch in tqdm(range(1, num_epoch + 1), desc='Epochs'):
            train_loss = self.step()
            valid_loss = self.step(train=False)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'loss = ({train_loss:>.7f}, {valid_loss:>.7f})'
            )

def load_data(train_ratio:float=0.8) -> tuple[Tensor, Tensor]:
    with open('x.csv') as f:
        reader = csv.reader(f, delimiter='\t')
        x = torch.as_tensor(
            [[float(v) for v in row] for row in reader],
            dtype=torch.float32
        )
        
    with open('u.csv') as f:
        reader = csv.reader(f)
        u = torch.as_tensor(
            [[float(v) for v in row] for row in reader],
            dtype=torch.float32
        )
        
    assert len(x) == len(u)
        
    permutation = numpy.random.permutation(len(x))
    x, u = x[permutation], u[permutation]
    
    split_i = int(len(x) * train_ratio)
    return (x[:split_i], u[:split_i]), (x[split_i:], u[split_i:])

def main() -> None:
    BATCH_SIZE = 1
    DEVICE     = 'cpu'
    
    (x_train, u_train), (x_valid, u_valid) = load_data()
    
    train_loader = DataLoader(
        PDEDataset(torch.as_tensor([[]] * len(x_train)), x_train, u_train),
        BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        PDEDataset(torch.as_tensor([[]] * len(x_valid)), x_valid, u_valid),
        BATCH_SIZE
    )
    
    w = W(
        in_features= x_train.shape[-1] * 2 + u_train.shape[-1],
        out_features=x_train.shape[-1]
    )
    summary(w, (x_train.shape[-1] * 2 + u_train.shape[-1],))
    
    trainer = Trainer(w, train_loader, valid_loader, DEVICE)
    trainer.fit(500)

if __name__ == '__main__':
    main()
    
    
class Net(nn.Module):
    def __init__(self):
        self.p_net = nn.Sequential()
        self.x_net = nn.Sequential()
        
    def forward(self, d):
        p = self.p_net(d)
        x = self.x_net(p)
        
        return p, x