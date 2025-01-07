import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, in_features ,out_features, device_id):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        nn.init.orthogonal_(self.weight)
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        nn.init.orthogonal_(self.weight1)
        
        self.weight2 = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        nn.init.orthogonal_(self.weight2)
        
        self.weight3 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.orthogonal_(self.weight3)
        
        self.weight4 = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        nn.init.orthogonal_(self.weight4)
        
        
    def forward(self, x):
        x = torch.matmul(self.weight,x)
        x = F.gelu(x)
        x[:self.in_features//2,:] += x[self.in_features//2:,:]
        x = F.normalize(x,dim=0)
        x = torch.matmul(self.weight1,x)
        x = F.gelu(x)
        x[:self.in_features//2,:] += x[self.in_features//2:,:]
        x = F.normalize(x,dim=0)
        x = torch.matmul(self.weight2,x)
        x = F.relu(x)
        x = F.normalize(x,dim=0)
        x = torch.matmul(self.weight3,x)
        x = F.gelu(x)
        x[:self.out_features//2,:] += x[self.out_features//2:,:]
        x = F.normalize(x,dim=0)
        x = torch.matmul(self.weight4,x)
        
        return x

    
def training_nn(dim,expand_dim):
    in_features=dim
    out_features=expand_dim

    model = Model(in_features,out_features,None)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    for epoch in tqdm(range(15)):

        X=torch.randn(in_features,512)
        X=F.normalize(X,dim=0)
        y=torch.matmul(torch.transpose(X,0,1),X)

        y_pred = model(X)

        y_pred = torch.matmul(torch.transpose(y_pred,0,1),y_pred)

        loss = criterion(y_pred,y)

        if epoch==1300 or epoch==1400:
            for params in optimizer.param_groups:
                params['lr'] /= 10

        loss.backward()
        
        if epoch%50==0 and epoch>0:
            print(loss)

        optimizer.step()

        optimizer.zero_grad()
        
    return model.eval()