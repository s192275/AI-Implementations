import torch 
import torchvision 
from torchvision import transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import nn.functional as F
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"You are using {device}")

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = torchvision.datasets.MNIST(
    root = "./data",
    train = True,
    transform = transforms
) 

test_data = torchvision.datasets.MNIST(
    root = "./data",
    train = False,
    transform = transforms
)

rand_num = random.randint(0, len(train_data))
train_img, train_label = train_data[rand_num]

plt.imshow(train_img.permute(1,2,0), cmap = "gray")
plt.title(train_label)

train_loader = DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = 32,
    shuffle = False
)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #6x28x28
        self.conv1 = nn.Conv2d(in_channels = 1, kernel_size = (5,5), stride = 1, padding = 0, out_channels = 6)
        #6x14x14
        self.pool1 = nn.AvgPool2d(2,2)
        #16x10x10
        self.conv2 = nn.Conv2d(in_channels = 6, kernel_size = (5,5), stride = 1, padding = 0, out_channels = 16)
        #16x5x5
        self.pool2 = nn.AvgPool2d(2,2) 
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = LeNet()
optim = torch.optim.Adam(model.parameters(), lr = 1e-3) 
loss = nn.MSELoss()

def train(dataloader, model, loss_fn, op_fn):
  size = len(dataloader.dataset)
  for batch, (x,y) in enumerate(dataloader): 
      y_pred = model(x) 
      y_onehot = F.one_hot(y, num_classes=10).float()
      y_onehot = y_onehot.to(device)
      loss = loss_fn(y_pred, y) 

      op_fn.zero_grad()
      loss.backward()
      op_fn.step()

      if batch % 100 ==0:
          loss, current = loss.item(), batch * len(x) 
          print(f"loss is {loss}  [{current} / {size}]")

def test(training_dataloader, model, loss_fn):
    size = len(training_dataloader.dataset)
    num_batches = len(training_dataloader) 
    correct, test_loss = 0,0 
    with torch.no_grad():
        for x,y in training_dataloader:
            y_pred = model(x) 
            test_loss += loss_fn(y_pred, y).item() 
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item() 
    test_loss /= num_batches 
    correct /= size 
    accuracy = correct*100
    print(f"accuracy is {correct*100} test_loss is {test_loss}")
    return accuracy, test_loss

EPOCHS = 20 
acc = []
test_loss = []
for epochs in range(EPOCHS):
  print(f"epoch: {epochs+1} ---------------------------")
  train(train_loader, model, loss, optim)
  accuracy, loss = test(test_loader, model, loss) 
  acc.append(accuracy)
  test_loss.append(loss)
