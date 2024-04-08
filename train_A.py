import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
num_epochs = 10
img_size = 256

wandb.login()

def cmd_parser():
  args = argparse.ArgumentParser()
  args.add_argument("--wandb_project", "-wp", default="Assignment2")  
  args.add_argument("--wandb_entity", "-we", default="Assignment2")
  args.add_argument("--batch_norm", "-bn", default="true", choices=["true", "false"])
  args.add_argument("--batch_size","-b", type=int, default=32)
  args.add_argument("--data_aug", "-da", default="true", choices=["true", "false"])
  args.add_argument("--dropout", "-dp", default=0, type=float)
  args.add_argument("--filt_org", "-fo", default="double", choices=["equal", "double", "half"])
  args.add_argument("--kernel_size", "-ks", default=[3,3,3,3,3])
  args.add_argument("--num_dense", "-nd", default=64, type=int)
  args.add_argument("--num_filters","-nf", default=128, type=int)
  args.add_argument("--optimizer","-o", default= "adam", choices=["adam","nadam"])
  args.add_argument("--learning_rate","-lr", default=0.003, type=float)
  args.add_argument("--activation", "-a", default="mish", choices=["relu","gelu","silu","mish"])
  return args.parse_args()
    
args = cmd_parser()

# Load and transform the data
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def accuracy(model, criterion, loader):
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * labels.size(0)
    accuracy = correct / total
    loss /= total
    return accuracy, loss

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, config, num_classes=10):
        super(CNN, self).__init__()
        self.config = config
        
        if config.filt_org == 'double':
            self.filt_factor = 2
        elif config.filt_org == 'half':
            self.filt_factor = 0.5
        else:
            self.filt_factor = 1
            
        if(self.config.data_aug == 'true'):
            self.train_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/train', transform=transform_aug)
        else:
            self.train_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/train', transform=transform)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [8000, 1999])

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.config.batch_size, shuffle=True)

        self.test_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/val', transform=transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        img_size = 256
        
        in_ch = 3
        out_ch = config.num_filters
        self.conv1 = nn.Conv2d(in_ch, out_ch, config.kernel_size[0], stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        img_size = (img_size - config.kernel_size[0] + 3) // 2
        in_ch = out_ch
        out_ch = int(config.num_filters * self.filt_factor)
        self.conv2 = nn.Conv2d(in_ch, out_ch, config.kernel_size[1], stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        img_size = (img_size - config.kernel_size[1] + 3) // 2
        in_ch = out_ch
        out_ch = int(config.num_filters * self.filt_factor**2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, config.kernel_size[2], stride=1, padding=1)  
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        img_size = (img_size - config.kernel_size[2] + 3) // 2
        in_ch = out_ch
        out_ch = int(config.num_filters * self.filt_factor**3)
        self.conv4 = nn.Conv2d(in_ch, out_ch, config.kernel_size[3], stride=1, padding=1)  
        self.bn4 = nn.BatchNorm2d(out_ch)
        
        img_size = (img_size - config.kernel_size[3] + 3) // 2
        in_ch = out_ch
        out_ch = int(config.num_filters * self.filt_factor**4)
        self.conv5 = nn.Conv2d(in_ch, out_ch, config.kernel_size[4], stride=1, padding=1)   
        self.bn5 = nn.BatchNorm2d(out_ch)
        
        img_size = (img_size - config.kernel_size[4] + 3) // 2
        self.x_shape = out_ch * img_size * img_size
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      
        
        self.fc1 = nn.Linear(self.x_shape, config.num_dense)
        self.bn1d = nn.BatchNorm1d(config.num_dense)
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc2 = nn.Linear(config.num_dense, num_classes)
        
        if config.activation == 'relu':
            self.activation = F.relu
        elif config.activation == 'gelu':
            self.activation = F.gelu
        elif config.activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.mish
        
    def forward(self, x):
        
        x = self.activation(self.conv1(x)) 
        if(self.config.batch_norm == 'true'):
            x = self.bn1(x)
        x = self.pool(x)  
        
        x = self.activation(self.conv2(x)) 
        if(self.config.batch_norm == 'true'):
            x = self.bn2(x)
        x = self.pool(x)   
        
        x = self.activation(self.conv3(x))
        if(self.config.batch_norm == 'true'):
            x = self.bn3(x)
        x = self.pool(x)   
        
        x = self.activation(self.conv4(x))
        if(self.config.batch_norm == 'true'):
            x = self.bn4(x)
        x = self.pool(x)   
        
        x = self.activation(self.conv5(x))
        if(self.config.batch_norm == 'true'):
            x = self.bn5(x)
        x = self.pool(x)   
        
        x = x.view(-1, self.x_shape)
        
        x = self.activation(self.fc1(x))
        if(self.config.batch_norm == 'true'):
            x = self.bn1d(x)
            
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
    def train(self, model, criterion, optimizer):
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            train_loss = 0
            correct = 0
            for i, (images, labels) in enumerate(self.train_loader):
                # Forward pass
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if (i+1)%10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Avg Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, train_loss/(i+1)))

            train_loss /= total_step
            train_acc = correct / (total_step * self.config.batch_size)

            val_acc, val_loss = accuracy(model, criterion, self.val_loader)

            print("Train\n:Accuracy:", train_acc, "Loss:", train_loss)
            print("Validation\n:Accuracy:", val_acc, "Loss:", val_loss, "\n")
  
optimizers = {
        'adam': optim.Adam,
        'nadam': optim.NAdam
}

wandb.init()
bn = 0
aug = 0
org = 1
ks = ""

if(args.batch_norm == 'true'):
    bn = 1
    
if(args.data_aug == 'true'):
    aug = 1
    
if(args.filt_org == 'double'):
    org = 2
elif(args.filt_org == 'half'):
    org = 0.5
    
for i in range(0,5,2):
    ks += str(args.kernel_size[i])
    
wandb.run.name =  (args.activation + "-bn_"+str(bn) + "-aug_"+str(aug) + "-drop_"+str(args.dropout) + 
                    "-bs_"+str(args.batch_size) +"-lr_"+str(args.learning_rate) + "-filt_"+str(args.num_filters) +
                    "-org_"+str(org) + "-ks_"+ks + "-fc_"+str(args.num_dense) + "-"+args.optimizer)

model = CNN(args, num_classes).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optimizers[args.optimizer](model.parameters(), lr=args.learning_rate)
model.train(model, criterion, optimizer)

wandb.finish()
