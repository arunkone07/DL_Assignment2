import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torchvision.models as models

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
  args.add_argument("--strategy", "-s", default=1, type=int)
  return args.parse_args()

args = cmd_parser()

# Load and transform the data
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_aug = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset=None
if(args.data_aug == 'true'):
    train_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/train', transform=transform_aug)
else:
    train_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/train', transform=transform)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [8000, 1999])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root='/content/inaturalist_12K/val', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

def resnet_1():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def resnet_2(k):
    model = models.resnet50(pretrained=True)
    params = list(model.parameters())
    for param in params[:k]:
        param.requires_grad = False #freezing
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet_3(num_dense):
    model = models.resnet50(pretrained=True)
    for params in model.parameters():
        params.requires_grad = False
    model.fc = nn.Sequential(
      nn.Linear(model.fc.in_features,num_dense),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(num_dense, num_classes)
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

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

def train(model, criterion, optimizer):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        train_loss = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
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
        train_acc = correct / (total_step * args.batch_size)

        val_acc, val_loss = accuracy(model, criterion, val_loader)

        print("Train:\nAccuracy:", train_acc, "Loss:", train_loss)
        print("Validation:\nAccuracy:", val_acc, "Loss:", val_loss, "\n")

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
                    "-org_"+str(org) + "-ks_"+ks + "-fc_"+str(args.num_dense) + "-"+args.optimizer + "-strategy_"+str(args.strategy))
model = None
if(args.strategy == 1):
    model = resnet_1()
elif(args.strategy == 2):
    model = resnet_2(10)
else:
    model = resnet_3(256)

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizers[args.optimizer](model.parameters(), lr=args.learning_rate)
train(model, criterion, optimizer)

wandb.finish()