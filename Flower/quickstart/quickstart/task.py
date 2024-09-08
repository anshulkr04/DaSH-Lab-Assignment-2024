"""quickStart: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import DirichletPartitioner

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions , partition_by="label" , alpha=0.1)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train_with_distillation(net , trainloader , valloader , epochs, device):
    """
    Train the student model on the training set using distillation
    """
    temperature=2.0
    alpha = 0.5

    net.to(device) 

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            student_outputs = net(images.to(device))

            if(epoch % 2 == 0):
                with torch.no_grad():
                    teacher_outputs = net(images.to(device))
                    
                soft_student_output = F.log_softmax(student_outputs / temperature, dim=1)
                soft_teacher_output = F.log_softmax(teacher_outputs / temperature, dim=1)
                distill_loss = F.kl_div(soft_student_output, soft_teacher_output, reduction='batchmean')*(temperature**2)
                print(f'Distillation loss for epoch={epoch}: {distill_loss}')
            else:
                distill_loss = 0
            
            ce_loss = criterion(student_outputs, labels.to(device))

            loss = (alpha*ce_loss + (1-alpha)*distill_loss)

            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            totalBatches += 1

            avgLoss = runningLoss / totalBatches

            print(f'Epoch [{epoch+1}/{epochs}], Average Train Loss: {avgLoss:.4f}')

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)


    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results

def newTrain(net, global_net, trainloader, epochs, device, beta=0.45, temp=1):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    kl_lossfn = torch.nn.KLDivLoss(reduction="batchmean").to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    global_net.eval()
    running_loss = 0.0
    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
#            print(device)
            optimizer.zero_grad()
            if (epoch  % 2  == 1):
        
                with torch.no_grad():
                    global_logits = global_net(images)
                local_logits = net(images)
                targets = F.softmax(global_logits/temp, dim=1)
                prob = F.log_softmax(local_logits/temp, dim=1)
                # kl_loss = torch.sum(targets*(targets.log() - prob))/prob.size()[0]
                kl_loss = kl_lossfn(prob, targets)
                betaval = beta
            else:
                kl_loss = 0
                betaval = 0

            print(f"Distillation Loss for {epoch} = {kl_loss}")
            ce_loss = criterion(net(images.to(device)), labels.to(device))
            loss = (1-betaval)*ce_loss + betaval*(temp**2)*kl_loss
            print(f"CELoss for {epoch} = {ce_loss}")
            print(f"Loss for {epoch} = {loss}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

        