from networks import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10('../datasets/cifar-10', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10('../datasets/cifar-10', train=False, download=True, transform=transform)

batch_size = 4
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

classes = train_set.classes
print(classes)


device = 'cuda'
net = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    cum_loss = 0
    for batch, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cum_loss += loss.item()
        if batch % 1000 == 999:
            print(f"[{batch + 1:5d}] loss: {cum_loss / 1000:0.4f}")
            cum_loss = 0

def test(dataloader, model, loss_fn):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            correct += (pred.argmax(1) == labels).sum()
            total += labels.size(0)
        print(f"correct: {correct}\ttotal: {total}")

test(test_loader, net, loss_fn)


n_epochs = 3
for epoch in range(n_epochs):
    train(train_loader, net, loss_fn, optimizer)
    test(test_loader, net, loss_fn)
