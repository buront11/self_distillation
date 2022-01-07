import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

# 元論文ではResNetだったがとりあえず簡単にCNNで試す
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_block1 = nn.Sequential(nn.Conv2d(3,64,1),
                                        nn.Conv2d(64,64,1),nn.MaxPool2d(2,2))

        self.cnn_block2 = nn.Sequential(nn.Conv2d(64,128,1),
                                        nn.Conv2d(128,128,1),nn.MaxPool2d(2,2))

        self.cnn_block3 = nn.Sequential(nn.Conv2d(128,256,1),
                                        nn.Conv2d(256,256,1))

        self.cnn_block4 = nn.Sequential(nn.Conv2d(256,256,1),
                                        nn.Conv2d(256,256,1),nn.MaxPool2d(2,2))

        self.full = nn.Sequential(nn.Linear(64*8*8,120),nn.ReLU(),
                                  nn.Linear(120,84),nn.ReLU(),
                                  nn.Linear(84,10))

    def forward(self,x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.cnn_block4(x)
        x = self.full(torch.flatten(x, 1))

        return x

if __name__=='__main__':
    import torch.optim as optim

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')