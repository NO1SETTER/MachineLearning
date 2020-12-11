import torch
import torch.nn as nn
from torchvision import datasets,transforms
import matplotlib.pylab as pyl

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.Subsamp1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.Subsamp2= nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Subsamp1(x)
        x = self.Conv2(x)
        x = self.Subsamp2(x)
        x = x.view(x.size(0), -1)
        x = self.Fc(x)
        return x



data_train = datasets.MNIST(root="./data/",
                            transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]),
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]),
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset =data_train,
                                                batch_size = 60,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset =data_test,
                                               batch_size = 60,
                                               shuffle = True)



#epochs = 8
epochs=7
batch_size=60
#file = open('params2.txt','w+')
#for learning_rate in range (1,6):
x=[]
y=[]
for i in range(int(epochs*len(data_train)/batch_size)):
    x.append(i+1)
for learning_rate in range (2,3):
    cnn = CNN()
    learning_rate=0.001*learning_rate
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    total_loss=0
    for epoch in range(epochs):#这里可以每一轮都当作一种参数尝试
        train_correct=0
        test_correct=0
        for i,(images,labels) in enumerate(data_loader_train):
            output=cnn.forward(images)
            _, pred = torch.max(output.data, 1)
            loss=cost(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            y.append(loss.item())
            train_correct+=torch.sum(pred == labels.data)
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d] Learning Rate:%.4f Batch [%d/%d] Loss: %.4f'
                    % (epoch + 1, epochs, learning_rate, i + 1, len(data_train) // batch_size, loss.item()))
        for (images,labels) in data_loader_test:
            output=cnn.forward(images)
            _,pred=torch.max(output.data,1)
            test_correct+=torch.sum(pred==labels.data)
        train_rate=float(train_correct)/float(len(data_train))
        test_rate=float(test_correct)/float(len(data_test))
        avg_loss=total_loss/float(len(data_train)*(epoch+1))
        str=("Epochs:%d Learning Rate:%.4f Average Loss:%.8f\n"
             " Train_accuracy:%.8f Test accuracy:%.8f\n"
             "---------------------------\n")%(epoch+1,learning_rate,avg_loss,train_rate,test_rate)
        print(str)
    pyl.plot(x, y)
    pyl.show()
        #file.write(str)