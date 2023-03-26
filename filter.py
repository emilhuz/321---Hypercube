
from pathlib import Path
import numpy as np

import torch
from torch import nn
from torchvision.transforms import Resize


def dense_to_one_hot(y, class_count):
    return torch.from_numpy(np.eye(class_count)[y])


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (256,256)
        self.resizer = Resize(size=self.input_shape)
        self.train_imgs = []
        self.train_labels = []
        self.training_batch_size = 16

        self.layer_depths = [3, 16, 32, 64, 64, 64, 64]
        self.convs = []
        self.maxpools = []
        for c in range(1,len(self.layer_depths)):
            self.convs.append(nn.Conv2d(self.layer_depths[c-1], self.layer_depths[c], kernel_size=3, padding=1))
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(128*4*4, 64, bias=True)
        self.fc_logits = nn.Linear(64, 2, bias=True)
        
        self.lossF = nn.CrossEntropyLoss()
    

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()



    def forward(self, x):
        if x.shape != self.input_shape:
            x = self.resizer(x)
        
        h = x
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.maxpools[i](h)
            h = torch.relu(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc_logits(h)
        return logits

    def predict(self,x):
        return torch.argmax(self.forward(x))

    def train(self, x, y, l2_coeff = 0.001):
        
        n_epochs = 8
        batch_size = 16
        
        convparams = []
        for c in self.convs:
            convparams.append(*c.parameters())
        optimizer = torch.optim.SGD([
                {"params": convparams, "weight_decay": l2_coeff},
                {"params": self.fc_logits.parameters(), "weight_decay": 0.}
            ], lr=1e-2)
        
        iterator = range(n_epochs)

        for _ in iterator:
            perm = torch.randperm(len(x))
            xs = x.clone().detach()[perm]
            ys = y.clone().detach()[perm]

            x_batches = torch.split(xs, batch_size)
            y_batches = torch.split(ys, batch_size)

            for i in range(len(x_batches)):
                logits = self.forward(x_batches[i])
                loss:torch.Tensor = self.lossF(logits, y_batches[i])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    
    def reveal_classification(self, img, label:int):
        self.train_imgs.append(img)
        self.train_labels.append(label)
        if len(self.train_imgs) == self.training_batch_size:
            self.train(torch.Tensor(self.train_imgs), torch.Tensor(self.train_labels))


