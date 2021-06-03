import torch
import torch.nn as nn
from torch.optim import SGD
from models.network import Network_v1 as net
from utils.model_utils import init_model
from datasets import MaskDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary
from time import perf_counter



model = net().cuda()
init_model(model)


# Model summary
summary(model,(3,128,128))

# Test input
# test_input = torch.randn(1,3,128,128).cuda()
# output = model(test_input)


loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.004, momentum=0.9)

file_paths = r'D:\facultate\Disertatie\Datasets\mask\face_mask_crop_dataset.txt'
with open(file_paths,'r') as reader:
    train_files = [file.strip() for file in reader.readlines()]

train_dataset = MaskDetection(train_files)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
iterations = 6780


epoch = 0
max_epochs = 100
loss_values = []


torch.cuda.synchronize()

for epoch in range(max_epochs):
    start = perf_counter()
    running_loss = 0

    for iteration, data in enumerate(train_dataloader,0):
        optimizer.zero_grad()

        train_image, target = data
        # Getting one sample
        train_image = train_image.cuda()
        target = target.cuda()

        output = model(train_image)
        loss = loss_function(output,target.long())

        # Get gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        loss_value = loss.item()
        running_loss+=loss_value
        # print('epoch {}, loss {}'.format(epoch, loss.item()))
    running_loss/=len(train_dataloader)
    loss_values.append(running_loss)
    stop = perf_counter()
    print(f"Epoch {epoch}: {running_loss} Time:{stop-start}s")
plt.plot(loss_values)

