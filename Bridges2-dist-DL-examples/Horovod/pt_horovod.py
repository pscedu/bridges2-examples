import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import argparse
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser(description='Input values.')
parser.add_argument('-nw', type=int, default=10, help='number of workers in dataloader')
parser.add_argument('-bz', type=int, default=64, help='Batch size')
parser.add_argument('-image_size', type=int, default=128, help='Image size')
parser.add_argument('-epoch_num', type=int, default=10, help='Number of training epochs')
parser.add_argument('-mp', action='store_true', help='Mixed Precision Training')
parser.add_argument('-imagenet', action='store_true', help='Mixed Precision Training')

args = parser.parse_args()
batch_size = args.bz
n_workers = args.nw
image_size = args.image_size
mixed_precision = args.mp
num_epochs = args.epoch_num
if hvd.local_rank() == 0:
    print('n_workers=',n_workers)
    print('batch size=',batch_size)
    print('image size=',image_size)
    print('Mixed precision training: ',mixed_precision)
    print('number of GPUs:',torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Set hyperparameters
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
])

# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='/direcotry/to/imagenet-mini/train',
    transform=transform
)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers,pin_memory=True,sampler=train_sampler)

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=None)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
if mixed_precision:
    scaler = torch.cuda.amp.GradScaler()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# Train the model...
for epoch in range(epoch_num):
    for i, data in enumerate(tqdm(train_loader)):
        # Move input and label tensors to the device
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)            


        # Backward pass
        if mixed_precision:
            scaler.scale(loss).backward()
            # Make sure all async allreduces are done
            optimizer.synchronize()
            # In-place unscaling of all gradients before weights update
            scaler.unscale_(optimizer)
            with optimizer.skip_synchronize():            
                scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print(f'Finished Training, Loss: {loss.item():.4f}')
