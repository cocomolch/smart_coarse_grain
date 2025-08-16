import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image
import argparse
from torch.utils.data import Subset
import math
import csv

class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform

        # Match files by name
        self.input_files = sorted(self.input_dir.glob("*"))
        self.target_files = sorted(self.target_dir.glob("*"))
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx])
        target_img = Image.open(self.target_files[idx])

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection

class upscaleModel(nn.Module):
    def __init__(self, upscale_factor=2):
        super(upscaleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.convPablo = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.convPablo(x))
        #x = F.tanh(self.convPablo(x))
        #x = self.dropout(x)
        #x = F.tanh(self.convPablo(x))
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        x = torch.sigmoid(x)
        #x = torch.clamp(x, 0, 1)  # Ensure output is in [0, 1] range
        return x


def train(log_csv, num_epoch,acc_size,lr, upscale_factor):

    #load trainind/validation data loader
    currentTransforms = transforms.Compose([transforms.ToTensor()])


    trainingData = PairedImageDataset(input_dir=r"O:/Data upscale train/Dataset/train/input/_upscaleFactor"+str(upscale_factor)+"/", target_dir=r"O:/Data upscale train/Dataset/train/target/_upscaleFactor"+str(upscale_factor)+"/", transform=currentTransforms)
    trainingData = Subset(trainingData, range(300))
    trainingLoader = DataLoader(trainingData, batch_size=1, shuffle=True)

    validationData = PairedImageDataset(input_dir=r"O:/Data upscale train/Dataset/validate/input/_upscaleFactor"+str(upscale_factor)+"/", target_dir=r"O:/Data upscale train/Dataset/validate/target/_upscaleFactor"+str(upscale_factor)+"/", transform=currentTransforms)
    validationLoader = DataLoader(validationData, batch_size=1, shuffle=False)

    # gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move your model to the device
    model = upscaleModel(upscale_factor=upscale_factor).to(device)
    loss = nn.L1Loss() #L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_loss, validation_loss = [], []
    
    for epoch in range(num_epoch):
        #TRAINING
        model.train()
        optimizer.zero_grad()
        accumulated_MSE = 0.0
        run_loss_train = 0.0
        run_loss_val = 0.0

        
        for i, (input, target) in enumerate(trainingLoader):
            
            #into gpu
            input, target = (input.to(device)/math.pi), (target.to(device)/math.pi)
            #forward pass
            #print(f'so many nan in train befor forward{torch.isnan(input).sum()/torch.numel(input)}')
            if not torch.isnan(input).any():
                #print(target.shape)
                
                forwardOut = model(input) #normalize to 0 1
                #print(f'so many nan in train {torch.isnan(forwardOut).sum()/torch.numel(forwardOut)}')
                if int(torch.isnan(forwardOut).sum().item()) < (forwardOut.numel()*0.05) and int(torch.isnan(target).sum().item()) < (target.numel()*0.05):
                    #set NaN to zero
                    forwardOut = torch.nan_to_num(forwardOut, nan=0.0)
                    target = torch.nan_to_num(target, nan=0.0)

                    #handle dimension mismatch before calculating MSE
                    out_h, out_w = forwardOut.shape[2], forwardOut.shape[3]
                    tgt_h, tgt_w = target.shape[2], target.shape[3]
                    min_h = min(out_h, tgt_h)
                    min_w = min(out_w, tgt_w)
                    forwardOut = forwardOut[:, :, :min_h, :min_w]
                    target     = target[:, :, :min_h, :min_w]
                    #now calc loss
                    MSE = loss(forwardOut, target)
                    MSE.backward()
                    if torch.isnan(MSE):
                        print(f"NaN detected in loss at step {i} (epoch {epoch})")
                        print(forwardOut)
                        continue
                    accumulated_MSE += MSE.item()
                    #backpropagate
                    if (i + 1) % acc_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        run_loss_train += accumulated_MSE
                        accumulated_MSE = 0.0
        train_loss.append(run_loss_train/len(trainingLoader))
        #VALIDATION
        model.eval()
        with torch.no_grad():
            for input, target in validationLoader:
                input, target = (input.to(device)/math.pi), (target.to(device)/math.pi)
                forwardOut = model(input)
                if int(torch.isnan(forwardOut).sum().item()) < (forwardOut.numel()*0.05) and int(torch.isnan(target).sum().item()) < (target.numel()*0.05):
                    #set NaN to zero
                    forwardOut = torch.nan_to_num(forwardOut, nan=0.0)
                    target = torch.nan_to_num(target, nan=0.0)
                    #handle dimension mismatch
                    out_h, out_w = forwardOut.shape[2], forwardOut.shape[3]
                    tgt_h, tgt_w = target.shape[2], target.shape[3]

                    min_h = min(out_h, tgt_h)
                    min_w = min(out_w, tgt_w)

                    forwardOut = forwardOut[:, :, :min_h, :min_w]
                    target     = target[:, :, :min_h, :min_w]
                    MSE = loss(forwardOut, target)
                    run_loss_val += MSE.item()
        validation_loss.append(run_loss_val/len(validationLoader))
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss[epoch]}, Val Loss: {validation_loss[epoch]}")

        saveIntoCSV(log_csv, train_loss[epoch], validation_loss[epoch], lr, acc_size, upscale_factor, num_epoch)


def saveIntoCSV(log_csv, train_loss, validation_loss, lr, acc_size, upscale_factor, num_epoch):
    with open(log_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow([ "train_loss", "val_loss", "lr", "acc_size", "upscale_factor", "num_epochs"])
        writer.writerow([ train_loss, validation_loss, lr, acc_size, upscale_factor, num_epoch])

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train an image upscaling model")
        parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs to train")
        parser.add_argument("--acc_size", type=int, default=5, help="Batch size for training")
        parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer")
        parser.add_argument("--upscale_factor", type=int, default=2, help="Upscaling factor for the model")
        parser.add_argument("--log_csv", type=str, default="training_log.csv", help="Path to the CSV file for logging training progress")
        
        args = parser.parse_args()

        num_epoch = args.num_epoch
        acc_size = args.acc_size
        lr = args.lr
        upscale_factor = args.upscale_factor
        log_csv = args.log_csv
        train(log_csv=log_csv, num_epoch=num_epoch, acc_size=acc_size, lr=lr, upscale_factor=upscale_factor)
