import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image
import argparse


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

class upscaleModel(nn.Module):
    def __init__(self, upscale_factor=2):
        super(upscaleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.convPablo = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        #x = F.tanh(self.convPablo(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


def train(num_epoch=100,batch_size=5,lr=0.01, upscale_factor=2):

    #load trainind/validation data loader
    currentTransforms = transforms.Compose([transforms.ToTensor()])


    trainingData = PairedImageDataset(input_dir=r"O:/Data upscale train/Dataset/train/input/_upscaleFactor"+str(upscale_factor)+"/", target_dir=r"O:/Data upscale train/Dataset/train/target/_upscaleFactor"+str(upscale_factor)+"/", transform=currentTransforms)
    trainingLoader = DataLoader(trainingData, batch_size=1, shuffle=True)

    validationData = PairedImageDataset(input_dir=r"O:/Data upscale train/Dataset/validate/input/_upscaleFactor"+str(upscale_factor)+"/", target_dir=r"O:/Data upscale train/Dataset/validate/target/_upscaleFactor"+str(upscale_factor)+"/", transform=currentTransforms)
    validationLoader = DataLoader(validationData, batch_size=1, shuffle=False)

    # Pick GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move your model to the device
    model = upscaleModel(upscale_factor=upscale_factor).to(device)
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_loss, validation_loss = [], []

    accumulated_MSE = 0.0
    for epoch in range(num_epoch):
        run_loss_train = 0.0
        run_loss_val = 0.0

        batch_count = 0
        model.train()


        for input, target in trainingLoader:
            input = input.to(device)
            target = target.to(device)
            forwardOut = model(input)
            #handle dimension mismatch before calculating MSE
            out_h, out_w = forwardOut.shape[2], forwardOut.shape[3]
            tgt_h, tgt_w = target.shape[2], target.shape[3]

            min_h = min(out_h, tgt_h)
            min_w = min(out_w, tgt_w)

            forwardOut = forwardOut[:, :, :min_h, :min_w]
            target     = target[:, :, :min_h, :min_w]
            #now calc loss
            MSE = loss(forwardOut, target)
            accumulated_MSE += MSE
            #backpropagate

            batch_count += 1
            if batch_count % batch_size == 0:
                optimizer.zero_grad()
                accumulated_MSE.backward()
                optimizer.step()
                run_loss_train += accumulated_MSE.item() / batch_size
                accumulated_MSE = 0.0
        
        train_loss.append(run_loss_train/len(trainingLoader))
        model.eval()
        
        with torch.no_grad():
            for input, target in validationLoader:
                input = input.to(device)
                target = target.to(device)
                forwardOut = model(input)
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


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train an image upscaling model")
        parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs to train")
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
        parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer")
        parser.add_argument("--upscale_factor", type=int, default=2, help="Upscaling factor for the model")
        args = parser.parse_args()

        num_epoch = args.num_epoch
        batch_size = args.batch_size
        lr = args.lr
        upscale_factor = args.upscale_factor

        train(num_epoch=num_epoch, batch_size=batch_size, lr=lr, upscale_factor=upscale_factor)
