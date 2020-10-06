from model import Generator
import torch
import torch.optim as optim
from dataset import FaceData
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def main():
    batch_size = 8
    generator = Generator()
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dataset = FaceData()
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    loss_content = nn.MSELoss()

    print("Start Training")
    current_epoch = 0
    generator = generator.cuda()
    for epoch in range(current_epoch, 100):
        for step, (img_Input, img_GT) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            img_GT = img_GT.cuda()
            img_Input = img_Input.cuda()
            result = generator(img_Input)
            loss_total = loss_content(result, img_GT)
            loss_total.backward()
            optimizer.step()
        save_image(denorm(result[0].cpu()), "./Result/{0}.png".format(epoch))

if __name__ == "__main__":
    main()