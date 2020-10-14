from model import Generator
from model import Discriminator
import torch
import torch.optim as optim
from dataset import FaceData
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.models import vgg19


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def main():
    batch_size = 32
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dataset = FaceData('train')
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()

    # content loss, perceptual loss vgg / i,j == 5,4
    vgg_net = vgg19(pretrained=True).features[:36].cuda()

    print("Start Training")
    current_epoch = 0
    for epoch in range(current_epoch, 100):
        for step, (img_Input, img_GT) in tqdm(enumerate(data_loader)):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            img_GT = img_GT.cuda()
            img_Input = img_Input.cuda()
            img_SR = generator(img_Input)

            # print()
            # print(img_SR.shape)
            # print(img_GT.shape)
            fake = discriminator(img_SR)
            real = discriminator(img_GT)

            loss_content = MSE(vgg_net(img_SR), vgg_net(img_GT))
            loss_D = BCE(fake, torch.zeros(batch_size, 1).cuda())
                     # BCE(real, torch.ones(batch_size, 1).cuda())
            loss_total = loss_content + loss_D

            if step%100 == 0:
                print('\n',"Loss_content : {:.4f}".format(loss_content.item()))
                print("Loss_D : {:.4f}".format(loss_D.item()))
                print("Loss : {:.4f}".format(loss_total.item()))

            loss_total.backward()
            # loss_content.backward(retain_graph=True)
            optimizer_G.step()

            # loss_D.backward()
            optimizer_D.step()
        save_image(denorm(img_SR[0].cpu()), "./Result/{0}.png".format(epoch))

if __name__ == "__main__":
    main()