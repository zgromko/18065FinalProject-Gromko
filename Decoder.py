import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Datahandler import gameDataset, equalityDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from GameGen import Game
import numpy as np


class Decoder(nn.Module):
    def __init__(self, board_shape, time):
        super().__init__()
        self.time = time
        self.board_shape = board_shape
        self.board_size = board_shape[0]*board_shape[1]
        self.neighbor_conv = nn.Conv2d(1, 32, kernel_size=(7,7), padding=3)
        self.board_layer1 = nn.Linear(self.board_size, 4*self.board_size)
        self.board_layer2 = nn.Linear(36*self.board_size, 9*self.board_size)
        self.board_layer3 = nn.Linear(9*self.board_size, 9*self.board_size)
        self.output_layer = nn.Linear(9*self.board_size, self.board_size)

    def forward(self, x):
        y = x
        for t in range(self.time):
            y1 = F.leaky_relu(self.board_layer1(y))
            y2 = y.reshape((x.size(0), 1, ) + self.board_shape)
            y2 = F.leaky_relu(self.neighbor_conv(y2))
            y2 = y2.reshape((x.size(0),32*self.board_size))
            y = torch.cat((y1, y2), 1)
            y = F.leaky_relu(self.board_layer2(y))
            y = F.leaky_relu(self.board_layer3(y))
            y = torch.sigmoid(self.output_layer(y))
        return y

    def change_time(self, t):
        self.time = t

class Discriminator(nn.Module):
    def __init__(self, board_shape, time, load_forward_net):
        super().__init__()
        self.time = time
        self.board_shape = board_shape
        self.board_size = board_shape[0]*board_shape[1]
        self.gameOutput = False
        self.equality = False
        self.neighbor_conv = nn.Conv2d(1, 8, kernel_size=(3,3), padding=1)
        self.board_layer = nn.Linear(8*self.board_size, 2*self.board_size)
        self.output_layer = nn.Linear(2*self.board_size, self.board_size)
        self.discriminator_layer1 = nn.Linear(2*self.board_size, 3*self.board_size)
        self.discriminator_DR = nn.Conv2d(1,1,kernel_size=(2,1), padding=0)
        self.discriminator_layer2 = nn.Linear(4*self.board_size, self.board_size)
        self.discriminator_layer3 = nn.Linear(self.board_size, 1)
        if load_forward_net:
            self.load_state_dict(torch.load('NetdGoodForward.pkl'))
            self.discriminator_layer1 = nn.Linear(2*self.board_size, 3*self.board_size)
            self.discriminator_DR = nn.Conv2d(1,1,kernel_size=(2,1), padding=0)
            self.discriminator_layer2 = nn.Linear(4*self.board_size, self.board_size)
            self.discriminator_layer3 = nn.Linear(self.board_size, 1)
                

    def forward(self, x):
        if self.gameOutput and not self.equality:
            y = x
            for t in range(self.time):
                y = y.reshape((x.size(0), 1, ) + self.board_shape)
                y = F.leaky_relu(self.neighbor_conv(y))
                y = y.reshape((x.size(0),8*self.board_size))
                y = F.leaky_relu(self.board_layer(y))
                y = torch.sigmoid(self.output_layer(y))
            return y
        elif not self.equality:
            y = x[:,:self.board_size]
            for t in range(self.time):
                y = y.reshape((x.size(0), 1, ) + self.board_shape)
                y = F.leaky_relu(self.neighbor_conv(y))
                y = y.reshape((x.size(0),8*self.board_size))
                y = F.leaky_relu(self.board_layer(y))
                y = torch.sigmoid(self.output_layer(y))
            z1 = F.leaky_relu(self.discriminator_layer1(torch.cat((y, x[:,self.board_size:]), 1)))
            z2 = F.leaky_relu(self.discriminator_DR(torch.cat((y, x[:,self.board_size:]), 1).reshape((x.size(0), 1, 2, self.board_size))))
            z = torch.cat((z1, z2.reshape(x.size(0), self.board_size)), 1)                                                                                                     
            z = F.leaky_relu(self.discriminator_layer2(z))
            return torch.sigmoid(self.discriminator_layer3(z))
        else:
            '''
            This is a relic of experimentation with separate training of discrimination,
            disconnected from the forward model
            '''
            z1 = F.leaky_relu(self.discriminator_layer1(x))
            z2 = F.leaky_relu(self.discriminator_DR(x.reshape((x.size(0), 1, 2, self.board_size))))
            z = torch.cat((z1, z2.reshape(x.size(0), self.board_size)), 1) 
            z = F.leaky_relu(self.discriminator_layer2(z))
            return torch.sigmoid(self.discriminator_layer3(z))

    def change_time(self, t):
        self.time = t

    def prep_forward(self):
        self.gameOutput = True
        self.equality = False
        for name, param in self.named_parameters():
            if "discriminator" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def prep_discriminator(self):
        self.gameOutput = False
        self.equality = False
        for name, param in self.named_parameters():
            if "discriminator" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def prep_equality(self):
        self.gameOutput = False
        self.equality = True
        for name, param in self.named_parameters():
            if "discriminator" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
def generate_rand_features(board_size, size):
    res = []
    for i in range(size):
        g = Game(board_size)
        
    return torch.Tensor(res)
                

criterion = nn.BCELoss()
DX = []
DGZ = []
backward_dynamics_frob_norm = []
batch_size = 32

def gen_game_data(size, board_size, time_diff):
    print("Generating dataset")
    start_time = time.time()
    dataset = gameDataset(board_size, time_diff, size)
    print("--- %s seconds ---" % (time.time() - start_time))
    return dataset

def gen_equality_data(size, board_size):
    print("Generating dataset")
    start_time = time.time()
    dataset = equalityDataset(board_size, size)
    print("--- %s seconds ---" % (time.time() - start_time))
    return dataset

def train_gan(train_dataset, test_dataset, epochs):
    dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
    netD.prep_discriminator()
    epoch_dx = []
    epoch_dgz = []
    max_norms = [i + 0.1 for i in range(6)]
    for epoch in range(epochs):
        if epoch % 2 == 0:
            backward_dynamics_frob_norm.append(assess_generator(test_dataset, max_norms))
        if epoch > 0:
            DX.append(sum(epoch_dx)/len(epoch_dx))
            DGZ.append(sum(epoch_dgz)/len(epoch_dgz))
            epoch_dx = []
            epoch_dgz = []
            
        for i, data in enumerate(dataloader):

            #Train discriminator on real data
            netD.zero_grad()
            gameInitial = data['initial']
            gameFinal = data['final']
            label = torch.full((gameInitial.size(0),), 1, dtype=torch.float)

            output = netD(torch.cat((gameInitial, gameFinal),1)).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            #Train discriminator on fake data
            fakeInitial = netG(gameFinal)
            label.fill_(float(0))
            output = netD(torch.cat((fakeInitial.detach(),gameFinal),1)).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            #Train generator
            netG.zero_grad()
            label.fill_(float(1))
            output = netD(torch.cat((fakeInitial,gameFinal),1)).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            epoch_dx.append(D_x)
            epoch_dgz.append(D_G_z1)
            
            if i % 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

train_forward_loss = []
forward_dynamics_frob_norm = []

def train_forward(train_dataset, test_dataset, epochs):
    dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
    netD.prep_forward()
    epoch_losses = []
    for epoch in range(epochs):
        if epoch % 2 == 0:
            forward_dynamics_frob_norm.append(assess_forward_dynamics(test_dataset, [0.1, 1.1, 2.1]))
        if epoch > 0:
            train_forward_loss.append(sum(epoch_losses)/len(epoch_losses))
            epoch_losses = []
        for i, data in enumerate(dataloader):   
            netD.zero_grad()
            gameInitials = data["initial"]
            gameFinals = data["final"]
            output = netD(gameInitials)
            
            errD = criterion(output, gameFinals)
            errD.backward()
            optimizerD.step()
            epoch_losses.append(errD.item())
            if i% 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, i, len(dataloader), errD.item()))

def train_equality(data, epochs):
    dataloader = DataLoader(data, batch_size = 32, shuffle=True)
    netD.prep_equality()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            game1 = data['game1']
            game2 = data['game2']
            label = data['label'].view(-1)
            output = netD(torch.cat((game1, game2), 1)).view(-1)
            errE = criterion(output, label)
            errE.backward()
            optimizerD.step()
            if i% 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, i, len(dataloader), errE.item()))

def view_gen_comparison(dataset, idx):
    fakeInitial = netG(torch.stack([dataset[idx]['final']]))
    fakeInitial = (fakeInitial > 0.5).int().reshape(board_size[0],board_size[1]).numpy()
    print("Generated initial")
    print(fakeInitial)
    print("Target final")
    print(dataset[idx]['final'].reshape(board_size[0],board_size[1]).int().numpy())
    print("Final from generated")
    g = Game(fakeInitial)
    g.next_state()
    print(g.board[-1])
    

def assess_generator(dataset, max_norms):
    correct = [0 for i in max_norms]
    for data in dataset:
        fakeInitial = netG(torch.stack([data['final']]))
        fakeInitial = (fakeInitial > 0.5).int().reshape(board_size[0],board_size[1]).numpy()
        g = Game(fakeInitial)
        g.next_state()
        frob_norm = np.linalg.norm(g.board[-1] - data['final'].reshape(board_size[0],board_size[1]).int().numpy())
        for i in range(len(max_norms)):
            if  frob_norm < max_norms[i]:
                correct[i] += 1
    return [norm_correct/len(dataset) for norm_correct in correct]

def assess_forward_dynamics(dataset, max_norms):
    correct = [0 for i in max_norms]
    netD.prep_forward()
    for data in dataset:
        fakeFinal = netD(torch.stack([data['initial']]))
        fakeFinal = (fakeFinal > 0.5).int()
        frob_norm = torch.norm((fakeFinal - (data['final'] > 0.5).int()).float())
        for i in range(len(max_norms)):
            if frob_norm < max_norms[i]:
                correct[i] += 1
    return [norm_correct/len(dataset) for norm_correct in correct]
