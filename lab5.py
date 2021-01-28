#################################################################
#--------------------------import-------------------------------#
#################################################################
import argparse
import evaluator
import dataloader
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader

#################################################################
#-------------------------Setting-------------------------------#
#################################################################

parser = argparse.ArgumentParser("cDCGAN")
parser.add_argument('--result_dir', type=str, default='result/image')
parser.add_argument('--result_generator', type=str, default='result/generator')
parser.add_argument('--result_discriminator', type=str, default='result/discriminator')
parser.add_argument('--result_module', type=str, default='result/module')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=500)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel
parser.add_argument('--nfeature', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.0002)
#betas = (0.0, 0.99) # adam optimizer beta1, beta2
betas = (0.0, 0.9)
config, _ = parser.parse_known_args()

#################################################################
#---------------------------Def---------------------------------#
#################################################################

"""
   Generator
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.nz + config.nfeature, 512, 4, 1, 0, bias=False),   #torch.Size([32, 512, 4, 4]) , kernal size, stride, padding
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, config.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )  
    def forward(self, x, attr):
        attr = attr.view(-1, config.nfeature, 1, 1)
        x = torch.cat([x, attr], 1)         
        x = self.main(x)
        return x 
        
"""
   Discriminator
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config.nfeature, 64 * 64)
        self.main = nn.Sequential(
            nn.Conv2d(config.nc + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
    
    def forward(self, x, attr):
        """
           attr : torch.Size([32, 1, 64, 64])
           x : torch.Size([32, 3, 64, 64])
        """
        attr = self.feature_input(attr).view(-1, 1, 64, 64)
        x = torch.cat([x, attr], 1)
        x = self.main(x)
        return x.view(-1, 1)
        
"""
   Training
"""
ev = evaluator.evaluation_model()
class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)
        #self.optimizer_g = optim.SGD(self.generator.parameters(), lr=config.lr)
        #self.optimizer_d = optim.SGD(self.discriminator.parameters(), lr=config.lr)
        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()        
        
    def train(self, dataloader):
        """
           noise size : torch.Size([32, 100, 1, 1])
           label_real : ([32,1]) all 1
           label_fake : ([32,1]) all 0
           one_hot_labels : ([32,24]) all 0
        """
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())    
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())    
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())    
        one_hot_labels = torch.FloatTensor(config.batch_size,config.nfeature)
        one_hot_labels = one_hot_labels.cuda()
        best_acc=0
        #----------------- prepare test data -----------------#
        with open('objects.json', 'r') as f:
            data = json.load(f)            
        with open('test.json', 'r') as f:
            test_data = json.load(f)
        
        fixed_noise =  Variable(torch.FloatTensor(32, 100, 1, 1).cuda()) # test noise
        fixed_noise.data.resize_(32, 100, 1, 1).normal_(0, 1)            # normal distribution
        fixed_labels = torch.zeros(32,24)                                # test label

        for i in range(32):
            for j in range(len(test_data[i])):
                fixed_labels[i,data[test_data[i][j]]]=1
        fixed_labels = Variable(fixed_labels.cuda())
        
        #----------------------test data--------------------------#
        
        for epoch in range(config.nepoch):
            
            for i, (data, attr) in enumerate(dataloader, 0):
                #------train discriminator------#
                print('----')
                print(attr)
                self.discriminator.zero_grad()

                batch_size = data.size(0)
                label_real = Variable(torch.FloatTensor(batch_size, 1).fill_(1).cuda())
                label_fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0).cuda())
                noise = Variable(torch.FloatTensor(batch_size, config.nz, 1, 1).cuda())
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)
                one_hot_labels.resize_(batch_size, config.nfeature).zero_()
                one_hot_labels = attr 
                
                """
                for i in range(len(attr)):
                    tolist=attr[i].split(",")
                    for j in range(len(tolist)):
                        tolist[j]=int(tolist[j])
                    for j in range(len(tolist)):
                        one_hot_labels[i][tolist[j]]=1
                """
                  
                """
                   real, one_hot_labels : training data (real exist data), labels of real training data
                """
                real = Variable(data.cuda())  
                one_hot_labels = Variable(one_hot_labels.cuda())
                dis_real = self.discriminator(real, one_hot_labels)
                fake = self.generator(noise, one_hot_labels )
                dis_fake = self.discriminator(fake.detach(), one_hot_labels )
                d_loss = self.loss(dis_real, label_real) + self.loss(dis_fake, label_fake) # real label
                d_loss.backward()
                self.optimizer_d.step()

                #-----------train generator------------#
                self.generator.zero_grad()
                dis_fake = self.discriminator(fake, one_hot_labels )
                g_loss = self.loss(dis_fake, label_real) # trick the fake into being real
                g_loss.backward()
                self.optimizer_g.step()
            #-------------print result-------------#
            print("epoch{:03d} dis_real: {}, dis_fake: {}".format(epoch, dis_real.mean(), dis_fake.mean()))
            print("epoch{:03d} d_loss: {}, g_loss: {}".format(epoch,d_loss,g_loss))
            test_fake = self.generator(fixed_noise, fixed_labels)
            acc=ev.eval(test_fake.data,fixed_labels)
            print("The epoch is {}  The accurayc is {}".format(epoch , acc ))
            if(acc>best_acc):
                torch.save({
                    'epoch': epoch,
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'acc': acc,
                    'fixed_noise':fixed_noise,
                    'fixed_labels':fixed_labels,
                },"{}/{}.tar".format(config.result_module,epoch))             
                best_acc=acc               
            vutils.save_image(test_fake.data, '{}/result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)



#################################################################
#---------------------------Main--------------------------------#
#################################################################
objects_path = 'objects.json'  #label 0~23
train_path = 'train.json'      #the label containing in each training image
test_path = 'test.json'        #the labels that GAN try to produce
image_path = 'iclevr/'
Batch_size = 32
IclverData_train = dataloader.Iclver_Data(objects_path, train_path, image_path)
train_loader     = DataLoader(dataset = IclverData_train, batch_size = Batch_size, num_workers = 4)

#------------train-------------#
#trainer = Trainer()
#trainer.train(train_loader)

#################################################################
#------------------------Load modle-----------------------------#
#################################################################
load_module ='/home/ubuntu/DL_LAB5/result/module/118.tar'

checkpoint = torch.load(load_module)
generator =  Generator()
discriminator =  Discriminator()
generator.cuda()
discriminator.cuda()
generator.load_state_dict(checkpoint['generator'])
discriminator.load_state_dict(checkpoint['discriminator'])
epoch = checkpoint['epoch']
fixed_noise = checkpoint['fixed_noise']
fixed_labels = checkpoint['fixed_labels']
test_fake  =   generator(fixed_noise, fixed_labels)
acc=ev.eval(test_fake.data,fixed_labels)
print("accuracy = ",acc)
print("epoch number = ", epoch)
