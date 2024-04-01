import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc_network = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,64),
            nn.LeakyReLU(0.1),
            nn.Linear(64,16),
            nn.LeakyReLU(0.1),
            nn.Linear(16,8),
            nn.LeakyReLU(0.1),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc_network(x)
    
class generator(nn.Module):
    def __init__(self,latent_dim,img_dim):
        super().__init__()
        self.gen_network = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,512),
            nn.LeakyReLU(0.1),
            nn.Linear(512,img_dim),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen_network(x)
    
#configuration and hyper-parameters
latent_dim = 64
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dim = 784
batch_size = 32
epochs = 70
gen = generator(latent_dim,img_dim).to(device)
disc = discriminator(img_dim).to(device)
fixed_noise = torch.randn((batch_size,latent_dim)).to(device)  #-->for visualization
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
data = datasets.MNIST(root='dataset/',transform=transforms,download=True)
dataLoader = DataLoader(data,batch_size=batch_size,shuffle=True)
gen_optimizer = optim.Adam(gen.parameters(),lr=lr)
disc_optimizer = optim.Adam(disc.parameters(),lr=lr)
criterion = nn.BCELoss()
real_writer = SummaryWriter(f"runs/GAN/real")
fake_writer = SummaryWriter(f"runs/GAN/fake")
step=0

#training loop
for epoch in range(epochs):
    for idx,(real,_) in enumerate(dataLoader):
        real = real.view(-1,28*28).to(device)
        #---------train the discriminator---------#
        #d_loss = max( lod(D_real) + log(1-D(G(noise))) )
        #BCEloss = -( y*log(y_hat) + (1-y)*log(1-y_hat) )
        #putting y=1 and y_hat = D_real
        #BCEloss = -log(D_real) --> minimizing this is same as maximising log(D_real)
        D_real = disc(real).view(-1)
        noise = torch.randn(batch_size,latent_dim).to(device)
        fake = gen(noise)
        D_fake = disc(fake).view(-1)
        loss_D_fake = criterion(D_fake,torch.zeros_like(D_fake))
        loss_D_real = criterion(D_real,torch.ones_like(D_real))
        D_loss = (loss_D_fake + loss_D_real)/2
        disc.zero_grad()
        D_loss.backward(retain_graph=True)
        disc_optimizer.step()

        #--------train the generator--------------#
        #g_loss = min( log( 1 - D(g(z))))
        out = disc(fake).view(-1)
        G_loss = criterion(out,torch.ones_like(out))
        gen.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        if idx == 0:  #-->visualize outputs and print loss after starting of each epoch
            print(f"epoch:{epoch}/{epochs}  G_loss:{G_loss}   D_loss:{D_loss}")

            with torch.no_grad():
                fake_imgs = gen(fixed_noise).reshape(-1,1,28,28)
                real_imgs = real.reshape(-1,1,28,28)  #reshape flattened matrix
                #build grids for displaying images
                real_imgs_grid = torchvision.utils.make_grid(real_imgs,normalize=True)
                fake_imgs_grid = torchvision.utils.make_grid(fake_imgs,normalize=True)
                #add the images
                real_writer.add_image("real images",real_imgs_grid,global_step=step)
                fake_writer.add_image("Generated images",fake_imgs_grid,global_step=step)
                step+=1
                

    