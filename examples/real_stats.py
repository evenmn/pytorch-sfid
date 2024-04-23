import torch
import pytorch_sfid as ps

Nr = 1800
Nf = 10000
A = 3

real_images = torch.rand(Nr, 1, 128, 128)
fake_images = torch.rand(Nf, 1, 128, 128)
real_attr = torch.rand((Nr, A))
fake_attr = torch.rand((Nf, A))

ps.set_config(ncenters=6, radius=0.2, batch_size=8)
real_stats = ps.get_stats(real_images, real_attr, prnt=True)
sfid = ps.get_sfid(fake_images, fake_attr, real_stats=real_stats, prnt=True)
print("SFID value is: ", sfid)
