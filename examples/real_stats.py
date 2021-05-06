import torch
import pytorch_sfid as ps

N = 1000
A = 1

real_images = torch.rand(N, 1, 32, 32)
real_attr = torch.rand((N, A))
fake_attr = torch.rand((N, A))

ps.set_config(ncenters=3, radius=0.6, batch_size=32)
real_stats = ps.get_stats(real_images, real_attr)
sfid = ps.get_sfid(real_images, fake_attr, real_stats=real_stats, prnt=True)
print("SFID value is: ", sfid)
