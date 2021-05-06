import torch
import pytorch_sfid as ps

N = 1000
real_images = torch.rand(N, 1, 32, 32)
real_attr = torch.rand((N, 3))
fake_attr = torch.rand((N, 3))

ps.set_config(ncenters=3, radius=0.6, batch_size=32)
sfid = ps.get_sfid(fake_images, fake_attr, real_images=real_images, real_attr=real_attr)
