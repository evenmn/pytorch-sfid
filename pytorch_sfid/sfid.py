import torch
import pytorch_fid_wrapper as pfw
from pytorch_sfid import params as ps_params


def get_stats(real):
    pass


def sfid_score(real_images, real_attr, fake_images, fake_attr, ncenters=None,
               radius=None, batch_size=None, dims=None, device=None):
    """Sliding Frechet Inception Distance

    Parameters
    ----------
    real : torch FloatTensor, Nr x Nc x H x W
        Real images
    real_attr : torch FloatTensor, Nr x Na
        Attributes of real images
    fake : torch FloatTensor, Nf x Nc x H x W
        Fake images
    fake_attr : torch FloatTensor, Nf x Na
        Attributes of fake images
    """

    assert real_images.shape[0] == real_attr.shape[0]
    assert fake_images.shape[0] == fake_attr.shape[0]
    assert real_attr.shape[1] == fake_attr.shape[1]

    nreal = real_images.shape[0]
    nfake = fake_images.shape[0]
    ncond = real_attr.shape[1]
    nbins = ncenters ** ncond

    if ncenters is None:
        ncenters = ps_params.ncenters
    if radius is None:
        radius = ps_params.radius

    # standardize attributes (between 0 and 1)
    attr = torch.cat([real_attr, fake_attr], dim=0)

    min_attr, _ = attr.min(0)
    max_attr, _ = attr.max(0)

    real_attr = (real_attr - min_attr) / (max_attr - min_attr)
    fake_attr = (fake_attr - min_attr) / (max_attr - min_attr)

    # sort attributes in (overlapping) Na-dimensional bins
    centers = torch.linspace(0, 1, ncenters)
    lower_edge = centers - radius
    upper_edge = centers + radius

    # find which samples that belong to which bins
    reals = torch.logical_and(real_attr[:, :, None] > lower_edge[None, None, :],
                              real_attr[:, :, None] < upper_edge[None, None, :])
    fakes = torch.logical_and(fake_attr[:, :, None] > lower_edge[None, None, :],
                              fake_attr[:, :, None] < upper_edge[None, None, :])

    reals = reals.swapaxes(0, 1).swapaxes(1, 2)
    fakes = fakes.swapaxes(0, 1).swapaxes(1, 2)
    bins_real = reals[0]
    bins_fake = fakes[0]
    for i in range(1, ncond):
        bins_real = torch.einsum('i...,j...->ij...', bins_real, reals[i])
        bins_fake = torch.einsum('i...,j...->ij...', bins_fake, fakes[i])

    bins_real = bins_real.reshape(-1, nreal)
    bins_fake = bins_fake.reshape(-1, nfake)

    val_fid = 0
    for i in range(nbins):
        indices_real = torch.where(bins_real[i])[0]
        indices_fake = torch.where(bins_fake[i])[0]
        real_local = real_images[indices_real]
        fake_local = fake_images[indices_fake]

        real_local = real_local.repeat(1, 3, 1, 1)
        fake_local = fake_local.repeat(1, 3, 1, 1)

        if real_local.shape[0] > 1 and fake_local.shape[0] > 1:
            print(real_local.shape[0], fake_local.shape[0])
            val_fid += pfw.fid(fake_local, real_images=real_local,
                               batch_size=batch_size, dims=dims, device=device)
    return val_fid / nbins


if __name__ == "__main__":
    real_images = torch.rand(10000, 1, 128, 128)
    real_attr = torch.rand((10000, 3))
    fake_attr = torch.rand((10000, 3))
    sfid_score(real_images, real_attr, real_images, fake_attr, radius=0.19, ncenters=3)
