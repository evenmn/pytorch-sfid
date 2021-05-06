import torch
import pytorch_fid_wrapper as pfw
from pytorch_sfid import params as ps_params


def get_bins(attr, ncenters, radius):
    """Sort attributes into bins and return indices
    
    Parameters:
    -----------
    attr : torch FloatTensor, N x Na
        Attributes
    """
    # sort attributes
    centers = torch.linspace(0, 1, ncenters)
    lower_edge = centers - radius
    upper_edge = centers + radius
    smt = torch.logical_and(attr[:, :, None] > lower_edge[None, None, :],
                            attr[:, :, None] < upper_edge[None, None, :])

    # sort into bins by doing the generalized out product between all rows
    smt = smt.swapaxes(0, 1).swapaxes(1, 2)
    bins = smt[0]
    for i in range(1, attr.shape[1]):
        bins = torch.einsum('i...,j...->ij...', bins, smt[i])

    # return flattened bin tensor
    return bins.reshape(-1, attr.shape[0])


def get_stats(real_images, real_attr, ncenters=None, radius=None,
              batch_size=None, dims=None, device=None):
    """Get statistics of real images
    
    Parameters:
    -----------
    real_images : torch FloatTensor, N x C x H x W
    """

    if ncenters is None:
        ncenters = ps_params.ncenters
    if radius is None:
        radius = ps_params.radius

    min_attr, _ = real_attr.min(0)
    max_attr, _ = real_attr.max(0)

    real_attr = (real_attr - min_attr) / (max_attr - min_attr)

    bins = get_bins(real_attr, ncenters, radius)

    real_m, real_s = [], []
    for i in range(bins.shape[0]):
        indices = torch.where(bins[i])[0]
        real_local = real_images[indices]
        real_local = real_local.repeat(1, 3, 1, 1)

        if real_local.shape[0] > 1:
            m, s = pfw.get_stats(real_local, batch_size, dims, device)
            real_m.append(m)
            real_s.append(s)
        else:
            real_m.append(None)
            real_s.append(None)
    return real_m, real_s, min_attr, max_attr


def get_sfid(fake_images, fake_attr, real_images=None, real_attr=None,
             real_stats=None, ncenters=None, radius=None, batch_size=None,
             dims=None, device=None):
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
    
    assert (real_images is not None and real_attr is not None) or real_stats is not None
    assert fake_images.shape[0] == fake_attr.shape[0]

    ncond = fake_attr.shape[1]
    nbins = ncenters ** ncond

    if ncenters is None:
        ncenters = ps_params.ncenters
    if radius is None:
        radius = ps_params.radius

    if real_images is not None:
        # standardize attributes (between 0 and 1)
        attr = torch.cat([real_attr, fake_attr], dim=0)

        min_attr, _ = attr.min(0)
        max_attr, _ = attr.max(0)

        real_attr = (real_attr - min_attr) / (max_attr - min_attr)
        fake_attr = (fake_attr - min_attr) / (max_attr - min_attr)

        # get bins
        bins_real = get_bins(real_attr, ncenters, radius)
        bins_fake = get_bins(fake_attr, ncenters, radius)

    else:
        real_m, real_s, min_attr, max_attr = real_stats
        fake_attr = (fake_attr - min_attr) / (max_attr - min_attr)
        bins_fake = get_bins(fake_attr, ncenters, radius)

    '''
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
    '''

    val_fid = 0
    for i in range(nbins):
        indices_fake = torch.where(bins_fake[i])[0]
        fake_local = fake_images[indices_fake]
        fake_local = fake_local.repeat(1, 3, 1, 1)

        if real_images is not None:
            indices_real = torch.where(bins_real[i])[0]
            real_local = real_images[indices_real]
            real_local = real_local.repeat(1, 3, 1, 1)

            if real_local.shape[0] > 1 and fake_local.shape[0] > 1:
                val_fid += pfw.fid(fake_local, real_images=real_local,
                                   batch_size=batch_size, dims=dims, device=device)
        else:
            real_m_local, real_s_local = real_m[i], real_s[i]
            if real_s_local is not None and fake_local.shape[0] > 1:
                val_fid += pfw.fid(fake_local, real_m=real_m_local, real_s=real_s_local,
                                   batch_size=batch_size, dims=dims, device=device)
        print(val_fid)
    return val_fid / nbins


if __name__ == "__main__":
    N = 1000
    real_images = torch.rand(N, 1, 128, 128)
    real_attr = torch.rand((N, 3))
    fake_attr = torch.rand((N, 3))

    real_stats = get_stats(real_images, real_attr)
    sfid_score(real_images, real_attr, real_stats=real_stats, radius=0.6, ncenters=3)
