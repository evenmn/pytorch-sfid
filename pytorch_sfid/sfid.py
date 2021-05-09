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
              batch_size=None, dims=None, device=None, prnt=False):
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
            m, s = pfw.get_stats(real_local, batch_size=batch_size, dims=dims, device=device)
            real_m.append(m)
            real_s.append(s)
        else:
            real_m.append(None)
            real_s.append(None)

        if prnt:
            print("[{}/{}]  num real: {:>6}".format(i, bins.shape[0], real_local.shape[0]))

    return real_m, real_s, min_attr, max_attr


def get_sfid(fake_images, fake_attr, real_images=None, real_attr=None,
             real_stats=None, ncenters=None, radius=None, batch_size=None,
             dims=None, device=None, prnt=False):
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

    if ncenters is None:
        ncenters = ps_params.ncenters
    if radius is None:
        radius = ps_params.radius

    ncond = fake_attr.shape[1]
    nbins = ncenters ** ncond

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
        fake_attr = torch.where(fake_attr > 1, torch.ones(1), fake_attr)
        fake_attr = torch.where(fake_attr < 0, torch.zeros(1), fake_attr)
        bins_fake = get_bins(fake_attr, ncenters, radius)

    fid_cum = 0
    for i in range(nbins):
        indices_fake = torch.where(bins_fake[i])[0]
        fake_local = fake_images[indices_fake]
        fake_local = fake_local.repeat(1, 3, 1, 1)

        if real_images is not None:
            indices_real = torch.where(bins_real[i])[0]
            real_local = real_images[indices_real]
            real_local = real_local.repeat_interleave(3)

            if real_local.shape[0] > 1 and fake_local.shape[0] > 1:
                fid_local = pfw.fid(fake_local, real_images=real_local,
                                    batch_size=batch_size, dims=dims, device=device)
        else:
            real_m_local, real_s_local = real_m[i], real_s[i]
            if real_s_local is not None and fake_local.shape[0] > 1:
                fid_local = pfw.fid(fake_local, real_m=real_m_local, real_s=real_s_local,
                                    batch_size=batch_size, dims=dims, device=device)
        if prnt:
            print("[{}/{}]  num fake: {:>6}  FID: {:10.4f}".format(i, nbins, fake_local.shape[0], fid_local))
        fid_cum += fid_local
    return fid_cum / nbins
