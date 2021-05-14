import torch
import pytorch_fid_wrapper as pfw
from pytorch_fid_wrapper.fid_score import calculate_frechet_distance as cfd
from pytorch_sfid import params as ps_params


def get_bins(attr, ncenters, radius):
    """Sort attributes into bins and return indices

    Parameters:
    -----------
    attr : torch FloatTensor, N x Na
        Attributes
    """
    # sort attributes
    centers = torch.linspace(0, 1, ncenters).to(attr.device)
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
    real_images : torch FloatTensor, N x 3 x H x W
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
        #real_local = real_local.repeat(1, 3, 1, 1)

        if real_local.shape[0] > 8:
            m, s = pfw.get_stats(real_local, batch_size=batch_size, dims=dims, device=device)
            m, s = torch.FloatTensor(m), torch.FloatTensor(s)
            if torch.all(torch.isfinite(m)) and torch.all(torch.isfinite(s)):
                real_m.append(m)
                real_s.append(s)
            else:
                real_m.append(None)
                real_s.append(None)
        else:
            real_m.append(None)
            real_s.append(None)

        if prnt:
            print("[{}/{}]  num real: {:>6}".format(i, bins.shape[0], real_local.shape[0]))

    return real_m, real_s, min_attr, max_attr


def calculate_frechet_distance(stats1, stats2, eps=1e-6, prnt=False):
    """Torch implementation of Frechet Distance
    """
    mu1s, sigma1s, _, _ = stats1
    mu2s, sigma2s, _, _ = stats2

    assert len(mu1s) == len(mu2s)
    nbins = len(mu1s)

    fid_cum = 0
    for i, (mu1, sigma1, mu2, sigma2) in enumerate(zip(mu1s, sigma1s, mu2s, sigma2s)):
        if mu1 is not None and mu2 is not None:
            fid_local = cfd(mu1, sigma1, mu2, sigma2, eps)
        if prnt:
            print("[{}/{}]  FID: {:10.4f}".format(i, nbins, fid_local))
        fid_cum += fid_local
    return fid_cum / nbins


def get_sfid(fake_images, fake_attr, real_images=None, real_attr=None,
             real_stats=None, ncenters=None, radius=None, batch_size=None,
             dims=None, device=None, prnt=False):
    """Sliding Frechet Inception Distance

    Parameters
    ----------
    real : torch FloatTensor, Nr x 3 x H x W
        Real images
    real_attr : torch FloatTensor, Nr x Na
        Attributes of real images
    fake : torch FloatTensor, Nf x 3 x H x W
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

        # get stats
        real_stats = get_stats(real_images, real_attr, ncenters, radius,
                               batch_size, dims, device, prnt)
        fake_stats = get_stats(fake_images, fake_attr, ncenters, radius, 
                               batch_size, dims, device, prnt)

    else:
        # standarize attributes (between 0 and 1)
        zero = torch.zeros(1).to(fake_attr.device)
        one = torch.ones(1).to(fake_attr.device)

        real_m, real_s, min_attr, max_attr = real_stats
        fake_attr = (fake_attr - min_attr) / (max_attr - min_attr)

        fake_attr = torch.where(fake_attr > 1, one, fake_attr)
        fake_attr = torch.where(fake_attr < 0, zero, fake_attr)

        # get stats
        fake_stats = get_stats(fake_images, fake_attr, ncenters, radius, 
                               batch_size, dims, device, prnt)

    sfid = calculate_frechet_distance(real_stats, fake_stats, prnt=prnt)
    return sfid
