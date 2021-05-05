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

    reals = torch.zeros((ncenters, nreal, ncond), dtype=bool)
    fakes = torch.zeros((ncenters, nfake, ncond), dtype=bool)
    for i, center in enumerate(centers):
        reals[i] = (real_attr > center - radius) & (center + radius > real_attr)
        fakes[i] = (fake_attr > center - radius) & (center + radius > fake_attr)

    print(reals.shape)
    stop

    # divide attributes into bins
    boundaries = torch.linspace(0, 1, ncenters)
    # torch.bucketize works the same way as np.digitize, but on torch tensors
    ind_real = torch.bucketize(real_attr, boundaries)
    ind_fake = torch.bucketize(fake_attr, boundaries)



    val_fid = 0
    for center in torch.linspace(0, 1, ncenters):
        real_ind = torch.where((center-radius)<real_attr<(center+radius))
        fake_ind = torch.where((center-radius)<fake_attr<(center+radius))

        real_local = real_images[real_ind]
        fake_local = fake_images[fake_ind]

        val_fid += pfw.fid(fake_local, real_images=real_local,
                           batch_size=batch_size, dims=dims, device=device)

    return val_fid / ncenters


if __name__ == "__main__":
    real_images = torch.rand(640000, 1, 128, 128)
    real_attr = torch.rand(640000, 100)
    sfid_score(real_images, real_attr, real_images, real_attr)
