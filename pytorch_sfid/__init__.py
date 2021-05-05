__version__ = "0.0.1"
from pytorch_sfid import params
import pytorch_fid_wrapper as pfw


def set_config(ncenters=None, radius=None, batch_size=None, dims=None,
               device=None):
    if ncenters is not None:
        assert isinstance(ncenters, int)
        assert ncenters > 0
        params.ncenters = ncenters
    if radius is not None:
        assert radius > 0
        assert 1 > radius
        params.radius = radius

    pfw.set_config(batch_size, dims, device)
