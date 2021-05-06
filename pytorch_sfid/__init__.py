__version__ = "0.0.2"
import pytorch_fid_wrapper as pfw
from pytorch_sfid import params
from pytorch_sfid.sfid import get_sfid, get_stats


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
