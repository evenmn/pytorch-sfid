# pytorch-sfid
Computing the Sliding Fréchet Inception Distance (SFID) between fake and real images with continous conditions, as suggested by [Ding et al., 2021](https://arxiv.org/abs/2011.07466). The package is heavily inspired by [pytorch-fid-wrapper](https://github.com/vict0rsch/pytorch-fid-wrapper), which again is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

NB: For `ncenters` given intervals and `ncond` conditions, the images are sorted in `ncenters^ncond` number of (overlapping) bins. Since this number of bins increases rapidly with `ncond`, a large number of conditions will cause out-of-memory error. In practice, up to 5 conditions is possible for a standard computer.

## Install
```bash
pip install git+https://github.com/evenmn/pytorch-sfid
```

## Prerequisites
1. torch
2. pytorch-fid-wrapper

## Usage
The package is centered around a function `get_sfid`, which takes a set of real and fake images and the corresponding conditions as `torch` tensors. The function `set_config` might be used to change the default `ncenters` and `radius`, but they can also be specified on-the-fly:

```python
import pytorch_sfid as ps

# optional
ps.set_config(ncenters=NCENTERS, radius=RADIUS, batch_size=BATCH_SIZE, dims=DIMS, device=DEVICE)

# optional
real_stats = ps.get_stats(real_images=REAL_IMAGES, real_attr=REAL_ATTR)

# get SFID
sfid = ps.get_sfid(FAKE_IMAGES, FAKE_ATTR, real_images=REAL_IMAGES, real_attr=REAL_ATTR)

or

sfid = ps.get_sfid(FAKE_IMAGES, FAKE_ATTR, real_stats=real_stats)
```
The number of centers and the radius should be chosen such that most bins will contain sufficiently many images.

## What does the code do?
The attributes are sorted into bins based on overlapping intervals. Then the indices of the images residing each bin is stored in an `ncond` x `ncenters`x `nimg` tensor, which is iterated. For each bin, we obtain the FID-score of the images, and the SFID score is found from summing over the FID score of all the bins:

<img src="https://latex.codecogs.com/gif.latex?\text{SFID}=\frac{1}{N_{\text{bins}}}\sum_{i=1}^{N_{\text{bins}}}\text{FID(bin)}" />.

The code itself utilizes `torch` only, and should be sufficiently fast. However, to find the FID values, we use `pytorch-fid`, which can be perceived as slow (even though it is as fast as it can be). 

## To do
1. Make more conditions possible by ignoring certain conditions or merge bins of similar conditions.
2. Allow different ncenters and radii for different conditions

## Acknowledgements
I will kindly thank Halvard Sutterud ([@halvarsu](https://github.com/halvarsu)) for implementation help and mental support during the development of this package. 

## License
[APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0)
