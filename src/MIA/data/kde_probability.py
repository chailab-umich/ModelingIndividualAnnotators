import torch
import torch.nn as nn
import numpy as np
from numpy import pi as π
from torch_dct import dct_2d, idct_2d
from numpy import histogram2d
import time

eps = 1e-12
def noise(std, precision):
    max_noise = (0.5*std) + eps
    return (-2*max_noise)*torch.rand(std.shape,device=std.device,dtype=precision) + max_noise
    # return np.random.uniform(-max_noise, max_noise, 1)[0]

# Expect input of soft_act: batch_size x max_num_annotators
# for batched samples that don't have evaluations from all evaluations these values should be torch.nan
def get_random_observations_fast(soft_act, soft_val, n=200, precision=torch.float64):
    device = soft_act.device
    max_n = soft_act.shape[1]
    batch_size = soft_act.shape[0]

    # Define output variables 
    random_act = torch.zeros((batch_size, n), dtype=precision, device=device)
    random_val = torch.zeros((batch_size, n), dtype=precision, device=device)

    # For each sample
    for batch_idx in range(len(soft_act)):
        # Get samples to take observations from 
        batch_act = soft_act[batch_idx]
        batch_val = soft_val[batch_idx]

        # Remove nan values from samples
        batch_act = batch_act[~batch_act.isnan()]
        batch_val = batch_val[~batch_val.isnan()]

        # Calculate the max number of annotators for this data 
        max_n = len(batch_act)

        # Expand the batches to have n copies of the data 
        eb_act = batch_act.unsqueeze(dim=0).repeat([n,1])
        eb_val = batch_val.unsqueeze(dim=0).repeat([n,1])

        # Now randomly shuffle the order of annotators in each of these n copies of the data
        label_permutations = torch.stack([torch.randperm(max_n) for _ in range(n)]).to(device)
        shuffled_act = torch.gather(eb_act, dim=-1, index=label_permutations)
        shuffled_val = torch.gather(eb_val, dim=-1, index=label_permutations)

        # For each copy of the data, randomly use up to max_n annotators in each
        num_labels_to_use = torch.randint(low=1, high=max_n+1, size=[200], device=device)

        # Calculate a numerical index [[0,1,2,...,max_n], [0,1,2,...,max_n]... (n times)]
        # Then when these are < num_labels_to_use we want to use that label
        # Since data was shuffled earlier this will ensure that selecting e.g. label 0 in this case will not always correspond to the first annotator provided
        indexes = torch._dim_arange(shuffled_act, dim=-1).unsqueeze(dim=0).repeat([n,1])
        indexes = indexes < num_labels_to_use.unsqueeze(dim=-1)

        # Inverse the selection to overwrite annotators not to be used in observation with torch.nan as nanmean can then be used to ignore this value 
        shuffled_act[~indexes] = torch.nan
        shuffled_val[~indexes] = torch.nan

        # Get observation means
        means = shuffled_act.nanmean(dim=-1)
        means_val = shuffled_val.nanmean(dim=-1)

        # Add sampled noise to each mean
        act_observation = means + noise(batch_act.std().unsqueeze(dim=0).repeat([n]), torch.float32)
        val_observation = means_val + noise(batch_val.std().unsqueeze(dim=0).repeat([n]), torch.float32)

        # Store the observations
        random_act[batch_idx,:] = act_observation
        random_val[batch_idx,:] = val_observation

    random_act = torch.clamp(random_act, min=-1, max=1)
    random_val = torch.clamp(random_val, min=-1, max=1)
    return random_act, random_val


def brent_torch_batched(fn, a, b, bs=32, precision=torch.float64):
    if precision == torch.float64:
        xtol=2e-12
        rtol=8.881784197001252e-16
    elif precision == torch.float32:
        xtol=2e-6
        rtol=4.76837158203125e-07 # Calculate via torch.finfo(torch.float32).eps*4 as in the scipy implementation
    elif precision == torch.float16:
        xtol=2e-3
        rtol=3.90625e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    goal = torch.as_tensor(0, dtype=precision, device=device)
    delta = torch.as_tensor([torch.finfo(precision).eps], device=device)
    a = torch.as_tensor(a, dtype=precision, device=device).unsqueeze(dim=0).repeat(bs,1)
    b = torch.as_tensor(b, dtype=precision, device=device).unsqueeze(dim=0).repeat(bs,1)
    a_fn = fn(a)
    b_fn = fn(b)
    check = a_fn * b_fn
    if (check >= 0).any():
        raise ValueError('Function not bracketed')

    # Swap a/b for batches where |f(a)| < |(b)|
    swaps = torch.abs(a_fn) < torch.abs(b_fn)
    if swaps.any():
        temp = a[swaps]
        temp_fn = a_fn[swaps]
        a[swaps] = b[swaps]
        b[swaps] = temp
        a_fn[swaps] = b_fn[swaps]
        b_fn[swaps] = temp_fn
        del temp, temp_fn

    c = a
    mflag = torch.as_tensor(True, device=device).unsqueeze(dim=0).repeat(bs,1)
    d = torch.inf # Ensure that conditions are false involving d for first runtime use of d 
    stop_calculation = torch.as_tensor(False, device=device).repeat(bs)
    iters = 0
    c_fn = a_fn
    found_roots = torch.zeros((bs,1), dtype=precision, device=device)
    while True:
        # print(a[~stop_calculation].shape)
        # c_fn = fn(c)#[~stop_calculation])
        do_quad_int = ~(torch.eq(a_fn, c_fn)) & ~(torch.eq(b_fn, c_fn))
        do_secant = ~do_quad_int

        s = torch.zeros(a.shape,dtype=precision,device=device)
        if do_quad_int.any():
            part1 = (a[do_quad_int]*b_fn[do_quad_int]*c_fn[do_quad_int])/((a_fn[do_quad_int]-b_fn[do_quad_int])*(a_fn[do_quad_int]-c_fn[do_quad_int]))
            part2 = (b[do_quad_int]*a_fn[do_quad_int]*c_fn[do_quad_int])/((b_fn[do_quad_int]-a_fn[do_quad_int])*(b_fn[do_quad_int]-c_fn[do_quad_int]))
            part3 = (c[do_quad_int]*a_fn[do_quad_int]*b_fn[do_quad_int])/((c_fn[do_quad_int]-a_fn[do_quad_int])*(c_fn[do_quad_int]-b_fn[do_quad_int]))
            s[do_quad_int] = part1 + part2 + part3

        if do_secant.any():
            s[do_secant] = b[do_secant] - b_fn[do_secant]*((b[do_secant]-a[do_secant])/(b_fn[do_secant]-a_fn[do_secant]))
            # s[do_secant] = (a_fn[do_secant]*b[do_secant] - b_fn[do_secant]*a[do_secant])/(a_fn[do_secant]-b_fn[do_secant])

        # print(s)

        # Now we do large check for bisection method and setting mflag
        # cond1 = (s < (3*a+b)/4) | (s > b)
        cond1 = (s-(3*a+b)/4)*(s-b) >= 0
        cond2 = mflag & (torch.abs(s-b) >= torch.abs(b-c)/2)
        cond3 = ~mflag & (torch.abs(s-b) >= torch.abs(c-d)/2)
        cond4 = mflag & (torch.abs(b-c) < torch.abs(delta))
        cond5 = ~mflag & (torch.abs(c-d) < torch.abs(delta))
        do_bisection = cond1 | cond2 | cond3 | cond4 | cond5
        if do_bisection.any():
            s[do_bisection] = (a[do_bisection]+b[do_bisection])/2
            mflag[do_bisection] = True
        if (~do_bisection).any():
            mflag[~do_bisection] = False

        s_fn = fn(s)
        d = c
        c = b
        c_fn = b_fn
        # print(a_fn.shape, s_fn.shape)
        b_become_s = (a_fn * s_fn) < 0
        a_become_s = ~b_become_s
        if b_become_s.any():
            # print(b.shape, b_become_s.shape, s.shape)
            b[b_become_s] = s[b_become_s]
            # b_fn = fn(b) # Can we just partially compute this? 
            b_fn[b_become_s] = s_fn[b_become_s]

        if a_become_s.any():
            a[a_become_s] = s[a_become_s]
            # a_fn = fn(a)
            a_fn[a_become_s] = s_fn[a_become_s]

        a_becomes_b = torch.abs(a_fn) < torch.abs(b_fn)
        if a_becomes_b.any():
            temp = a[a_becomes_b]
            temp_fn = a_fn[a_becomes_b]
            a[a_becomes_b] = b[a_becomes_b]
            b[a_becomes_b] = temp
            a_fn[a_becomes_b] = b_fn[a_becomes_b]
            b_fn[a_becomes_b] = temp_fn
            # b_fn = fn(b)

        # print(b_fn, b_fn == 0, b_fn.isclose(goal, atol=xtol, rtol=rtol), torch.abs(b-a) < torch.as_tensor(1e-8, dtype=torch.float64), torch.abs(b-a).isclose(goal, atol=xtol, rtol=rtol))
        # Either if absolute b_fn is withing xtol of 0 then root. If b-a < xtol + rtol*a then root (as the root is within these bounds up to the tolerance)
        b_root = b_fn.isclose(goal, atol=xtol, rtol=rtol) | b.isclose(a, atol=xtol, rtol=rtol)# & ~stop_calculation.unsqueeze(dim=1)
        if b_root.any():
            found_roots[b_root] = b[b_root]

        s_root = s_fn.isclose(goal, atol=xtol, rtol=rtol)# & ~stop_calculation.unsqueeze(dim=1)
        if s_root.any():
            found_roots[s_root] = s[s_root]

        stop_calculation = (stop_calculation | b_root.squeeze()) | s_root.squeeze()
        if stop_calculation.all():
            # print('kde finished in', iters)
            break
        iters += 1
        if iters > 100:
            raise ValueError('Max iterations')
    # r_val = fn(found_roots)
    # print('SHOULD BE CLOSE TO 0:', r_val.mean(), r_val.min(), r_val.max())
    return found_roots

# Finally develop the probability distributions 
def kde_probability_bs(act, val, prob_grid_size=5, temperature=512, use_soft_histogram=True, density_grid_size=512, precision=torch.float64, skip_observations=False):
    device = act.device
    # print('Getting kde probability distribution')
    assert len(act) == len(val)
    value_error = True
    attempt = 1
    number_of_observations = 200
    while value_error:
        try:
            if attempt >= 15:
                # all_values = [act, val, prob_grid_size, temperature, use_soft_histogram, density_grid_size]
                # save_pk(f'/z/tavernor/ModelingIndividualEvaluators/failed_brent_logs/failed_brent{time.time()}.pk', all_values)
                # save_pk(f'/z/tavernor/PaperReimplementations/Interspeech2024/failed_brent_logs/failed_brent_latest.pk', all_values)
                return
            if not skip_observations:
                random_act, random_val = get_random_observations_fast(act, val, n=number_of_observations, precision=precision)
            else:
                random_act, random_val = torch.clamp(act, min=-1, max=1), torch.clamp(val, min=-1, max=1)
                # Since we can't regenerate these, we instead set attempt to already be 15 so that on failed optimisation kde fails immediately
                attempt = 15
            n = density_grid_size
            probability_distributions = torch.zeros((random_val.shape[0], prob_grid_size, prob_grid_size), dtype=torch.float32)
            # return kde2dtorchbs(random_act, random_val, soft_hist=use_soft_histogram, n=n, limits=((-1,1), (-1,1)), temperature=temperature), (random_act, random_val) # This might need to be done per batch
            density, grid, bandwidth = kde2dtorchbs(random_act.unsqueeze(dim=-1), random_val.unsqueeze(dim=-1), soft_hist=use_soft_histogram, n=n, limits=((-1,1), (-1,1)), temperature=temperature, precision=precision) # This might need to be done per batch
            # return kde2dtorch(random_val[0], random_act[0], n=n, limits=((-1,1), (-1,1))) # This might need to be done per batch
            if torch.isnan(density).any():
                print(f'NaN values in density (attempt {attempt}), retying with new {number_of_observations} random observations')
                attempt += 1
            else:
                value_error = False

                size = density.shape[1]
                xbins = torch.linspace(0,size,prob_grid_size+1, dtype=torch.int)
                ybins = torch.linspace(0,size,prob_grid_size+1, dtype=torch.int)
                # Want to iterate between each pair of bins to find mean values
                for x in range(1,prob_grid_size+1):
                    for y in range(1,prob_grid_size+1):
                        densities_in_range = density[:, xbins[x-1]:xbins[x], ybins[y-1]:ybins[y]]
                        mean = densities_in_range.mean(dim=[1,2])
                        probability_distributions[:,x-1,y-1] = mean
                # Test
                # probability_distributions = torch.log_softmax(probability_distributions.view(random_val.shape[0], -1), dim=1).view(probability_distributions.shape)
                return probability_distributions.to(device)
        except ValueError as e:
            print('bandwidth optimisation failed, attempt:', attempt)
            # raise e
            attempt += 1

class kde_debug:
    def __init__(self):
        self.line = 0
        _, self.total_memory = torch.cuda.mem_get_info()

    def log(self):
        torch.cuda.empty_cache()
        alloc_prop = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())*100
        total_prop = (torch.cuda.memory_allocated() / self.total_memory)*100
        print(self.line, f'{alloc_prop:.5f} | {total_prop:.5f}')
        self.line += 1

def kde2dtorchbs(x, y, n=256, limits=None, soft_hist=True, temperature=None, precision=torch.float64):
    """
    Estimates the 2d density from discrete observations.

    The input is two lists/arrays `x` and `y` of numbers that represent
    discrete observations of a random variable with two coordinate
    components. The observations are binned on a grid of n×n points.
    `n` will be coerced to the next highest power of two if it isn't
    one to begin with.

    Data `limits` may be specified as a tuple of tuples denoting
    `((xmin, xmax), (ymin, ymax))`. If any of the values are `None`,
    they will be inferred from the data. Each tuple, or even both of
    them, may also be replaced by a single value denoting the upper
    bound of a range centered at zero.

    After binning, the function determines the optimal bandwidth
    according to the diffusion-based method. It then smooths the
    binned data over the grid using a Gaussian kernel with a standard
    deviation corresponding to that bandwidth.

    Returns the estimated `density` and the `grid` (along each of the
    two axes) upon which it was computed, as well as the optimal
    `bandwidth` values (per axis) that the algorithm determined.
    Raises `ValueError` if the algorithm did not converge or `x` and
    `y` are not the same length.
    """
    # debug = kde_debug()
    if temperature is None:
        temperature = n/2
    # debug.log()
    # Convert to arrays in case lists are passed in.
    # x = array(x)
    # y = array(y)

    # Make sure numbers of data points are consistent.
    N = x.shape[1]
    # debug.log()
    if y.shape[1] != N:
        raise ValueError('x and y must have the same length.')
    # debug.log()

    N = torch.as_tensor(N, device=x.device)
    # debug.log()

    # Round up number of bins to next power of two.
    n = int(2**np.ceil(np.log2(n)))
    # debug.log()

    # Determine missing data limits.
    if limits is None:
        xmin = xmax = ymin = ymax = None
    elif isinstance(limits, tuple):
        (xlimits, ylimits) = limits
        if xlimits is None:
            xmin = xmax = None
        elif isinstance(xlimits, tuple):
            (xmin, xmax) = xlimits
        else:
            xmin = -xlimits
            xmax = +xlimits
        if ylimits is None:
            ymin = ymax = None
        elif isinstance(ylimits, tuple):
            (ymin, ymax) = ylimits
        else:
            ymin = -ylimits
            ymax = +ylimits
    else:
        xmin = -limits
        xmax = +limits
        ymin = -limits
        ymax = +limits
    if None in (xmin, xmax):
        delta = x.max() - x.min()
        if xmin is None:
            xmin = x.min() - delta/4
        if xmax is None:
            xmax = x.max() + delta/4
        xmin = xmin.item()
        xmax = xmax.item()
    if None in (ymin, ymax):
        delta = y.max() - y.min()
        if ymin is None:
            ymin = y.min() - delta/4
        if ymax is None:
            ymax = y.max() + delta/4
        ymin = ymin.item()
        ymax = ymax.item()
    Δx = xmax - xmin
    Δy = ymax - ymin
    # debug.log()

    # Bin samples on regular grid.
    device = x.device
    # debug.log()
    if soft_hist:
        # print('soft hist')
        hist_fn = SoftHistogram2Dbsfast(n, xmin, xmax, temperature, device=device, precision=precision)
        # debug.log()
        binned, xedges, yedges = hist_fn(x,y)
        # debug.log()
        del hist_fn
        # debug.log()
        # binned = binned.to(precision)
        # debug.log()
    else:
        print('hard hist')
        binned = []
        # debug.log()
        for i in range(x.shape[0]):
            (binned_part, xedges, yedges) = histogram2d(x[i].squeeze().cpu().numpy(), y[i].squeeze().cpu().numpy(), bins=n,
                                        range=((xmin, xmax), (ymin, ymax)))
            binned.append(binned_part)
        # debug.log()
        binned = torch.from_numpy(np.array(binned)).to(precision).to(device) # xedges, yedges should be the same for all runs
        # debug.log()

    # (binned, xedges, yedges) = out.hist, out.bin_edges[0], out.bin_edges[1]
    grid = (xedges[:-1], yedges[:-1])
    # debug.log()

    # Compute discrete cosine transform, then adjust first component.
    if precision == torch.float16:
        binned = binned.float() # Convert to float32 for dct as this does not support float16
    transformed = dct_2d(binned/N)
    # debug.log()
    del binned
    # debug.log()
    transformed[:, 0, :] /= 2
    # debug.log()
    transformed[:, :, 0] /= 2
    # debug.log()

    # Pre-compute squared indices and transform components before solver loop.
    min_precision = torch.float32 if precision == torch.float16 else precision
    k  = torch.arange(n, dtype=min_precision, device=device)          # "float" avoids integer overflow.
    # debug.log()
    k2 = k**2
    # debug.log()
    a2 = transformed**2
    # debug.log()

    # Define internal functions to be solved iteratively.
    def γ(t):
        Σ = ψ(0, 2, t) + ψ(2, 0, t) + 2*ψ(1, 1, t)
        γ = (2*π*N*Σ)**(-1/3)
        return (t - γ) / γ

    def ψ(i, j, t):
        if i + j <= 4:
            Σ  = abs(ψ(i+1, j, t) + ψ(i, j+1, t))
            C  = (1 + 1/2**(i+j+1)) / 3
            Πi = np.prod(np.arange(1, 2*i, 2))
            Πj = np.prod(np.arange(1, 2*j, 2))
            t  = (C*Πi*Πj / (π*N*Σ)) ** (1/(2+i+j))
        w = 0.5 * torch.ones(n, device=device, dtype=precision)
        w[0] = 1
        w = w * torch.exp(-π**2 * k2*t)
        wx = w * k2**i
        wy = w * k2**j
        return (-1)**(i+j) * π**(2*(i+j)) * torch.bmm(torch.bmm(wy.unsqueeze(dim=1),a2), wx.unsqueeze(dim=-1)).squeeze(dim=-1)
    # debug.log()
    # Solve for optimal diffusion time t*.
    try:
        # print('calling brent torch batched')
        # debug.log()
        # lambda t: t - γ(t) will return float32 for float16 precision due to dct requiring float32 conversion and issues with inf occuring
        if precision == torch.float16:
            fn = lambda t: (t - γ(t)).to(precision)
        else:
            fn = lambda t: t - γ(t)
        ts = brent_torch_batched(fn, 0.0, 0.1, bs=x.shape[0], precision=precision)
        # debug.log()
        # ts = ts.to(precision)
        # debug.log()
    except ValueError:
        raise ValueError('Bandwidth optimization did not converge.') from None

    # Calculate diffusion times along x- and y-axis.
    # debug.log()
    ψ02 = ψ(0, 2, ts)
    # debug.log()
    ψ20 = ψ(2, 0, ts)
    # debug.log()
    ψ11 = ψ(1, 1, ts)
    # debug.log()
    tx1 = (ψ02**(3/4) / (4*π*N*ψ20**(3/4) * (ψ11 + torch.sqrt(ψ02*ψ20))) )**(1/3)
    # debug.log()
    tx2 = (ψ20**(3/4) / (4*π*N*ψ02**(3/4) * (ψ11 + torch.sqrt(ψ02*ψ20))) )**(1/3)
    # debug.log()
    del ψ02, ψ20, ψ11, a2, γ, ψ
    # debug.log()

    # Note:
    # The above uses the nomenclature from the paper. In the Matlab
    # reference, tx1 is called t_y, while tx2 is t_x. This is a curious
    # change in notation. It may be related to the fact that image
    # coordinates are typically in (y,x) index order, whereas matrices,
    # such as the binned histogram (in Matlab as much as in Python),
    # are in (x,y) order. The Matlab code eventually does return
    # image-like index order, though it never explicitly transposes
    # the density matrix. That is implicitly handled by its custom
    # implementation of the inverse transformation (idct2d), which
    # only employs one matrix transposition, not two as its forward
    # counterpart (dct2d).

    # Apply Gaussian filter with optimized kernel.
    batched_outer = torch.bmm(torch.exp(-π**2 * k2 * tx2/2).unsqueeze(dim=-1), torch.exp(-π**2 * k2 * tx1/2).unsqueeze(dim=-2))
    # debug.log()
    smoothed = transformed * batched_outer
    # debug.log()
    del transformed, batched_outer, k2
    # debug.log()

    # Reverse transformation after adjusting first component.
    # debug.log()
    smoothed[:, 0, :] *= 2
    # debug.log()
    smoothed[:, :, 0] *= 2
    # debug.log()

    inverse = idct_2d(smoothed)
    # debug.log()
    del smoothed
    # debug.log()

    # Normalize density.
    density = inverse * n/Δx * n/Δy
    # Now remove negatives in density - as in the original matlab code 
    eps = torch.finfo(density.dtype).eps
    density[density < 0] = eps
    # debug.log()

    # Determine bandwidth from diffusion times.
    bandwidth = ([torch.sqrt(tx2).detach().cpu()*Δx, torch.sqrt(tx1).detach().cpu()*Δy])
    # debug.log()

    # Return results.
    return density, grid, bandwidth

class SoftHistogram2Dbsfast(nn.Module):
    def __init__(self, bins, min, max, sigma, device=None, precision=torch.float64):
        super(SoftHistogram2Dbsfast, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=device, dtype=precision) + 0.5)

    def forward(self, x, y):
        # torch_xy.unsqueeze(0).unsqueeze(0) - centers2d.unsqueeze(-2)
        x = x - self.centers # bs x 200 x 64
        y = y - self.centers # bs x 200 x 64
        # x = torch.unsqueeze(x, 1).unsqueeze(1) - torch.unsqueeze(self.centers, -2)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        y = torch.sigmoid(self.sigma * (y + self.delta/2)) - torch.sigmoid(self.sigma * (y - self.delta/2))
        x = x.transpose(-2,-1)
        x = torch.bmm(x,y)
        return x, self.min + self.delta*torch.arange(256+1), self.min + self.delta*torch.arange(256+1)
