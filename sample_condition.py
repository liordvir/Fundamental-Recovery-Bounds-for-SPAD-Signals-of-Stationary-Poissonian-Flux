from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
import numpy as np
from tqdm import tqdm

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import  normalize_np
from torchvision.transforms.functional import rgb_to_grayscale
import lpips
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
import random
import sys
import io
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def calculate_psnr(img1, img2):
    """
    Calculates PSNR between two grayscale images (assuming range [0, 1]).
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    if img1.max() == img1.min():
        img1_norm = img1
    else:
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
    if img2.max() == img2.min():
        img2_norm = img2
    else:
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())
    mse = np.mean((img1_norm - img2_norm) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def to_lpips_tensor(img_np):
    """
    Helper: Converts a (H, W) numpy array [0, 1] to a (1, 3, H, W) tensor [-1, 1]
    """
    t = torch.from_numpy(img_np).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)

    # Repeat grayscale to 3 channels and normalize to [-1, 1]
    t = t.repeat(1, 3, 1, 1)
    t = (t * 2) - 1
    return t

def set_seed(seed=42):
    # Python's built-in random
    random.seed(seed)
    # Environment variable for some underlying C++ operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Crucial for deterministic GPU behavior (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def ml_reconstruction_hybrid(N, bin_times, B, tau_s):
    """
    Vectorized MLE reconstruction for an entire image grid.
    N, sum_t_tilde are numpy arrays.
    """
    # Denominator: (B - N) * tau_s + sum(t_i)
    total_effective_time = (B - N) * tau_s + bin_times

    # Use a mask or epsilon to avoid division by zero errors
    # This ensures that pixels with no photons simply result in 0 intensity
    lambda_hat = np.divide(N, total_effective_time,
                           out=np.zeros_like(N, dtype=float),
                           where=total_effective_time != 0)

    return lambda_hat

def ml_reconstruction(events, ref_img, T, tau_d, q):
    """
    MLE Reconstruction robust to both NumPy and Tensor inputs.
    Calculates Phi = N / (q * Active_Time)
    """
    count_map, last_t_map = events

    # 0. Robust Conversion: Ensure all inputs are Tensors on the same device as ref_img
    device = ref_img.device
    dtype = ref_img.dtype

    # helper to convert anything to a tensor on the right device
    to_tensor = lambda x: torch.as_tensor(x, device=device, dtype=dtype)

    count_map = to_tensor(count_map)
    last_t_map = to_tensor(last_t_map)
    T_tensor = to_tensor(T)
    tau_d_tensor = to_tensor(tau_d)
    q_tensor = to_tensor(q)

    # 1. Total time spent in "dead state"
    total_dead_time = count_map * tau_d_tensor

    # 2. Total active observation time (Denominator)
    # logic: if the last photon's dead time extends past T, the observation
    # window effectively ends at (last_t + tau_d).
    is_dead_at_end = (last_t_map + tau_d_tensor) > T_tensor

    # Now all arguments to torch.where are Tensors on the same device
    effective_T = torch.where(is_dead_at_end, last_t_map + tau_d_tensor, T_tensor)

    active_time = effective_T - total_dead_time

    # 3. Apply MLE Formula: Phi = N / (q * Active_Time)
    # Using a small epsilon to avoid division by zero
    reconstruction = count_map / (q_tensor * active_time + 1e-12)

    # Clean up any potential numerical instabilities
    reconstruction = torch.nan_to_num(reconstruction, nan=0.0, posinf=0.0, neginf=0.0)

    return reconstruction

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def create_master_svg(data, output_svg="overall_reconstruction.svg", clip_percentile=None):
    """
    Args:
        data: Dict containing 'gt', 'spad', and 'methods'
        output_svg: filename
        clip_percentile: If float (e.g., 99.9), clips the top X percentile of
                         pixel values to remove outliers.
    """
    method_names = ["ML", "Discrete", "Temporal", "Hybrid"]
    cols = ["Ground Truth", "SPAD Measurements", "Events Histogram",
            "Maximum Likelihood", "Discrete Mode", "Temporal Mode", "Hybrid Mode"]
    gamma = 1
    num_cols = len(cols)
    num_img = len(data["methods"]["ML"])

    fig, axes = plt.subplots(num_img, num_cols, figsize=(3 * num_cols, 3 * num_img), squeeze=False)

    for i in range(num_img):
        # --- Helper: Handles Outliers and Normalization ---
        def get_clean_img(arr, is_spad=False):
            if hasattr(arr, 'detach'): arr = arr.detach().cpu().numpy()
            img_data = arr.copy().astype(np.float32)

            # 1. Percentile Clipping (Handling Outliers)
            if clip_percentile is not None:
                v_max = np.percentile(img_data, clip_percentile)
                img_data = np.clip(img_data, a_min=None, a_max=v_max)

            # 2. Normalization for display
            if img_data.max() > 0:
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

            # Save to buffer as PNG to "flatten" the data for SVG compatibility
            buf = io.BytesIO()
            plt.imsave(buf, img_data, cmap='gray', format='png')
            buf.seek(0)
            return plt.imread(buf)

        # 1. Ground Truth
        axes[i, 0].imshow(get_clean_img(data["gt"][i]))

        # 2. SPAD Image
        # Note: we pass the raw spad data to the helper which now handles normalization
        axes[i, 1].imshow(get_clean_img(data["spad"][i], is_spad=True), interpolation='nearest')

        # 3. Histogram Logic
        spad_raw = data["spad"][i]
        if hasattr(spad_raw, 'detach'): spad_raw = spad_raw.detach().cpu().numpy()

        # We still want the histogram to show the true max, or clipped?
        # Usually, clipping the hist x-axis makes it more readable too.
        max_val = spad_raw.max() if clip_percentile is None else np.percentile(spad_raw, clip_percentile)

        if max_val > 50:
            bins = np.logspace(0, np.log10(max_val + 1), num=30)
            bins = np.insert(bins, 0, 0)
        else:
            bins = np.arange(max_val + 2) - 0.5

        axes[i, 2].hist(spad_raw.ravel(), color='black', log=True, bins=bins, rwidth=0.8)

        if max_val > 100:
            axes[i, 2].set_xscale('symlog')
            axes[i, 2].xaxis.set_major_locator(LogLocator(base=10.0))
            axes[i, 2].xaxis.set_major_formatter(ScalarFormatter())
        else:
            axes[i, 2].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        axes[i, 2].set_box_aspect(1)
        axes[i, 2].tick_params(labelsize=14, which='both')
        axes[i, 2].set_yticks([])
        axes[i, 2].set_ylim(bottom=1)
        if clip_percentile:
            axes[i, 2].set_xlim(right=max_val * 1.1)  # Keep the hist focused on non-outliers

        # 4-6. Reconstruction Methods
        for m_idx, m_name in enumerate(method_names):
            col_idx = m_idx + 3
            recon_img = get_clean_img(data["methods"][m_name][i]["image"])
            axes[i, col_idx].imshow(recon_img)

        # Labels and formatting
        for j in range(num_cols):
            if i == 0:
                axes[i, j].set_title(cols[j], fontweight='bold', fontsize=16)
            if j != 2:
                axes[i, j].set_axis_off()  # cleaner than setting ticks to []

    plt.tight_layout()
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    # Make deterministic
    set_seed(42)

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Do Inference
    gamma = 1
    recon_methods = ['ML', 'Hybrid', "Discrete", "Temporal"]
    for recon_method in recon_methods:
        os.makedirs(os.path.join(out_path, 'recon', recon_method), exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    fid_trackers = {}
    with open(os.devnull, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            # Initialize LPIPS model
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            # Initialize FID model metric
            for method in recon_methods:
                fid_trackers[method] = FrechetInceptionDistance(feature=2048).to(device)
        finally:
            sys.stdout = old_stdout

    all_results = {
        "gt": [],
        "spad": [],
        "methods": {
            "Temporal": [],
            "Discrete": [],
            "ML": [],
            "Hybrid": []
        },
        "fid": {
            "Temporal": [],
            "Discrete": [],
            "ML": [],
            "Hybrid": []
        }
    }
    config_string = str(int(operator.ref_lux)) + 'lux_' + str(int(operator.T * 1e9)) + 'ns_' + str(
        int(operator.tau_d * 1e9)) + 'ns'
    for i, ref_img in enumerate(loader):
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        ref_img = rgb_to_grayscale(ref_img)

        # normalize image to [0,1] range
        ref_img = (ref_img + 1) / 2.0 # + 0.1
        all_results["gt"].append(to_numpy(ref_img.squeeze()))
        real_data=False

        events_count_temporal, last_event_times, _ = operator.forward(ref_img, sim_type='Temporal')
        events_count_hybrid, _, sum_timestamps = operator.forward(ref_img, sim_type='Hybrid')

        all_results["spad"].append(to_numpy(events_count_temporal))

        t_ref, fid_ref_uint8 = None, None

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        real_alpha = None
        for recond_idx, recon_method in enumerate(recon_methods):
            tqdm.write('Image ' + str(i) + ': ' + recon_method + ' method...')
            if recon_method == "ML":
                if real_data:
                    B = operator.T /(operator.tau_d + operator.tau_s)
                    sample = ml_reconstruction_hybrid(events_count_temporal.cpu().numpy(), sum_timestamps.cpu().numpy(), B, operator.tau_s)
                    real_alpha = sample.max()
                else:
                    sample = ml_reconstruction((events_count_temporal, last_event_times), ref_img, operator.T, operator.tau_d, operator.q)
                sample_norm = normalize_np(sample)

            elif recon_method == "Discrete":
                sample = sample_fn(x_start=x_start, measurement=(events_count_hybrid),
                                   record=True, save_root=out_path, operator=operator, recon_method=recon_method, real_alpha=real_alpha)
                if sample.shape[1] == 3:
                    sample = rgb_to_grayscale(sample.squeeze()).squeeze()
                sample = sample.detach().cpu().squeeze().numpy()
                sample_norm = normalize_np(sample, gamma=gamma)
            elif recon_method == "Hybrid":
                sample = sample_fn(x_start=x_start, measurement=(events_count_hybrid, sum_timestamps),
                                   record=True, save_root=out_path, operator=operator, recon_method=recon_method, real_alpha=real_alpha)
                if sample.shape[1] == 3:
                    sample = rgb_to_grayscale(sample.squeeze()).squeeze()
                sample = sample.detach().cpu().squeeze().numpy()
                sample_norm = normalize_np(sample)
            else:
                sample = sample_fn(x_start=x_start, measurement=(events_count_temporal, last_event_times),
                                   record=True, save_root=out_path, operator=operator, recon_method=recon_method, real_alpha=real_alpha)
                if sample.shape[1] == 3:
                    sample = rgb_to_grayscale(sample.squeeze()).squeeze()
                sample = sample.detach().cpu().squeeze().numpy()
                sample_norm = normalize_np(sample)

            # Save results
            data_to_save = events_count_hybrid.detach().cpu().numpy() if torch.is_tensor(events_count_hybrid) else events_count_hybrid

            recon_save_data = sample_norm.detach().cpu().numpy() if torch.is_tensor(sample_norm) else sample_norm

            plt.imsave(os.path.join(out_path, 'input', fname), data_to_save, cmap='gray')
            plt.imsave(os.path.join(out_path, 'label', fname), ref_img.detach().cpu().squeeze().numpy(), cmap='gray')
            plt.imsave(os.path.join(out_path, 'recon', recon_method, fname), recon_save_data, cmap='gray')

            ref_show = normalize_np(ref_img.squeeze().squeeze()).detach().cpu().numpy()
            if torch.is_tensor(sample_norm):
                recon_show = sample_norm.clone().detach().cpu().numpy()
            else:
                recon_show = sample_norm

            # calculate metrics
            psnr_val = calculate_psnr(ref_show, recon_show)
            ssim_val = ssim(ref_show, recon_show, data_range=1.0)
            t_recon = torch.from_numpy(recon_show).float().to(device)
            if t_recon.ndim == 2: t_recon = t_recon.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            t_recon = t_recon * 2 - 1
            t_ref = torch.from_numpy(ref_show).float().to(device)
            if t_ref.ndim == 2: t_ref = t_ref.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            t_ref = t_ref * 2 - 1
            with torch.no_grad():
                lpips_val = loss_fn_vgg(t_ref, t_recon).item()


            all_results["methods"][recon_method].append({
                "image": to_numpy(recon_show),
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val
            })
            print('PSNR: ' + str(psnr_val) + '\nSSIM: ' + str(ssim_val) + '\nLPIPS: ' + str(lpips_val) + '\n')


    svg_save_path = "results/spad/recon/" + config_string + '.svg'
    create_master_svg(all_results, svg_save_path, clip_percentile=99)

if __name__ == '__main__':
    main()
