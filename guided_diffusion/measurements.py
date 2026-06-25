'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from torchvision.transforms.functional import rgb_to_grayscale
from motionblur.motionblur import Kernel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

import numpy as np
from PIL import Image


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name='spad')
class SpadOperator(NonLinearOperator):
    def __init__(self, ref_lux, use_max_N, max_N, pixel_pitch_m, fill_factor, q, tau_d, tau_s, Pap, jitter_sigma, DCR, T, wavelength_m, afterpulse_delay_mean, lum_eff, device):
        self.ref_lux = float(ref_lux)
        self.pixel_pitch_m = float(pixel_pitch_m)
        self.fill_factor = float(fill_factor)
        self.q = float(q)
        self.tau_d = float(tau_d)
        self.tau_s = float(tau_s)
        self.Pap = float(Pap)
        self.jitter_sigma = float(jitter_sigma)
        self.DCR = float(DCR)
        self.T = float(T)
        self.wavelength_m = float(wavelength_m)
        self.afterpulse_delay_mean = float(afterpulse_delay_mean)
        self.lum_eff = float(lum_eff)
        # Physical constants
        self.h = 6.62607015e-34  # Planck's constant [J*s]
        self.c = 299792458.0  # speed of light [m/s]
        self.use_max_N = bool(use_max_N)
        self.max_N = int(max_N)
        self.device = device


    def ml_recon(self, data):
        recon = data / (self.q * (self.T - data * self.tau_d))
        return recon

    def forward(self, data, **kwargs):
        events_count, last_event_times, sum_times = None, None, None
        if self.use_max_N:
            spad_forward = self.simulate_spad_image_by_max_N(data, self.max_N)
        else:
            sim_type = kwargs['sim_type']
            data_in_pixels = bool(kwargs.get('data_in_pixels', True))
            if sim_type == 'Temporal':
                events_count, last_event_times = self.simulate_spad_image_memory_efficient(data, data_in_pixels)
            elif sim_type == 'Discrete':
                events_count, _ = self.simulate_hybrid_updated(data, data_in_pixels)
            elif sim_type == 'Hybrid':
                events_count, sum_times, last_event_times = self.simulate_hybrid_updated(data, data_in_pixels)
            else:
                raise('Simulation type unknown: ' + sim_type)
        return events_count, last_event_times, sum_times

    def power_to_photon_flux(self, I_W_map):
        """
        Equation (5): Phi_bar = I_W / E_p
        where E_p = h * c / lambda.
        Returns photons/sec per pixel (float array).
        """
        E_p = self.h * self.c / self.wavelength_m
        return I_W_map / E_p

    def image_to_lux(self, img_norm):
        """
        Map normalized image (0..1) to lux map, given that maximum (1.0) corresponds to ref_lux.
        img_norm: float array in [0,1]
        """
        return img_norm * self.ref_lux

    def lux_to_power_per_pixel(self, I_lux_map, pixel_area_m2):
        """
        Equation (4): I_W = (A_p / 683 lm/W) * I_lux
        where I_lux is in lux (lumens/m^2), A_p is pixel area in m^2, 683 lm/W is luminous efficacy at 555 nm.
        Returns power on pixel in Watts.
        """
        return (I_lux_map * pixel_area_m2) / 683.0

    def get_sigma_drift(self, tau):
        """
        Calculates sigma_drift based on the Allan Variation zones.
        Hardcoded values for a high-end SPAD clock (e.g., typical TCXO/FPGA PLL).
        """
        if tau <= 0:
            return 0.0

        # Coefficients (Example values for a high-end oscillator)
        N0 = 1e-22  # White FM intensity (Zone I)
        sf = 1e-11  # Flicker Floor (Zone II)
        N2 = 1e-20  # Random Walk FM intensity (Zone III)

        # Zone boundaries (Break points tau_1, tau_2)
        tau_1 = 0.1  # 100ms
        tau_2 = 1.0  # 1s

        if tau < tau_1:
            # Zone I: White FM -> sigma_x^2 = N0 * tau
            var_drift = N0 * tau
        elif tau < tau_2:
            # Zone II: Flicker Floor -> sigma_x^2 = sf^2 * tau^2
            var_drift = (sf ** 2) * (tau ** 2)
        else:
            # Zone III: Random Walk FM -> sigma_x^2 = (1/3) * N2 * tau^3
            var_drift = (1 / 3) * N2 * (tau ** 3)

        return np.sqrt(var_drift)

    # def simulate_pixel_times(self, phi_bar, device='cuda'):
    #     """
    #     Simulate arrival times for a single pixel using PyTorch.
    #     """
    #     # 1. Primary photon counts
    #     mean_primary = phi_bar * self.T
    #     if mean_primary <= 0:
    #         primary_times = torch.empty(0, device=device)
    #     else:
    #         # torch.poisson expects a tensor
    #         Np = torch.poisson(torch.tensor([mean_primary], device=device)).item()
    #         if Np > 0:
    #             primary_times = torch.rand(int(Np), device=device) * self.T
    #         else:
    #             primary_times = torch.empty(0, device=device)
    #
    #     # 2. Quantum efficiency thinning
    #     if primary_times.numel() > 0:
    #         keep_mask = torch.rand(primary_times.shape, device=device) < self.q
    #         primary_times = primary_times[keep_mask]
    #
    #     # 3. Dark counts
    #     Nd = torch.poisson(torch.tensor([self.DCR * self.T], device=device)).item()
    #     dark_times = torch.rand(int(Nd), device=device) * self.T if Nd > 0 else torch.empty(0, device=device)
    #
    #     # 4. Combine and Sort
    #     times = torch.cat((primary_times, dark_times))
    #     if times.numel() == 0:
    #         return torch.empty(0, device=device)
    #
    #     times, _ = torch.sort(times)
    #
    #     # 5. Afterpulsing
    #     if self.Pap > 0:
    #         rand_vals = torch.rand(times.shape, device=device)
    #         mask = rand_vals < self.Pap
    #         n_ap = torch.sum(mask).item()
    #         if n_ap > 0:
    #             # Exponential sampling: -ln(1-U) / lambda. Scale is 1/lambda.
    #             # Or use the built-in:
    #             m = torch.distributions.Exponential(torch.tensor([1.0 / self.afterpulse_delay_mean], device=device))
    #             delays = m.sample((int(n_ap),)).squeeze()
    #
    #             ap_times = times[mask] + delays
    #             ap_times = ap_times[ap_times < self.T]
    #             if ap_times.numel() > 0:
    #                 times = torch.cat((times, ap_times))
    #                 times, _ = torch.sort(times)
    #
    #     # 6. Apply dead time (Sequential Logic)
    #     if self.tau_d is not None and self.tau_d > 0 and times.numel() > 0:
    #         kept = []
    #         last = torch.tensor(-float('inf'), device=device)
    #         # We move to a list for the sequential check
    #         for i in range(times.size(0)):
    #             t = times[i]
    #             if t >= last + self.tau_d:
    #                 kept.append(t)
    #                 last = t
    #         times = torch.stack(kept) if kept else torch.empty(0, device=device)
    #
    #     # 7. Apply timing jitter (Gaussian)
    #     if self.jitter_sigma and self.jitter_sigma > 0 and times.numel() > 0:
    #         jitter = torch.randn(times.shape, device=device) * self.jitter_sigma
    #         times = times + jitter
    #         # Discard events outside [0, T)
    #         times = times[(times >= 0.0) & (times < self.T)]
    #         times, _ = torch.sort(times)
    #
    #     return times

    def simulate_pixel_times(self, phi_bar, rng):
        """
        Simulate arrival times for a single pixel.
        Returns array of detection times (in seconds) relative to exposure start [0, T).
        Steps:
          - Sample N_primary ~ Poisson(phi_bar * T)
          - For each, choose uniform arrival time in [0, T)
          - Quantum efficiency thinning: keep each event with probability q
          - Add dark counts: Nd ~ Poisson(DCR * T), uniform times
          - Combine, then for each detection add afterpulse with prob Pap at delay ~ Exp(mean)
          - Sort times and apply dead-time: keep first then skip any within tau_d of last kept
          - Add timing jitter (Gaussian), discard times outside [0, T)
        """
        # Primary photon counts
        mean_primary = phi_bar * self.T
        if mean_primary <= 0:
            primary_times = np.empty(0)
        else:
            Np = rng.poisson(mean_primary)
            if Np > 0:
                primary_times = rng.random(Np) * self.T
            else:
                primary_times = np.empty(0)

        # Quantum efficiency thinning
        if primary_times.size > 0:
            keep_mask = rng.random(primary_times.size) < self.q
            primary_times = primary_times[keep_mask]

        # Dark counts
        Nd = rng.poisson(self.DCR * self.T)
        dark_times = rng.random(Nd) * self.T if Nd > 0 else np.empty(0)

        # Combine primary and dark
        times = np.concatenate((primary_times, dark_times))
        if times.size == 0:
            return np.empty(0)
        times.sort()

        # Afterpulsing: for each detection, with prob Pap add an extra event at t + delay (exponential)
        # Keep only afterpulses that fall within [0, T)
        if self.Pap > 0:
            rand = rng.random(times.size)
            mask = rand < self.Pap
            n_ap = mask.sum()
            if n_ap > 0:
                delays = rng.exponential(scale=self.afterpulse_delay_mean, size=n_ap)
                ap_times = times[mask] + delays
                ap_times = ap_times[ap_times < self.T]
                if ap_times.size > 0:
                    times = np.concatenate((times, ap_times))
                    times.sort()

        # Apply dead time: keep first, then skip events within tau_d of last kept
        if self.tau_d is not None and self.tau_d > 0 and times.size > 0:
            kept = []
            last = -np.inf
            for t in times:
                if t >= last + self.tau_d:
                    kept.append(t)
                    last = t
            times = np.array(kept, dtype=float)

        # Apply timing jitter (add Gaussian noise)
        if self.jitter_sigma and self.jitter_sigma > 0 and times.size > 0:
            times = times + rng.normal(loc=0.0, scale=self.jitter_sigma, size=times.size)
            # discard events outside [0, T)
            times = times[(times >= 0.0) & (times < self.T)]
            times.sort()

        return times

    def simulate_pixel_times_ideal(self, phi_bar, rng):
        """
        Simulate arrival times for a single pixel with ONLY dead time applied.
        (Bypasses DCR, afterpulsing, and timing jitter).
        """
        # 1. Effective primary photon generation (Rate * T * Quantum Efficiency)
        # Using getattr with a default of 1.0 in case 'q' was removed from init
        q = getattr(self, 'q', 1.0)
        mean_detected = phi_bar * self.T

        if mean_detected <= 0:
            return np.empty(0)

        Np = rng.poisson(mean_detected)
        if Np == 0:
            return np.empty(0)

        # 2. Assign uniform arrival times in [0, T) and sort sequentially
        times = rng.random(Np) * self.T
        times.sort()

        # 3. Apply dead time: keep first, then skip events within tau_d of last kept
        if getattr(self, 'tau_d', 0) > 0:
            kept = []
            last = -np.inf
            for t in times:
                if t >= last + self.tau_d:
                    kept.append(t)
                    last = t
            times = np.array(kept, dtype=float)

        return times

    def simulate_pixel_memory_efficient(self, phi_bar, device='cuda', correlated=True):
        """
        Optimized for Zone III (Large T).
        Fully implemented in PyTorch to run on GPU/CPU.
        """
        # 1. Effective rate calculation
        effective_rate = (phi_bar * self.q) + self.DCR

        if effective_rate <= 0:
            return 0, 0.0, self.T

        # 2. Simulation state
        # Keeping these as Python floats/ints is fine for a single-pixel loop,
        # but we use torch.rand/exponential for the GPU speed.
        current_time = 0.0
        count = 0
        last_t_ideal = 0.0

        # Pre-scale for exponential sampling
        # Exp(rate) = -ln(1-U) / rate
        inv_rate = 1.0 / effective_rate

        # 3. Sequential Arrival Simulation
        while True:
            # Sample next arrival time: delta_t ~ Exp(effective_rate)
            # Using a small batch of 1 here; if T is huge, we could sample in chunks to optimize
            u = torch.rand(1, device=device)
            delta_t = -inv_rate * torch.log(1 - u)

            current_time += delta_t.item()

            if current_time >= self.T:
                break

            # 4. Dead Time & Afterpulsing Logic
            if count == 0 or current_time >= last_t_ideal + self.tau_d:
                count += 1
                last_t_ideal = current_time

                # Handle Afterpulsing
                if self.Pap > 0:
                    if torch.rand(1, device=device).item() < self.Pap:
                        u_ap = torch.rand(1, device=device)
                        ap_delay = -self.afterpulse_delay_mean * torch.log(1 - u_ap)

                        if last_t_ideal + ap_delay.item() < self.T:
                            count += 1
                            last_t_ideal += ap_delay.item()

        # 5. Apply Noise
        # Assuming apply_noise_to_final is updated to handle/return torch tensors
        t_N_recorded, T_recorded = self.apply_noise_to_final(last_t_ideal, device=device, correlated=correlated)

        return count, t_N_recorded, T_recorded

    # def simulate_pixel_memory_efficient(self, phi_bar, rng, correlated=True):
    #     """
    #     Optimized for Zone III (Large T).
    #     Uses inter-arrival times to avoid memory overflow.
    #     """
    #     # Total effective rate (Primary + Dark Counts)
    #     # We apply QE thinning here to the rate to save steps
    #     effective_rate = (phi_bar * self.q) + self.DCR
    #
    #     if effective_rate <= 0:
    #         return 0, 0.0, self.T  # count, last_t, T_recorded (ideal placeholder)
    #
    #     # Simulation state
    #     current_time = 0.0
    #     count = 0
    #     last_t_ideal = 0.0
    #
    #     # 1. Sequential Arrival Simulation
    #     while True:
    #         # Sample next arrival time from an Exponential distribution (Poisson Process)
    #         # delta_t ~ Exp(rate)
    #         delta_t = rng.exponential(1.0 / effective_rate)
    #         current_time += delta_t
    #
    #         if current_time >= self.T:
    #             break
    #
    #         # 2. Dead Time & Afterpulsing Logic
    #         # If this photon is outside the dead time of the last KEPT photon
    #         if count == 0 or current_time >= last_t_ideal + self.tau_d:
    #             count += 1
    #             last_t_ideal = current_time
    #
    #             # Handle Afterpulsing: If it occurs, it essentially "extends" the dead time
    #             if self.Pap > 0 and rng.random() < self.Pap:
    #                 ap_delay = rng.exponential(self.afterpulse_delay_mean)
    #                 # Afterpulse effectively occupies the sensor,
    #                 # so we treat its arrival as a 'last_t' for dead-time purposes
    #                 if last_t_ideal + ap_delay < self.T:
    #                     # Note: We don't increment 'count' for afterpulses if
    #                     # your model treats them as sensor-blinding noise,
    #                     # OR we do if you want them in your 'N'.
    #                     # Assuming you want them counted:
    #                     count += 1
    #                     last_t_ideal += ap_delay
    #
    #     # 3. Apply Jitter and Drift to the FINAL result only
    #     # This avoids calculating noise for millions of filtered-out photons
    #     t_N_recorded, T_recorded = self.apply_noise_to_final(last_t_ideal, rng, correlated)
    #
    #     return count, t_N_recorded, T_recorded

    # def apply_noise_to_final(self, t_N_ideal, rng, correlated):
    #     """
    #     Applies the Zone III noise model to the end results.
    #     """
    #     # Helper to get sigma based on your Allan Zone function
    #     sigma_drift_tN = self.get_sigma_drift(t_N_ideal)
    #     drift_tN = rng.normal(0, sigma_drift_tN) if sigma_drift_tN > 0 else 0.0
    #
    #     jitter_tN = rng.normal(0, self.jitter_sigma)
    #     t_N_recorded = t_N_ideal + jitter_tN + drift_tN if t_N_ideal > 0 else 0.0
    #
    #     jitter_T = rng.normal(0, self.jitter_sigma)
    #
    #     if correlated:
    #         # Correlation logic for T
    #         delta_tau = self.T - t_N_ideal
    #         sigma_drift_delta = self.get_sigma_drift(delta_tau)
    #         drift_delta = rng.normal(0, sigma_drift_delta) if sigma_drift_delta > 0 else 0.0
    #         T_recorded = self.T + jitter_T + (drift_tN + drift_delta)
    #     else:
    #         sigma_drift_T = self.get_sigma_drift(self.T)
    #         T_recorded = self.T + jitter_T + rng.normal(0, sigma_drift_T)
    #
    #     return t_N_recorded, T_recorded


    def apply_noise_to_final(self, t_N_ideal, rng, correlated):
        """
        Applies the Zone III noise model to the end results.
        """
        # Helper to get sigma based on your Allan Zone function
        sigma_drift_tN = self.get_sigma_drift(t_N_ideal)
        drift_tN = rng.normal(0, sigma_drift_tN) if sigma_drift_tN > 0 else 0.0

        jitter_tN = rng.normal(0, self.jitter_sigma)
        t_N_recorded = t_N_ideal + jitter_tN + drift_tN if t_N_ideal > 0 else 0.0

        jitter_T = rng.normal(0, self.jitter_sigma)

        if correlated:
            # Correlation logic for T
            delta_tau = self.T - t_N_ideal
            sigma_drift_delta = self.get_sigma_drift(delta_tau)
            drift_delta = rng.normal(0, sigma_drift_delta) if sigma_drift_delta > 0 else 0.0
            T_recorded = self.T + jitter_T + (drift_tN + drift_delta)
        else:
            sigma_drift_T = self.get_sigma_drift(self.T)
            T_recorded = self.T + jitter_T + rng.normal(0, sigma_drift_T)

        return t_N_recorded, T_recorded

    def simulate_spad_image_memory_efficient(self, data, data_in_pixels=True):
        rng = np.random.default_rng(None)

        if data.shape[1] == 3:
            img = rgb_to_grayscale(data.squeeze()).squeeze()
        else:
            img = data.squeeze()

        h_pix, w_pix = img.shape

        # Initialize buffers for the results
        # count_map: Total events per pixel
        # last_t_map: Timestamp of the final event (0.0 if no events)
        count_map = np.zeros((h_pix, w_pix), dtype=np.int32)
        last_t_map = np.zeros((h_pix, w_pix), dtype=np.float64)

        if data_in_pixels:
            Ap = (self.pixel_pitch_m ** 2) * self.fill_factor
            I_lux_map = self.image_to_lux(img)
            I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)
            phi_map = self.power_to_photon_flux(I_W_map)
            lambda_map = (phi_map * self.q) + self.DCR
        else:
            lambda_map = data.squeeze()

        coords = list(np.ndindex(h_pix, w_pix))

        for y, x in coords:
            phi = float(lambda_map[y, x])
            times = self.simulate_pixel_times_ideal(phi_bar=phi, rng=rng)

            if times.size > 0:
                count_map[y, x] = times.size
                last_t_map[y, x] = times[-1]  # Get the last event timestamp

        return count_map, last_t_map
    #
    # def simulate_spad_image_memory_efficient(self, data, data_in_pixels=True):
    #     # Determine the device from the input data
    #     device = data.device
    #
    #     if data.shape[1] == 3:
    #         # Assuming rgb_to_grayscale is torch-compatible
    #         img = rgb_to_grayscale(data.squeeze()).squeeze()
    #     else:
    #         img = data.squeeze()
    #
    #     h_pix, w_pix = img.shape
    #
    #     # Initialize buffers directly on the GPU
    #     count_map = torch.zeros((h_pix, w_pix), dtype=torch.int32, device=device)
    #     last_t_map = torch.zeros((h_pix, w_pix), dtype=torch.float64, device=device)
    #
    #     if data_in_pixels:
    #         Ap = (self.pixel_pitch_m ** 2) * self.fill_factor
    #         I_lux_map = self.image_to_lux(img)
    #         I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)
    #         phi_map = self.power_to_photon_flux(I_W_map)
    #     else:
    #         phi_map = data.squeeze()
    #
    #     # Loop through pixels
    #     # Note: Using nested loops is slow in Python, but matches your logic.
    #     for y in range(h_pix):
    #         for x in range(w_pix):
    #             phi = phi_map[y, x].item()
    #
    #             # Call your previously updated torch function
    #             times = self.simulate_pixel_times(phi_bar=phi, device=device)
    #
    #             if times.numel() > 0:
    #                 count_map[y, x] = times.numel()
    #                 # Access the last element safely in torch
    #                 last_t_map[y, x] = times[-1]
    #
    #     return count_map, last_t_map

    def simulate_spad_image(self, data):
        """
        High-level simulator for entire image.
        Returns (xs, ys, ts) arrays and optionally saves them.
        """
        rng = np.random.default_rng(None)

        if data.shape[1] == 3:
            img = rgb_to_grayscale(data.squeeze()).squeeze()
        else:
            img = data.squeeze()

        xs_list = []
        ys_list = []
        ts_list = []

        img_norm = img  # 2D
        h_pix, w_pix = img_norm.shape
        # pixel area: pixel_pitch^2 * fill_factor
        Ap = (self.pixel_pitch_m ** 2) * self.fill_factor

        # lux map
        I_lux_map = self.image_to_lux(img_norm)

        # power per pixel (W)
        I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)

        # photon flux (photons/s per pixel)
        phi_map = self.power_to_photon_flux(I_W_map)

        # iterate pixels; since counts are low for low-light, per-pixel loops are fine
        #print('\nSimulating Spad Image...')
        coords = list(np.ndindex(h_pix, w_pix))
        # for y in (range(h_pix)):
        #     for x in range(w_pix):
        for y, x in tqdm(coords, desc="Simulating SPAD Pixels", leave=False, delay=0.1):
            phi = float(phi_map[y, x])
            times = self.simulate_pixel_times(phi_bar=phi, rng=rng)
            if times.size > 0:
                xs_list.append(np.full(times.size, x, dtype=np.int32))
                ys_list.append(np.full(times.size, y, dtype=np.int32))
                ts_list.append(times)

        if len(xs_list) == 0:
            xs = np.empty(0, dtype=np.int32)
            ys = np.empty(0, dtype=np.int32)
            ts = np.empty(0, dtype=np.float64)
        else:
            xs = np.concatenate(xs_list)
            ys = np.concatenate(ys_list)
            ts = np.concatenate(ts_list)

        # Create structured AER array
        if ts.size > 0:
            order = np.argsort(ts)
            xs = xs[order]
            ys = ys[order]
            ts = ts[order]

        return xs, ys, ts

    def project(self, data, measurement, **kwargs):
        return data

    def simulate_hybrid_updated(self, data, data_in_pixels=True):
        """
        Unified Hybrid SPAD simulation fully implemented in PyTorch.
        Returns N_map, sum_t_map, and the absolute timestamp of the last event.
        """
        # 1. Preprocessing
        if data.shape[1] == 3:
            img = rgb_to_grayscale(data.squeeze()).squeeze()
        else:
            img = data.squeeze()

        if not torch.is_tensor(img):
            img = torch.from_numpy(img).to(data.device)

        if data_in_pixels:
            Ap = (self.pixel_pitch_m ** 2) * self.fill_factor
            I_lux_map = self.image_to_lux(img)
            I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)
            phi_map = self.power_to_photon_flux(I_W_map)
            lambda_map = (phi_map * self.q) + self.DCR
        else:
            lambda_map = data.squeeze()

        # 2. Binning Logic
        total_bin_period = self.tau_d + self.tau_s
        B = int(self.T / total_bin_period)

        # 3. Determine N (Event Counts)
        pb_map = 1 - torch.exp(-lambda_map * self.tau_s)
        N_map = torch.binomial(torch.full_like(pb_map, B), pb_map)

        sum_t_map = torch.zeros_like(lambda_map)
        last_t_map = torch.zeros_like(lambda_map)  # New map for last event absolute time

        # 4. Compute Sum of Relative Timestamps and Last Absolute Timestamp
        unique_counts = torch.unique(N_map)

        for n_val in unique_counts:
            n_val_int = int(n_val.item())
            if n_val_int == 0:
                continue

            mask = (N_map == n_val)
            num_pixels = torch.sum(mask).item()
            lmb = lambda_map[mask].unsqueeze(1)

            # Inverse Transform Sampling
            u = torch.rand((num_pixels, n_val_int), device=lambda_map.device)

            term1 = 1.0 / lmb
            term2 = 1 - u * (1 - torch.exp(-lmb * self.tau_s))
            offsets = -term1 * torch.log(term2)

            # --- Calculation for Last Absolute Event ---
            # 1. Sample which bins the N events occurred in without replacement.
            # We sort them to find the index of the 'last' bin.
            # Bins are indexed 0 to B-1.
            all_bins = torch.stack([torch.randperm(B, device=lambda_map.device)[:n_val_int] for _ in range(num_pixels)])
            last_bin_indices, _ = torch.max(all_bins, dim=1)  # Shape: (num_pixels,)

            # 2. Calculate absolute start time of those bins
            # t_start = bin_index * (tau_d + tau_s) + tau_d
            bin_start_times = last_bin_indices.float() * total_bin_period + self.tau_d

            # 3. Pick the last relative offset for each pixel
            # (Note: offsets are i.i.d., so we can just take the last column)
            last_offsets = offsets[:, -1]

            # 4. Final absolute time
            last_t_map[mask] = bin_start_times + last_offsets
            # --------------------------------------------

            sum_t_map[mask] = torch.sum(offsets, dim=1)

        return N_map, sum_t_map, last_t_map

    # def simulate_hybrid_updated(self, data, data_in_pixels=True):
    #     """
    #     Unified Hybrid SPAD simulation fully implemented in PyTorch.
    #     """
    #     # 1. Preprocessing
    #     if data.shape[1] == 3:
    #         img = rgb_to_grayscale(data.squeeze()).squeeze()
    #     else:
    #         img = data.squeeze()
    #
    #     # Ensure we are working with a tensor
    #     if not torch.is_tensor(img):
    #         img = torch.from_numpy(img).to(data.device)
    #
    #     if data_in_pixels:
    #         Ap = (self.pixel_pitch_m ** 2) * self.fill_factor
    #         I_lux_map = self.image_to_lux(img)
    #         I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)
    #         phi_map = self.power_to_photon_flux(I_W_map)
    #         lambda_map = (phi_map * self.q) + self.DCR
    #     else:
    #         lambda_map = data.squeeze()
    #
    #     # 2. Binning Logic
    #     total_bin_period = self.tau_d + self.tau_s
    #     B = int(self.T / total_bin_period)
    #
    #     # 3. Determine N (Event Counts)
    #     # Use torch.exp to stay on GPU
    #     pb_map = 1 - torch.exp(-lambda_map * self.tau_s)
    #
    #     # torch.binomial requires 'count' to be a tensor of the same shape or broadcastable
    #     # We ensure B is a float tensor for the probability calculation
    #     N_map = torch.binomial(torch.full_like(pb_map, B), pb_map)
    #
    #     sum_t_map = torch.zeros_like(lambda_map)
    #
    #     # 4. Compute Sum of Relative Timestamps (Delta t)
    #     unique_counts = torch.unique(N_map)
    #
    #     for n_val in unique_counts:
    #         n_val_int = int(n_val.item())
    #         if n_val_int == 0:
    #             continue
    #
    #         mask = (N_map == n_val)
    #         num_pixels = torch.sum(mask).item()
    #
    #         # Pull out relevant lambdas: Shape (num_pixels, 1)
    #         lmb = lambda_map[mask].unsqueeze(1)
    #
    #         # Inverse Transform Sampling for truncated exponential in [0, tau_s]
    #         # u: Shape (num_pixels, n_val_int)
    #         u = torch.rand((num_pixels, n_val_int), device=lambda_map.device)
    #
    #         # The math remains the same, but using torch functions
    #         # offsets: Shape (num_pixels, n_val_int)
    #         term1 = 1.0 / lmb
    #         term2 = 1 - u * (1 - torch.exp(-lmb * self.tau_s))
    #         offsets = -term1 * torch.log(term2)
    #
    #         # Sum along the events dimension (axis 1) and assign to mask
    #         sum_t_map[mask] = torch.sum(offsets, dim=1)
    #
    #     return N_map, sum_t_map

    # def simulate_hybrid_updated(self, data, data_in_pixels=True):
    #     """
    #     Unified Hybrid SPAD simulation.
    #     - N_map: (H, W) array of event counts (identical to gated model).
    #     - sum_t_map: (H, W) array of the sum of RELATIVE timestamps [0, tau_s].
    #     """
    #     if data.shape[1] == 3:
    #         img = rgb_to_grayscale(data.squeeze()).squeeze()
    #     else:
    #         img = data.squeeze()
    #
    #     if torch.is_tensor(img):
    #         img = img.detach().cpu().numpy()
    #
    #     if data_in_pixels:
    #         Ap = (self.pixel_pitch_m ** 2) * self.fill_factor
    #         I_lux_map = self.image_to_lux(img)
    #         I_W_map = self.lux_to_power_per_pixel(I_lux_map, Ap)
    #         phi_map = self.power_to_photon_flux(I_W_map)
    #         lambda_map = (phi_map * self.q) + self.DCR
    #     else:
    #         lambda_map = data.squeeze()
    #
    #     # 2. Binning Logic
    #     total_bin_period = self.tau_d + self.tau_s
    #     B = int(self.T / total_bin_period)
    #
    #     # 3. Determine N (Event Counts)
    #     # This part is exactly the same as your gated_spad_image function
    #     pb_map = 1 - torch.exp(-lambda_map * self.tau_s)
    #     rng = np.random.default_rng()
    #     N_map = rng.binomial(n=B, p=pb_map)
    #
    #     sum_t_map = np.zeros_like(lambda_map, dtype=np.float64)
    #
    #     # 4. Compute Sum of Relative Timestamps (Delta t)
    #     # Vectorize across pixels with the same event counts
    #     unique_counts = np.unique(N_map)
    #     for n_val in unique_counts:
    #         if n_val == 0:
    #             continue
    #
    #         mask = (N_map == n_val)
    #         num_pixels = np.sum(mask)
    #         lmb = lambda_map[mask][:, np.newaxis]  # Shape (num_pixels, 1)
    #
    #         # Inverse Transform Sampling for truncated exponential in [0, tau_s]
    #         # This gives the relative arrival time within the sensing window
    #         u = rng.uniform(size=(num_pixels, n_val))
    #         offsets = - (1.0 / lmb) * np.log(1 - u * (1 - np.exp(-lmb * self.tau_s)))
    #
    #         # We only return the sum of these relative offsets
    #         sum_t_map[mask] = np.sum(offsets, axis=1)
    #
    #     return N_map, sum_t_map

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model

    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)
