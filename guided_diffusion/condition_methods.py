from abc import ABC, abstractmethod
import torch
import math
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='spad')
class PosteriorSamplingSpad(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.T_bin = torch.tensor(2e-6, device='cuda:0', dtype=torch.float32)
        self.B = torch.ceil(torch.tensor(operator.T, device='cuda:0') / 2e-6)

    def grad_timeseries(self, x_0_hat, measurement, alpha, beta):
        device = x_0_hat.device

        # 1. Unpack and ensure tensors
        if isinstance(measurement, tuple):
            # Expecting (count_map, last_t_map)
            counts_img, t_last_img = torch.as_tensor(measurement[0], device=device).float(), torch.as_tensor(measurement[1], device=device).float()#[torch.as_tensor(m, device=device).float() for m in measurement]
            q = self.operator.q
            tau_d = self.operator.tau_d
            t_exp = self.operator.T
            dcr = self.operator.DCR
        else:
            # Fallback if passed as a single stacked tensor [2, H, W]
            counts_img = torch.as_tensor(measurement["events_count"], device=device).float()
            t_last_img = torch.as_tensor(measurement["t_N_map"], device=device).float()
            q = 1 #torch.as_tensor(measurement["q_map"], device=device).float() #self.operator.q
            tau_d = torch.as_tensor(measurement["tau_d_map"], device=device).float()
            t_exp = torch.as_tensor(measurement["T_map"], device=device).float()
            dcr = self.operator.DCR

        # 2. Setup dimensions
        _, _, H, W = x_0_hat.shape
        Phi = x_0_hat.view(H, W)  # Work with 2D view for easy mapping

        # 3. Calculate Effective Duration
        # For every pixel, determine if the sensor was in dead-time when the exposure ended
        is_dead_at_end = t_exp < (t_last_img + tau_d)

        # If dead at end, observation window is effectively [0, t_last + tau_d]
        # Otherwise, it's the full t_exp
        effective_duration = torch.where(is_dead_at_end, t_last_img + tau_d, torch.tensor(t_exp, device=device))

        # 4. Apply the Formula (Vectorized over the whole image)
        # Denominator: q * alpha * (Phi + beta) + DCR
        # This represents the expected rate of arrivals
        denom = q * alpha * (Phi + beta) + dcr

        # Term 1: The "Gain" from observed counts
        # term1 = N * (d/dPhi log(expected_rate))
        term1 = (counts_img * q * alpha) / denom

        # Term 2: The "Loss" from total observation time
        # Note: Every count N subtracts one tau_d from the sensitive time
        active_time = effective_duration - (counts_img * tau_d)
        term2 = alpha * q * active_time

        # 5. Final Gradient Assembly
        # For pixels with 0 counts: counts_img=0 and t_last=0.
        # The formula naturally collapses to: 0 - (alpha * q * t_exp),
        # which is the correct penalty for seeing nothing.
        result_grid = term1 - term2

        return result_grid  # Returns (H, W) tensor

    def grad_binomial(self, x_0_hat, measurement, alpha, beta):
        """
            Adjusted Binomial Gradient for the Gated Model.
            Each bin = Active Gate (Tb) + Dead Time (tau_d).
            Total bin period = 2 * tau_d.
            """
        device = x_0_hat.device

        # 1. Unpack maps
        if isinstance(measurement, (list, tuple)):
            counts_img = torch.as_tensor(measurement[0], device=device).float()
        else:
            counts_img = torch.as_tensor(measurement, device=device).float()

        q = self.operator.q
        tau_d = self.operator.tau_d
        tau_s = self.operator.tau_s
        t_exp = self.operator.T
        dcr = self.operator.DCR

        # 2. Setup dimensions
        _, _, H, W = x_0_hat.shape
        Phi = x_0_hat.view(H, W)

        # 3. Parameters for the Gated Model
        total_bin_period = tau_s+tau_d #2 * tau_d  # Total time per bin (including dead time)
        B = t_exp / total_bin_period  # Total number of discrete bins

        # Rate lambda = alpha * q * (Phi + beta) + DCR
        rate = alpha * q * (Phi + beta) + dcr

        # 4. Apply the Gated Binomial Gradient Formula (Eq 10)
        # Gradient w.r.t Lambda:
        # Term 1: (N * Tb) / (B * (exp(rate * Tb) - 1))
        # Term 2: Tb * (B - N)

        # Pre-calculate exp(lambda * Tb) - 1
        exp_term = torch.exp(rate * tau_s) - 1
        exp_term = torch.clamp(exp_term, min=1e-12)  # Numerical stability

        term1 = (counts_img * tau_s) / exp_term
        term2 = tau_s * (B - counts_img)

        grad_wrt_lambda = term1 - term2

        # 5. Chain Rule: Multiply by d_lambda / d_Phi
        # lambda = alpha * q * Phi + constant -> derivative is alpha * q
        result_grid = grad_wrt_lambda * (alpha * q)

        return result_grid

    def compute_hybrid_grad(self, measurement, x_0_hat, alpha, beta):
        device = x_0_hat.device

        # 1. Unpack Measurements
        N = torch.as_tensor(measurement[0], device=device).float()
        # sum_t is now the sum of relative timestamps within bins
        sum_t = torch.as_tensor(measurement[1], device=device).float()

        # 2. Operator Constants
        q = self.operator.q
        tau_d = self.operator.tau_d
        tau_s = self.operator.tau_s
        T = self.operator.T
        dcr = self.operator.DCR

        # Model specific constants
        Tb = tau_s
        total_bin_period = tau_s + tau_d
        B = T / total_bin_period

        # 3. Calculate Lambda (Rate)
        _, _, H, W = x_0_hat.shape
        Phi = x_0_hat.view(H, W)
        lmb = q * alpha * (Phi + beta) + dcr
        lmb = torch.clamp(lmb, min=1e-12)

        # 4. Calculate the PDF Term for Relative Timestamps
        # When t_i is relative to the bin start, the 'absolute time' offset
        # (N-1)*T type terms drop out.
        # We only care about the expected arrival time within a single bin width (Tb).
        # For a Poisson process truncated to the first event in a bin of width Tb:
        # The term usually derived is related to the probability of NO events
        # occurring before the detection.

        # New offset logic:
        # If the paper follows the standard first-photon arrival MLE:
        # grad = N/lmb - \sum(t_i) - (B - N) * Tb
        # where (B-N) is the number of bins where NO photon was detected.

        idle_bins_contribution = (B - N) * Tb

        # 5. Hybrid Gradient w.r.t Lambda
        # The gradient is the score function: (Observed - Expected)
        # sum_t is the arrivals, idle_bins_contribution is the censored time
        grad_wrt_lmb = (N / lmb) - (sum_t + idle_bins_contribution)

        # 6. Chain Rule to Image Space
        result_grid = grad_wrt_lmb * (q * alpha)

        return result_grid

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):

        # convert pixel space to photon flux
        ref_lux = self.operator.ref_lux
        pixel_area = self.operator.pixel_pitch_m**2 * self.operator.fill_factor
        lum_eff = self.operator.lum_eff
        E_p = self.operator.c * self.operator.h / self.operator.wavelength_m
        beta = 1
        real_alpha = kwargs.get('real_alpha')
        if real_alpha is not None:
            alpha = 0.5 * real_alpha / self.operator.q
        else:
            alpha = 0.5 * (ref_lux * pixel_area / lum_eff) / E_p
        if x_0_hat.shape[1] == 3:
            x_0_hat = rgb_to_grayscale(x_0_hat)

        with torch.no_grad():
            recon_method = kwargs.get('recon_method', 'Temporal')
            if recon_method == 'Temporal':
                grad_spad = self.grad_timeseries(x_0_hat=x_0_hat, measurement=measurement, alpha=alpha, beta=beta)
            elif recon_method == 'Discrete':
                grad_spad = self.grad_binomial(x_0_hat=x_0_hat, measurement=measurement, alpha=alpha, beta=beta)
            elif recon_method == 'Hybrid':
                grad_spad = self.compute_hybrid_grad(x_0_hat=x_0_hat, measurement=measurement, alpha=alpha, beta=beta)
            else:
                raise('Unrecognized reconstruction method')

            grad_median = grad_spad.abs().median()
            threshold = max(grad_median.item() * 10, 1.0)

            # 2. Apply the clip
            grad_spad = grad_spad.clamp(max=threshold)
            _, _, H, W = x_0_hat.shape
            max_grad_val = self.operator.q * alpha / self.operator.DCR

        if grad_spad.shape != x_0_hat.shape:
            grad_spad = grad_spad.view_as(x_0_hat)

        x_0_hat.backward(gradient=grad_spad)

        with torch.no_grad():
            method_scale = 0.0125 #0.0005 #0.00001 #0.0125
            scalar_scale = method_scale*230 / max_grad_val
            x_t += scalar_scale * self.scale * x_prev.grad

        return x_t, grad_spad.abs().mean()

