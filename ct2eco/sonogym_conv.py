"""Convolution-based US simulator, vendored from SonoGym (MIT licence, Copyright (c) 2024 The Isaac Lab Project Developers):
https://github.com/SonoGym/SonoGym  source/spinal_surgery/spinal_surgery/lab/sensors/ultrasound/simulate_US_conv.py @ e67be58

Kept: the `simulate_US_image_given_rand_map` path SonoGym uses at runtime, unchanged except:
- unused methods, the debug print and the pyvista/pydicom/PIL imports are removed;
- `Vl_img` is a per-pixel (n, H, W) field instead of a (n, l, l) grid indexed by [n // fl, h // fl, w // fl]:
  the original indexing makes every frame of a batch with n < fl reuse frame 0's large-scale speckle.
"""

import torch
import torch.nn.functional as F


class USSimulatorConv:
    def __init__(self, us_cfg, device):
        system_params = us_cfg["system_params"]
        self.device = device
        self.f = system_params["frequency"]
        self.I0 = system_params["I0"]
        self.e = system_params["element_size"]
        self.sx_E = system_params["sx_E"]
        self.sy_E = system_params["sy_E"]
        self.sx_B = system_params["sx_B"]
        self.sy_B = system_params["sy_B"]
        self.beta = system_params["TGC_beta"]
        self.beta_edge = system_params["TGC_edge"]
        self.label_to_params_dict = {
            key: torch.tensor([item[k] for k in ("alpha", "z", "mu0", "mu1", "s0", "Al", "fl")], device=device)
            for key, item in us_cfg["label_to_ac_params_dict"].items()
        }
        self.kernel_size = tuple(us_cfg["kernel_size"])
        self.PSF_E = self.compute_PSF_kernel(self.sx_E, self.sy_E)
        self.PSF_B = self.compute_PSF_kernel(self.sx_B, self.sy_B)
        self.E_S_ratio = us_cfg["E_S_ratio"]
        self.if_large_scale_speckle = us_cfg["large_scale_speckle"]

    def compute_PSF_kernel(self, sx, sy):
        center = (torch.tensor(self.kernel_size, device=self.device) - 1) / 2
        x_inds = torch.arange(0, self.kernel_size[0], device=self.device)
        y_inds = torch.arange(0, self.kernel_size[1], device=self.device)
        x_grid, y_grid = torch.meshgrid(x_inds, y_inds, indexing="ij")
        x = x_grid - center[0]
        y = y_grid - center[1]
        density_x = torch.exp(-0.5 * x**2 / sx**2) * torch.cos(2 * torch.pi * self.f * x)
        density_y = torch.exp(-0.5 * y**2 / sy**2)
        return (density_x * density_y).reshape((1, 1) + self.kernel_size)

    def assign_params_map(self, label_img):
        param_map = torch.zeros(label_img.shape + (7,), device=label_img.device)
        for label in torch.unique(label_img).tolist():
            param_map[label_img == label, :] = self.label_to_params_dict[int(label)]
        return param_map

    def compute_attenuation_map(self, alpha_map):
        alpha_l_map = torch.cumsum(alpha_map, dim=1) * self.e
        return torch.exp(-alpha_l_map * self.f)

    def compute_edge_map(self, label_img):
        edge_map = torch.zeros(label_img.shape, device=label_img.device)
        pad_img = F.pad(label_img, (1, 1, 1, 1), "reflect")
        edge_map[~(pad_img[:, :-2, 1:-1] == pad_img[:, 1:-1, 1:-1])] = 1
        return edge_map

    def compute_ct_edge_map(self, ct_img):
        edge_map = torch.zeros(ct_img.shape, device=ct_img.device)
        pad_img = F.pad(ct_img, (1, 1, 1, 1), "reflect")
        label_up = pad_img[:, :-2, 1:-1]
        label_down = pad_img[:, 1:-1, 1:-1]
        edge_map[abs(label_up - label_down) > (label_down * 0.3)] = 1
        return edge_map

    def compute_image_gradient(self, img):
        pad_img = F.pad(img, (1, 1, 1, 1), "reflect")
        grad_x = img - pad_img[:, :-2, 1:-1]
        grad_y = 0.5 * ((img - pad_img[:, 1:-1, :-2]) + (pad_img[:, 1:-1, 2:] - img))
        return torch.stack([grad_x, grad_y], dim=-1)

    def compute_cos_map(self, img_grad):
        return img_grad[:, :, :, 0] / (torch.linalg.norm(img_grad, dim=-1) + 1e-5)

    def simulate_US_image_given_rand_map(self, label_img, T0_img, T1_img, Vl_img, if_ct=False, ct_img=None):
        """label_img, T0_img, T1_img, Vl_img, ct_img: (n, H, W) float, H = depth."""
        I0_map = torch.ones(label_img.shape, device=label_img.device) * self.I0
        params_map = self.assign_params_map(label_img)
        alpha_map = params_map[:, :, :, 0]
        z_map = ct_img * 1e2 if if_ct else params_map[:, :, :, 1]
        T_params_map = params_map[:, :, :, 2:5]

        atten_map = self.compute_attenuation_map(alpha_map - self.beta)
        atten_map_E = self.compute_attenuation_map(alpha_map - self.beta_edge)
        edge_map = self.compute_ct_edge_map(ct_img) if if_ct else self.compute_edge_map(label_img)
        cos_map = self.compute_cos_map(self.compute_image_gradient(edge_map))

        pad_z_map = F.pad(z_map, (1, 1, 1, 1), mode="reflect")
        z2_map = z_map
        z1_map = pad_z_map[:, :-2, 1:-1]

        # reflection term
        E_map = I0_map * atten_map_E
        E_map = E_map * (z1_map - z2_map) ** 2 / (z1_map + z2_map + 1e-5) ** 2
        if not if_ct:
            E_map = E_map * edge_map
            E_map *= cos_map
        E_map = E_map[:, None, :, :]
        E_map[E_map > 0.1] = 0.1
        E_map = F.conv2d(input=E_map, weight=self.PSF_E, stride=1, padding="same")[:, 0, :, :]

        # scatterers
        S_map = T0_img * T_params_map[:, :, :, 2] + T_params_map[:, :, :, 0]
        S_map[~(T1_img <= T_params_map[:, :, :, 1])] = 0
        if self.if_large_scale_speckle:
            S_map = S_map * (1 + params_map[:, :, :, 5] * Vl_img)

        B_map = I0_map * atten_map * F.conv2d(S_map[:, None], self.PSF_B, padding="same")[:, 0, :, :]
        return self.E_S_ratio * E_map + B_map
