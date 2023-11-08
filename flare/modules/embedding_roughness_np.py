# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flare.modules import math_np as math
import torch
import numpy as np


def reflect(viewdirs, normals):
    """Reflect view directions about normals.
    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.
    Args:
      viewdirs: [..., 3] array of view directions.
      normals: [..., 3] array of normal directions (assumed to be unit vectors).
    Returns:
      [..., 3] array of reflection directions.
    """
    return 2.0 * torch.sum(
        normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs


def l2_normalize(x, eps=torch.tensor(torch.finfo(torch.float32).eps)):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, dim=-1, keepdims=True), eps))


def compute_weighted_mae(weights, normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = torch.tensor(1 - torch.finfo(torch.float32).eps)
    return (weights * torch.arccos(
        torch.clip((normals * normals_gt).sum(-1), -one_eps,
                   one_eps))).sum() / weights.sum() * 180.0 / torch.pi


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.
    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).
    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def generate_ide_fn(deg_view, device):
    """Generate integrated directional encoding (IDE) function.
    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.
    Args:
      deg_view: number of spherical harmonics degrees to use.
    Returns:
      A function for evaluating integrated directional encoding.
    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        print('WARNING: Only deg_view of at most 5 is numerically stable.')
    #   raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)
    mat = mat.to(device)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).
        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
          An array with the resulting IDE.
        """
        # expects 1/roughness
        kappa_inv = 1. / (kappa_inv + 10e-20)
        # kappa_inv = roughness
   
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat(
            [(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = torch.tensor(0.5 * ml_array[1, :] * (ml_array[1, :] + 1), dtype=torch.float32).to(device)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn


def generate_dir_enc_fn(deg_view):
    """Generate directional encoding (DE) function.
    Args:
      deg_view: number of spherical harmonics degrees to use.
    Returns:
      A function for evaluating directional encoding.
    """
    integrated_dir_enc_fn = generate_ide_fn(deg_view)

    def dir_enc_fn(xyz):
        """Function returning directional encoding (DE)."""
        return integrated_dir_enc_fn(xyz, torch.zeros_like(xyz[..., :1]))

    return dir_enc_fn