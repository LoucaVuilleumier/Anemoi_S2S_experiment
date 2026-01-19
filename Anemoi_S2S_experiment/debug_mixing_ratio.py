#!/usr/bin/env python3
"""Quick debug script to test mixing ratio calculation."""

import torch
import math

def compute_r_sur(t2m: torch.Tensor, dp2m: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """Compute water-vapour mixing ratio at the surface (g/kg) with PyTorch."""
    a_pos, b_pos, c_pos = 17.368, 238.83, 6.107
    a_neg, b_neg, c_neg = 17.856, 245.52, 6.108
    d = 622.0  # Direct result in g/kg
    
    e = torch.where(
        t2m >= 0.0,
        c_pos * torch.exp((a_pos * dp2m) / (dp2m + b_pos)),
        c_neg * torch.exp((a_neg * dp2m) / (dp2m + b_neg)),
    )
    e = torch.clamp(e, min=0.0)
    sp_safe = torch.clamp(sp, min=1e-6)
    denom = sp_safe - e
    denom = torch.clamp(denom, min=1e-6)
    r_sur = d * (e / denom)
    
    return r_sur, e, denom

# Test with reasonable values
print("=== Testing with reasonable values ===")
t2m = torch.tensor([20.0])    # 20°C
dp2m = torch.tensor([15.0])   # 15°C dewpoint
sp = torch.tensor([1013.25]) # 1013.25 hPa (standard sea level pressure)

r, e, denom = compute_r_sur(t2m, dp2m, sp)
print(f"T={t2m[0]:.1f}°C, DP={dp2m[0]:.1f}°C, SP={sp[0]:.1f} hPa")
print(f"Vapor pressure e={e[0]:.2f} hPa")
print(f"Denominator (sp-e)={denom[0]:.2f} hPa")
print(f"Mixing ratio r={r[0]:.2f} g/kg")
print()

# Test with problematic values
print("=== Testing with potentially problematic values ===")
t2m = torch.tensor([30.0])    # 30°C
dp2m = torch.tensor([29.0])   # 29°C dewpoint (very high humidity)
sp = torch.tensor([1013.25]) # 1013.25 hPa

r, e, denom = compute_r_sur(t2m, dp2m, sp)
print(f"T={t2m[0]:.1f}°C, DP={dp2m[0]:.1f}°C, SP={sp[0]:.1f} hPa")
print(f"Vapor pressure e={e[0]:.2f} hPa")
print(f"Denominator (sp-e)={denom[0]:.2f} hPa")
print(f"Mixing ratio r={r[0]:.2f} g/kg")
print()

# Test with very low surface pressure (might be the issue)
print("=== Testing with low surface pressure ===")
t2m = torch.tensor([20.0])    # 20°C
dp2m = torch.tensor([15.0])   # 15°C dewpoint
sp = torch.tensor([50.0])     # 50 hPa (very low pressure - high altitude)

r, e, denom = compute_r_sur(t2m, dp2m, sp)
print(f"T={t2m[0]:.1f}°C, DP={dp2m[0]:.1f}°C, SP={sp[0]:.1f} hPa")
print(f"Vapor pressure e={e[0]:.2f} hPa")
print(f"Denominator (sp-e)={denom[0]:.2f} hPa")
print(f"Mixing ratio r={r[0]:.2f} g/kg")
print()

# Test with dewpoint higher than temperature (impossible physically)
print("=== Testing with impossible dewpoint > temperature ===")
t2m = torch.tensor([20.0])    # 20°C
dp2m = torch.tensor([25.0])   # 25°C dewpoint (impossible!)
sp = torch.tensor([1013.25]) # 1013.25 hPa

r, e, denom = compute_r_sur(t2m, dp2m, sp)
print(f"T={t2m[0]:.1f}°C, DP={dp2m[0]:.1f}°C, SP={sp[0]:.1f} hPa")
print(f"Vapor pressure e={e[0]:.2f} hPa")
print(f"Denominator (sp-e)={denom[0]:.2f} hPa")
print(f"Mixing ratio r={r[0]:.2f} g/kg")
print()