import numpy as np
import ase.units as units

def planck(temperature: float, frequency: float) -> float:
    
    #! TODO: HOW DO THEY DO PHYSICAL CONSTANTS
    x = (units._hbar * frequency) / (units._k * temperature)
    
    if x > 1e2:
        # Very large x → n ≈ 0 (avoid overflow)
        return 0.0
    else:
        return np.reciprocal(np.expm1(x))
