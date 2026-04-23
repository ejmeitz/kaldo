from dataclasses import dataclass
import numpy as np
import ase.units as units
from typing import Callable, TypeVar
from kaldo.util import planck

T = TypeVar('T')

@dataclass(frozen=True)
class CanonicalConfiguration:
    """One configuration sampled from the harmonic canonical ensemble. """
    displacements: np.ndarray  # shape: (n_modes,) --> (ux1, uy1, uz1, ux2, uy2, uz2, ...)
    random_numbers: np.ndarray  # shape: (n_modes,)


class CanonicalConfigurations:

    def __init__(
        self,
        temperature : float,
        quantum : bool = False,
        rng = np.random.default_rng()
        #TODO HOW TO GET SUPERCELL IFCs or freqs + e-vecs
    ):
        self.temperature = temperature
        self.quantum = quantum
        self.rng = rng

        self.n_modes = ???

        #! NEED TO PASS SUPERCELL FREQUENCIES, EIGENVECTORS, MASSES
        #! EVECS SHOULD BE THE ROWS
        #! NEED TO BE CAREFUL OF UNITS TOO
        self.prefactors = self._prepare()
        self.randn_storage = np.zeros(self.n_modes)
        self.tmp_storage = np.zeros((self.n_modes, self.n_modes))
    
    def _quantum_amplitude(self, frequency : float, mass : float) -> float:
        n = planck(self.temperature, frequency)
        return np.sqrt((units._hbar * (2*n + 1)) / (2 * mass * frequency))

    def _classical_amplitude(self, frequency : float, mass : float) -> float:
        return np.sqrt((units._k * self.temperature) / mass) / frequency

    def _amplitude(self, frequencies : np.ndarray[float], masses : np.ndarray[float]) -> float:
        out = np.zeros_like(frequencies, dtype=float)

        #TODO FREQ_TOL IS DEFINED IN TDEP BUT KALDO DOESNT HAVE THIS CONCEPT
        valid = np.abs(frequencies) >= freq_tol

        if self.quantum:
            out[valid] = self._quantum_amplitude(frequencies[valid], masses[valid])
        else:
            out[valid] = self._classical_amplitude(frequencies[valid], masses[valid])

        return out

    def _prepare(
        self,
        frequencies : np.ndarray[float], # (n_modes,)
        masses : np.ndarray[float], # (n_modes,)
        eigenvectors : np.ndarray[float], # (n_modes, n_modes)
        dim : int = 3
        ) -> np.ndarray[float]:

        # Check for imaginary modes
        if np.amin(frequencies) < 0:
            raise ValueError("Imaginary modes detected, cannot generate canonical configurations")

        # Extend masses from N_atoms to N_modes
        masses_extended = np.repeat(np.asarray(masses, dtype=float), int(dim))
        # Update dimensions so broadcasting works properly
        frequencies = frequencies[:, None] # shape: (N_modes, 1)
        masses_extended = masses_extended[None, :] # shape: (1, N_modes)
        # Amplitude of each mode, dimensions give N_modes x N_modes result
        # [[w1*m1, w1*m2, w1*m3, ...], [w2*m1, w2*m2, w2*m3, ...], ...]
        mean_amplitude_matrix = self._amplitude(frequencies, masses_extended)
        # Pre-scale modes by their average amplitudes, evecs must be rows
        prefactors = np.multiply(eigenvectors, mean_amplitude_matrix) 

        return prefactors

    def __len__(self) -> int:
        return self.n_configs

    def _get_displacements(self) -> np.ndarray[float]:
        self.rng.standard_normal(out = self.randn_storage) # Need to check if this is thread safe
        np.copyto(self.tmp_storage, self.prefactors)
        self.tmp_storage *= self.randn_storage # Scale average mode amplitudes by random numbers
        return np.sum(self.tmp_storage, axis = 0)
        
    def get_configuration(self) -> CanonicalConfiguration:  
        displacements = self._get_displacements()
        return CanonicalConfiguration(displacements, np.copy(self.randn_storage))

    def apply_to_configuration(self, f : Callable[[np.ndarray, np.ndarray], T]) -> T:
        displacements = self._get_displacements()
        return f(displacements, self.randn_storage)