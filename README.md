# Wavefunction Analysis

A Python package for analyzing quantum chemical wavefunctions, built on top of [PySCF](https://github.com/pyscf/pyscf) and model Hamiltonian methods. This package is currently **under active development** and provides advanced tools for investigating electronic structure and dynamics. Key features include:

*   **Electronic Structure**:
    *   **MRSF DFT**: Mixed-Reference Spin-Flip DFT for robust description of ground and excited states, including conical intersections.
    *   **Polariton Chemistry**: Investigating light-matter interactions in cavity QED settings.
*   **Exciton and Molecular Dynamics**: Simulating time-dependent processes to understand energy transfer and structural evolution.
*   **Embedding Theory**: Utilizing Density Matrix Embedding Theory (DMET) and localized orbitals (e.g., Pipek-Mezey) to treat strong correlation in large systems.
*   **Molecular Property Analysis**: Calculating response properties, EPR parameters, intermolecular interactions (SAPT), and Energy Density.
*   **Spin Models**: Studying strongly correlated spin systems (Ising, Heisenberg) using Matrix Product States (DMRG), Monte Carlo simulations, and Quantum Computing algorithms (QAOA).

For more details and API references, please visit the [wavefunction_analysis](https://zheng-pei-c.github.io/wavefunction_analysis/) webpage.



## Installation

Currently, this package does not have a `setup.py`. To use it, you need to add the package directory to your `PYTHONPATH`.

```bash
git clone https://github.com/Zheng-Pei-c/wavefunction_analysis.git
export PYTHONPATH=$PYTHONPATH:/path/to/wavefunction_analysis
```



## Dependencies

-   [Python 3.x](https://www.python.org/)
-   [PySCF](https://github.com/pyscf/pyscf)
-   [numpy](https://numpy.org/)
-   [scipy](https://scipy.org/)
-   [opt_einsum](https://github.com/dgasmith/opt_einsum)
-   [itertools](https://docs.python.org/3/library/itertools.html)
-   [Cirq](https://quantumai.google/cirq)
-   [SymPy](https://www.sympy.org/)



## Usage Examples



### 1. Unit Conversion

The package provides a utility to convert between various physical units (time, length, energy, frequency, temperature, mass). See [`wavefunction_analysis/utils/unit_conversion.py`](https://github.com/Zheng-Pei-c/wavefunction_analysis/blob/main/wavefunction_analysis/utils/unit_conversion.py).

```python
from wavefunction_analysis.utils import convert_units

# Convert 100 fs to atomic units
t_au = convert_units(100, 'fs', 'au')
print(f"100 fs = {t_au} au")

# Convert 0.1 Hartree to eV
e_ev = convert_units(0.1, 'hartree', 'ev')
print(f"0.1 Hartree = {e_ev} eV")
```



### 2. DMET Analysis

Here is an example of how to perform Density Matrix Embedding Theory (DMET) analysis on a water cluster (adapted from [`wavefunction_analysis/embedding/fragment_entangle.py`](https://github.com/Zheng-Pei-c/wavefunction_analysis/blob/main/wavefunction_analysis/embedding/fragment_entangle.py)).

```python
from pyscf import gto, scf
from wavefunction_analysis.embedding.fragment_entangle import get_embedding_system

# Define molecule (Water trimer example)
mol = gto.Mole()
mol.build(
    atom = """
       O         0.4183272099    0.1671038379    0.1010361156
       H         0.8784893276   -0.0368266484    0.9330933285
       H        -0.3195928737    0.7774121014    0.3045311682
       O         3.0208058979    0.6163509592   -0.7203724735
       H         3.3050376617    1.4762564664   -1.0295977027
       H         2.0477791789    0.6319690134   -0.7090745711
       O         2.5143150551   -0.2441947452    1.8660305097
       H         2.8954132119   -1.0661605274    2.1741344071
       H         3.0247679096    0.0221180670    1.0833062723
    """,
    basis = '6-311++g**',
    verbose=0
)

# Define fragments (atom indices)
frgm_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Run SCF
mf = scf.RHF(mol)
mf.kernel()

# Perform DMET analysis
# Calculate embedding energy for the whole system from embedded fragments
get_embedding_system(mf, frgm_idx)
```



### 3. Molecular Dynamics

You can run molecular dynamics simulations using the `dynamics` module. See [`wavefunction_analysis/dynamics/molecular_dynamics.py`](https://github.com/Zheng-Pei-c/wavefunction_analysis/blob/main/wavefunction_analysis/dynamics/molecular_dynamics.py) for more details.

```python
from wavefunction_analysis.dynamics.molecular_dynamics import MolecularDynamics

# Setup MD parameters
key = {
    'atmsym': ['H', 'He'],
    'coordinate': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]],
    'velocity': [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]],
    'dt': 10,
    'total_time': 1000, # au
    'update_method': 'velocity_verlet'
}

# Setup Electronic Dynamics parameters
ed_key = {
    'functional': 'hf',
    'basis': 'sto-3g',
    'charge': 1,
    'ortho_method': 'lowdin'
}

# Initialize and run
md = MolecularDynamics(key, ed_key)
md.run_dynamics()
md.plot_time_variables(fig_name='md_trajectory.png')
```



## Modules

The package contains the following modules:

-   **`dynamics`**: Tools for simulating exciton dynamics and other time-dependent processes.
-   **`embedding`**: Methods for embedding calculations.
-   **`opt`**: Optimization routines.
-   **`plot`**: Visualization tools for plotting results.
-   **`polariton`**: Analysis of polaritonic systems.
-   **`property`**: Calculation of molecular properties (e.g., EPR parameters, SAPT dispersion).
-   **`spins`**: Spin-related analysis.

-   **`utils`**: General utility functions and PySCF parsers.



## TODO

-   [ ] **Spins Module**: Method developments for the model Hamiltonian of spin systems (Monte Carlo, DMRG, Quantum Computing).
-   [ ] **Finite Temperature**: Add finite temperature support to electronic structure methods, dynamics, and spin models.
-   [ ] **Transport Module**: Implement `transport` module with system-bath interaction.
-   [ ] **Periodic Systems**: Add support for periodic system calculations.
-   [ ] **Grassmann Manifold Optimization**: Implement optimization algorithms on Grassmann manifolds for electronic states in `opt` module.
-   [ ] **Polariton Module**: Merge polariton to electronic structure module.
-   [ ] **Documentation**: Expand documentation for all modules.
-   [ ] **Tests**: Add comprehensive unit tests.



## Acknowledgements

This project utilizes AI-assisted coding tools, including [GitHub Copilot](https://github.com/github/copilot.vim) and [Google Gemini](https://gemini.google.com/), for code generation and documentation.



## License

[MIT License](LICENSE)
