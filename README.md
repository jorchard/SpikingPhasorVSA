# SpikingPhasorVSA

This code demonstrates the capabilities of spiking phasors for implementing Vector Symbolic Algebras (VSAs), as outlined in the paper,

> Orchard J, Furlong PM, Simone K, "Efficient Hyperdimensional Computing with Spiking Phasors", *Neural Computation*, (in press), 2024.

The following jupyter notebooks contain demonstrations.
- `SP_VSA_Demos.ipynb`: Simple phase addition (binding), phase subtraction (unbinding), and phase multiplication (fractional binding)
- `SP_Spatial_Memory.ipynb`: encoding different object in different locations, Fig. 6
- `SP_Functions.ipynb`: encoding functions, Fig. 8
- `SP_integrator.ipynb`: integrating an input signal, Fig. 9
- `SP_LDN.ipynb`: Legendre Delay Network, encoding the history of an input signal

This code uses:
`brian2`, `numpy`, `matplotlib`, `tqdm`

For questions, contact <jorchard@uwaterloo.ca>.
