# Multi-Species-Cellular-Automata

Part of my research undertaken when interning in the Institute for Perception, Action and Behaviour at the University of Edinburgh. 

It involved developing models for studying competitive evolution – specifically investigating any effect this may have on the emergent properties (patterns etc.) which can arise with cellular automata (and reaction diffusion models: [https://github.com/abrichardson00/Multi-Species-Reaction-Diffusion](https://github.com/abrichardson00/Multi-Species-Reaction-Diffusion)). Multi-species variations of these models were developed, and while they usually yield noisy or stationary (i.e. boring) lattices, it is possible to configure the update functions for the species such that patterns emerge.

![Example patterns](CA_examples.png?raw=true "Example patterns")

The cellular automata (CA) model is in principle quite simple: a global lattice of integer ‘states’ (brightness of pixel) are acted upon and changed by different CA species (colour of pixel) which compete for occupation of the lattice sites.

![Model illustration](CA.png?raw=true "Model illustration")

When multiple adjacent species compete for one lattice site, the species whose update function maps the state to the highest value gains possession of the site. More specifically, a normalized strength of an update is considered when comparing the actions of multiple species – i.e. a CA who only maps all states to the highest possible state isn’t necessarily a successful strategy for gaining possession – good strategies must involve updating some states to high valued states, and others to low valued states depending on the neighbouring lattice sites.

The ratio of the lattice occupied by a given CA species is a fitness measure that lends itself nicely to competitive / natural inspired evolution. The appropriately normalized `strengths' result in a given species constantly depending on the current update functions of other species. While written in Python, the implementation is fast – making use of vectorized Scipy operations, and Numba compiling. Hundreds of species can be be simulated at once on the same lattice.

There are interesting possible directions for investigation:

If evolving multple species competitively over time (genetic algorithm, fitness function described above), how do the update functions change, along with the behaviour of the whole system? Can patterns emerge through this process? Perhaps compartmentalization is an inevitible strategy that emerges - much like we see in biology (cells). 

Perhaps when including crossovers as part of a genetic algorithm (i.e. sexual reproduction) one can observe different species emerging. I.e. a very large number of CA species are initialized and evolved, and any species can crossover reproduce with another species to the next generation. Then over time different subsets of species diverge far enough from each other such that any crossover attempt with species outside their subset always fails to be successful.

# Usage
Run the file CA_animation.py to initialize and visualize a system:

python3 CA_animation.py size n_species kernel_radius n_states init_voronoi coherence

E.g:

python3 CA_animation.py 200 100 2 20 0 0.9

**Requirements:** numpy, matplotlib, scipy, numba, tqdm
