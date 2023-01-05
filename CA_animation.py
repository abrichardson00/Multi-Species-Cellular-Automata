import sys
import os
import copy
import pickle
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib
from multi_species_CA import CA
from multi_species_CA import MultiSpeciesCA


"""
python script for animating some multi species cellular automata

edit main function appropriately... can load saved models etc.

usage:

python3 CA_animation.py size n_species kernel_radius n_states init_voronio coherence
e.g:
python3 CA_animation.py 200 100 2 20 0 0.9
"""



def model_animation(model, size, init_voronoi):
    model.init_lattices(size, init_voronoi)
    #model.init_lattice_2_teams(1)

    
    print("rule and strength coherences:\n")
    for i in range(len(model.CAs)):
        ca = model.CAs[i]
        print("CA " + str(i+1) + " rules: " + str(ca.rule_coherence()) + "\tstrengths: " + str(ca.strength_coherence()))
    print("")

    fig, axs = plt.subplots(nrows=1,ncols=2)
    axs[0].plot(model.CAs[0].rules)
    axs[1].plot(model.CAs[1].rules)
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$f(x)$", rotation=0)
    plt.show()

    #plt.imshow(model.kernel2)
    #plt.show()

    # SETUP ANIMATION --------------------------------------
    fig, ax = plt.subplots()
    ax.set_facecolor('black')

    colourmap = cm.get_cmap('gist_rainbow')
    implot = ax.imshow(model.species_lattice, cmap=colourmap, interpolation='nearest', alpha=0.4 + 0.6*(model.lattice+1)*(1.0/(model.n_states+1)))
    fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0,vmax=(model.n_species-1)) , cmap=colourmap), ax=ax)
    
    population_size_data = []

    def update(frame):
        model.update()
        #model.update()
        if (frame % 5 == 0):
            population_size_data.append(model.population_size_ratios())
        if (frame % 20 == 0): print("iteration " + str(frame))
        implot.set_data(model.species_lattice)
        implot.set(alpha=0.4 + 0.6*(model.lattice+1)*(1.0/(model.n_states+1)))

    ani = animation.FuncAnimation(fig, update, interval=10)
    plt.show()

    # SAVE THE MODEL ---------------------------------------
    x = input("save the model? (y/n): ")
    if x == "y" or x == "Y":
        file_name = input("enter model name: ")
        print("saving the model")
        file = open(file_name, 'wb')
        pickle.dump(model, file)
        file.close()
    else:
        print("not saving")

    population_size_data = np.array(population_size_data)
    for s in range(model.n_species):
        print("species " + str(s) + ": " + str(np.std(population_size_data[1:,s])))

    # PLOT POPULATION SIZES ---------------------------------
    fig, ax = plt.subplots()
    
    cmapvals = np.linspace(0,1,model.n_species)
    #print(population_size_data.shape)
    for i in range(model.n_species):
        ax.plot(population_size_data[:,i], color=colourmap(cmapvals[i]))
    #ax.legend([str(i) for i in range(model.n_species)])
    ax.set_ylabel("species fitnesses")
    plt.show()



def load_model(model_name):
    file = open(model_name, 'rb')
    model = pickle.load(file)
    file.close()
    return model



# Main entry point of the program
if __name__ == "__main__":
    
    # Read input arguments
    args = sys.argv
    size = int(args[1])
    n_species = int(args[2])
    radius = int(args[3])
    n_states = int(args[4])
    init_voronoi = bool(int(args[5]))
    coherence = float(args[6])
    print(init_voronoi)

    model = MultiSpeciesCA(size, n_species, radius, n_states, init_voronoi, coherence)
    #model.mutate(10)
    #model_name = "ga4_some_replacement"
    #model = load_model(model_name)

    model_animation(model, size, init_voronoi)
