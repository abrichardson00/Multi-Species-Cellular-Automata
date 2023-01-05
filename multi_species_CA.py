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
from numba import jit, njit, prange 
from tqdm import tqdm

from scipy import signal
from scipy.signal import convolve2d


def gkern(kernlen=5, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d * ((kernlen**2)/np.sum(gkern2d)) # whole array now also sums to kernlen**2



def estimate_coherence(A):
    return 1 - np.mean(np.abs(np.diff(A)))

def coherence_states(A):
    return np.mean(A[:-1]==A[1:])


class CA(object):

    def __init__(self, n_rules, n_states, coherence):

        self.n_states = n_states
        self.n_rules = n_rules
        
        # coherence is currently not used, but can be used with init_rules_and_strengths()
        self.coherence = coherence
        self.init_rules_and_strengths(coherence)
        
        """
        #self.rules = ((self.n_states/self.n_rules)*np.arange(self.n_rules)).astype('int64')
        #self.rules = (rand(self.n_rules)*self.n_states).astype('int64')
        self.rules = (self.n_states//2)*np.ones(self.n_rules).astype('int64')
        self.rule_strengths = (0.00001*rand(self.n_rules) + self.rules)*(1.0/self.n_states) # <- slight noise for random breaking of any 'ties'
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths)) # <- normalize strengths!
        """

    def rule_coherence(self):
        return 1 - (np.mean(np.abs(np.diff(self.rules)))/self.n_states)

    def strength_coherence(self):
        return 1 - np.mean(np.abs(np.diff(self.rule_strengths)))

    ### currently not used
    def init_rules_and_strengths_uniform(self, c):
        self.rules = (rand()*self.n_states*np.ones(self.n_rules)).astype('int64')
        self.rule_strengths = (0.000001*rand(self.n_rules) + self.rules)*(1.0/self.n_states)
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    ### currently not used
    def init_rules_and_strengths(self, c):
        self.rules = np.zeros(self.n_rules).astype('int64')
        #self.mutate_rules(10000)
        val = int(rand()*self.n_states)
        for i in range(self.n_rules):
            if rand()<0.5:
                mult = 1
                if val == self.n_states-1:
                    self.rules[i] = val
                    continue
            else:
                mult = -1
                if val == 0:
                    self.rules[i] = val
                    continue
            
            rand_val = rand()*(1-c)*self.n_states
            if rand_val < 1:
                if rand() < rand_val: 
                    val = (val + mult)%self.n_states
            else:
                val = (val + mult*int(rand_val))%self.n_states
            #if val > self.n_states-1:
            #    val = self.n_states-1
            
            self.rules[i] = val

        self.rule_strengths = (0.0001*rand(self.n_rules) + self.rules)*(1.0/self.n_states)
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    ### currently not used
    def init_rules_and_strengths_better(self, c):
        self.rules = np.zeros(self.n_rules).astype('int64')
        #self.mutate_rules(10000)
        val = int(rand()*self.n_states)
        for i in range(self.n_rules):
            if rand()<0.5:
                mult = 1
                if val == self.n_states-1:
                    self.rules[i] = val
                    continue
            else:
                mult = -1
                if val == 0:
                    self.rules[i] = val
                    continue
            
            rand_val = rand()*(1-c)*self.n_states
            if rand_val < 1:
                if rand() < rand_val: 
                    val = (val + mult)%self.n_states
            else:
                val = (val + mult*int(rand_val))%self.n_states
            #if val > self.n_states-1:
            #    val = self.n_states-1
            
            self.rules[i] = val

        self.rule_strengths = (0.0001*rand(self.n_rules) + self.rules)*(1.0/self.n_states)
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    def init_rules_and_strengths_2(self, c):
        n = int(1.0 / (c))*self.n_rules
        self.rules = np.zeros(self.n_rules).astype('int64')
        for i in range(n):
            rand_idx = int(rand()*self.n_rules)
            rand_idx_2 = (rand_idx + int(rand()*self.n_rules*(c)))%self.n_rules
            rand_value = int(rand()*self.n_states)
            if rand_idx <= rand_idx_2:
                self.rules[rand_idx:rand_idx_2] = rand_value
            else:
                self.rules[0:rand_idx_2] = rand_value
                self.rules[rand_idx:] = rand_value

        self.rule_strengths = (0.0001*rand(self.n_rules) + self.rules)*(1.0/self.n_states)
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    def mutate_rules_and_strengths(self, n):
        for i in range(n):
            rand_idx = int(rand()*self.n_rules)
            rand_idx_2 = (rand_idx + int(rand()*self.n_rules*(1.0/n)))%self.n_rules
            rand_value = int(rand()*self.n_states)
            rand_strength_value = rand()
            if rand_idx <= rand_idx_2:
                self.rules[rand_idx:rand_idx_2] = rand_value
                self.rule_strengths[rand_idx:rand_idx_2] = rand_strength_value
            else:
                self.rules[0:rand_idx_2] = rand_value
                self.rules[rand_idx:] = rand_value
                self.rule_strengths[0:rand_idx_2] = rand_strength_value
                self.rule_strengths[rand_idx:] = rand_strength_value
            self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    def mutate_rules(self, n):
        for i in range(n):
            rand_idx = int(rand()*self.n_rules)
            rand_idx_2 = (rand_idx + int(rand()*self.n_rules*0.2))%self.n_rules
            rand_value = int(rand()*self.n_states)
            if rand_idx <= rand_idx_2:
                self.rules[rand_idx:rand_idx_2] = rand_value
            else:
                self.rules[0:rand_idx_2] = rand_value
                self.rules[rand_idx:] = rand_value
        
    def mutate_strengths(self, n):
        for i in range(n):
            rand_idx = int(rand()*self.n_rules)
            rand_idx_2 = (rand_idx + int(rand()*self.n_rules*0.5))%self.n_rules
            rand_value = rand()
            if rand_idx <= rand_idx_2:
                self.rule_strengths[rand_idx:rand_idx_2] = rand_value
            else:
                self.rule_strengths[0:rand_idx_2] = rand_value
                self.rule_strengths[rand_idx:] = rand_value
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))

    #def mutate(self):
        #self.mutate_rules(1)
        #self.mutate_strengths(1)
    #    self.mutate_rules_and_strengths(1)

    def mutate(self, n):
        for i in range(n):
            rand_idx = int(rand()*self.n_rules)
            self.rules[rand_idx] = int(rand()*self.n_states)
        self.rule_strengths = (0.000001*rand(self.n_rules) + self.rules)*(1.0/self.n_states)
        self.rule_strengths = self.rule_strengths*(1.0/np.sum(self.rule_strengths))


def dist(a, b):
    return np.linalg.norm(a - b)

@jit(nopython=True)#, parallel=True)
def update_fast_3(size, rule_indices, is_boundary_cell, species_lattice, all_CA_rules ,all_CA_rule_strengths, next_lattice, next_species_lattice, kernel, n_species):
    
    #next_lattice = np.zeros((size,size))
    #next_species_lattice = np.zeros((size,size))
    
    n1s = [-1, 0, 1, 0]
    n2s = [0, 1, 0, -1]
    
    kern_len = kernel.shape[0]
    kern_start = 0 - (kern_len//2)
    strengths = np.zeros(n_species)

    for i in range(size):
            for j in range(size):
                #if is_boundary_cell[i,j]:
                best_strength = 0
                best_i = i
                best_j = j
                #for n in [[-1,0],[0,1],[1,0],[0,-1]]:
                for s in range(n_species):
                    strengths[s] = 0.0
                
                for n1 in range(kern_len):
                    for n2 in range(kern_len):
                        i_plus_n = (i+n1+kern_start)%size
                        j_plus_n = (j+n2+kern_start)%size
                        #print(i_plus_n)
                        strength = kernel[n1,n2]*all_CA_rule_strengths[species_lattice[i_plus_n,j_plus_n], rule_indices[i,j]]
                        strengths[species_lattice[i_plus_n, j_plus_n]] += strength
                        #if strength > best_strength:
                        #    best_strength = strength
                        #    best_i = i_plus_n
                        #    best_j = j_plus_n
                next_species = np.argmax(strengths)
                next_species_lattice[i,j] = next_species
                next_lattice[i,j] = all_CA_rules[next_species, rule_indices[i,j]]
                #else:
                #    next_lattice[i,j] = all_CA_rules[species_lattice[i,j], rule_indices[i,j]]
                #self.lattice[i,j] = self.behaviour.rule(rule_indices[i,j])
    #print("breh")
    return next_lattice, next_species_lattice


@jit(nopython=True)#, parallel=True)
def update_fast_2(size, rule_indices, is_boundary_cell, species_lattice, all_CA_rules ,all_CA_rule_strengths, next_lattice, next_species_lattice, kernel):
    
    #next_lattice = np.zeros((size,size))
    #next_species_lattice = np.zeros((size,size))
    
    n1s = [-1, 0, 1, 0]
    n2s = [0, 1, 0, -1]
    
    kern_len = kernel.shape[0]
    kern_start = 0 - (kern_len//2)

    for i in range(size):
            for j in range(size):
                #if is_boundary_cell[i,j]:
                best_strength = 0
                best_i = i
                best_j = j
                #for n in [[-1,0],[0,1],[1,0],[0,-1]]:
                
                for n1 in range(kern_len):
                    for n2 in range(kern_len):
                        i_plus_n = (i+n1+kern_start)%size
                        j_plus_n = (j+n2+kern_start)%size
                        #print(i_plus_n)
                        strength = kernel[n1,n2]*all_CA_rule_strengths[species_lattice[i_plus_n,j_plus_n], rule_indices[i,j]]
                        if strength > best_strength:
                            best_strength = strength
                            best_i = i_plus_n
                            best_j = j_plus_n
                next_species_lattice[i,j] = species_lattice[best_i, best_j]
                next_lattice[i,j] = all_CA_rules[species_lattice[best_i,best_j], rule_indices[i,j]]
                #else:
                #    next_lattice[i,j] = all_CA_rules[species_lattice[i,j], rule_indices[i,j]]
                #self.lattice[i,j] = self.behaviour.rule(rule_indices[i,j])
    #print("breh")
    return next_lattice, next_species_lattice

@jit(nopython=True)#, parallel=True)
def update_fast(size, rule_indices, is_boundary_cell, species_lattice, all_CA_rules, all_CA_rule_strengths, next_lattice, next_species_lattice):
    
    #next_lattice = np.zeros((size,size))
    #next_species_lattice = np.zeros((size,size))
    
    n1s = [-1, 0, 1, 0]
    n2s = [0, 1, 0, -1]
    
    for i in range(size):
            for j in range(size):
                if is_boundary_cell[i,j]:
                    best_strength = 0
                    best_i = i
                    best_j = j
                    #for n in [[-1,0],[0,1],[1,0],[0,-1]]:
                    
                    for n in range(4):
                        i_plus_n = (i+n1s[n])%size
                        j_plus_n = (j+n2s[n])%size
                        #print(i_plus_n)
                        strength = all_CA_rule_strengths[species_lattice[i_plus_n,j_plus_n], rule_indices[i,j]]
                        if strength > best_strength:
                            best_strength = strength
                            best_i = i_plus_n
                            best_j = j_plus_n
                    next_species_lattice[i,j] = species_lattice[best_i, best_j]
                    next_lattice[i,j] = all_CA_rules[species_lattice[best_i,best_j], rule_indices[i,j]]
                else:
                    next_lattice[i,j] = all_CA_rules[species_lattice[i,j], rule_indices[i,j]]
                #self.lattice[i,j] = self.behaviour.rule(rule_indices[i,j])
    #print("breh")
    return next_lattice, next_species_lattice

class MultiSpeciesCA(object):

    def __init__(self, lattice_size, n_species, radius, n_states, init_voronoi, coherence):

        self.size = lattice_size
        
        self.radius = radius
        self.width = 2*radius + 1
        self.n_states = n_states
        self.n_rules = self.n_states*(self.width**2)

        self.n_species = n_species
        self.n_species_max = 300

        self.init_voronoi = init_voronoi
        self.init_lattices(self.size, self.init_voronoi)

        self.coherence = coherence

        self.kernel = np.ones((self.width, self.width))
        #self.kernel = gkern(self.width, std=self.width//2)
        self.kernel2 = gkern(2*(self.radius//2)+1, std=(self.width//2)*0.5)
        #print(kernel2)
        self.edge_kernel = np.array([[0,-1, 0],[-1, 4,-1],[0,-1, 0]])

        self.CAs = [CA(self.n_rules, self.n_states, self.coherence) for i in range(n_species)]
        self.n_saves = 0

        self.all_CA_rules = np.array([ca.rules for ca in self.CAs])
        self.all_CA_rule_strengths = np.array([ca.rule_strengths for ca in self.CAs])

        self.target_lattice = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.target_lattice[i,:] = i % self.n_species


    def init_lattices(self, size, voronoi):
        self.size = size
        self.lattice = (rand(self.size, self.size)*self.n_states).astype('int64')
        self.next_lattice = np.copy(self.lattice)
        
        if voronoi:
            # random species sections (voronoi tesselation):
            self.species_lattice = np.zeros((self.size, self.size), dtype='int64')
            n_points = [(rand()*self.size, rand()*self.size) for i in range(self.n_species)]
            for i in range(self.size):
                for j in range(self.size):
                    dists = [dist(np.array([i,j]), n_points[s]) for s in range(len(n_points))]
                    self.species_lattice[i,j] = np.argmin(dists)
        else:
            # uniform random mix of species accross lattice:
            self.species_lattice = (rand(self.size, self.size)*self.n_species).astype('int64')
            #self.species_lattice = np.zeros((self.size, self.size)).astype('int64')


        self.next_species_lattice = np.copy(self.species_lattice)

    def init_lattice_2_teams(self, n_team_1):
        self.lattice = (rand(self.size, self.size)*n_states).astype('int64')
        self.next_lattice = np.copy(self.lattice)

        self.species_lattice = (rand(self.size, self.size)*4).astype('int64')
        self.species_lattice[(self.species_lattice == 3)] = 0
        self.next_species_lattice = np.copy(self.species_lattice)


    def mutate(self, n):
        for ca in self.CAs:
            ca.mutate(n)
        self.update_CA_matrices()

    def mutate_species(self, s, n):
        self.CAs[s].mutate(n)
        self.update_CA_matrices()

    def append_mutation(self, n):
        i = int(rand()*self.size)
        j = int(rand()*self.size)
        ca = self.CAs[self.species_lattice[i,j]]
        new_ca = copy.deepcopy(ca)
        new_ca.mutate(n)
        if self.n_species < self.n_species_max:
            self.CAs.append(new_ca)
            self.n_species += 1
            self.species_lattice[i,j] = self.n_species - 1
        else:
            population_sizes = self.population_size_ratios()
            #worst_ca_index = np.argmin(population_sizes)
            worst_ca_indices = np.argsort(population_sizes)[:20]
            worst_ca_index = worst_ca_indices[int(rand()*20)]
            self.CAs[worst_ca_index] = new_ca
            self.species_lattice[i,j] = worst_ca_index
        
        # THIS COULD BE MORE EFFICIENT:
        self.update_CA_matrices()


    def update_CA_matrices(self):
        self.all_CA_rules = np.array([ca.rules for ca in self.CAs])
        self.all_CA_rule_strengths = np.array([ca.rule_strengths for ca in self.CAs])
    
    def update(self):
        rule_indices = convolve2d(self.lattice, self.kernel, mode='same', boundary='wrap').astype(int)
        is_boundary_cell = convolve2d(self.species_lattice, self.edge_kernel, mode='same', boundary='wrap') != 0  # true if at boundary

        #self.next_lattice, self.next_species_lattice = update_fast(self.size, rule_indices, is_boundary_cell, self.species_lattice, self.all_CA_rules, self.all_CA_rule_strengths, self.next_lattice, self.next_species_lattice)
        self.next_lattice, self.next_species_lattice = update_fast_2(self.size, rule_indices, is_boundary_cell, self.species_lattice, self.all_CA_rules, self.all_CA_rule_strengths, self.next_lattice, self.next_species_lattice, self.kernel)
        #self.next_lattice, self.next_species_lattice = update_fast_3(self.size, rule_indices, is_boundary_cell, self.species_lattice, self.all_CA_rules, self.all_CA_rule_strengths, self.next_lattice, self.next_species_lattice, self.kernel, self.n_species)
        
        self.lattice[:,:] = self.next_lattice
        self.species_lattice[:,:] = self.next_species_lattice

    def population_size_ratios(self):
        sizes = np.zeros(self.n_species)
        for i in range(self.n_species):
            sizes[i] = np.sum(self.species_lattice==i)
        return sizes * (1.0 / self.size**2)
    
    def lattice_coherence_estimate(self, lattice, N):
        coherences = np.zeros(N)
        for n in range(N):
            i = int(rand()*(self.size-(2*self.radius)))+self.radius
            j = int(rand()*(self.size-(2*self.radius)))+self.radius
            value = lattice[i,j]
            
            surrounding_values = lattice[i-self.radius:i+1+self.radius, j-self.radius:j+1+self.radius]
            coherences[n] = np.mean((surrounding_values==value)*self.kernel)
        return np.mean(coherences)

    
    def simulate_batch_2(self, n_sims, iterations):
        n_values = iterations//5

        fitness_data = np.zeros((self.n_species,n_sims,n_values))
        #population_variances = np.zeros((self.n_species, n_sims))
        #lattice_coherences = np.zeros((n_sims, n_values))
        #species_coherences = np.zeros((n_sims, n_values))
        

        prev_lattice = np.zeros_like(self.lattice)
        for n in range(n_sims):
            self.init_lattices(self.size, self.init_voronoi)
            for i in range(20):
                # some equilibration time
                self.update()

            #population_sizes = np.zeros((n_values, self.n_species))
            for i in range(n_values):
                for j in range(4):
                    self.update()
                prev_lattice[:,:] = self.lattice
                self.update()

                #lattice_coherences[n,i] = self.lattice_coherence_estimate(self.lattice, 20)
                #species_coherences[n,i] = self.lattice_coherence_estimate(self.species_lattice, 20)
                #population_sizes[i,:] = self.population_size_ratios()

                is_different = (self.lattice!=prev_lattice)
                # is this a bad way to estimate entropy? should consider more than just 2 points in time - i.e. maybe a history of 5 steps or something with gaussian decaying weights
                for s in range(n_species):
                    s_lattice = self.species_lattice==s 
                    n_s = np.sum(s_lattice) 
                    if n_s != 0:
                        fitness_data[s,n,i] = np.sum(s_lattice&is_different)/(self.size*self.size)
            
            #population_variances[:,n] = np.var(population_size_ratios, axis=0)

        fitness_means = np.mean(fitness_data, axis=2)
        if fitness_means.shape[1] != n_sims:
            print("ummm bad bad")
        return np.mean(fitness_means, axis=1)#, np.mean(population_variances, axis=1), np.mean(lattice_coherences), np.mean(species_coherences) # shape: (n_species,)

    def simulate_batch(self, n_sims, iterations):
        n_values = iterations//5

        fitness_data = np.zeros((self.n_species,n_sims,n_values))

        prev_lattice = np.zeros_like(self.lattice)
        for n in range(n_sims):
            self.init_lattices(self.size, self.init_voronoi)
            for i in range(20):
                # some equilibration time
                self.update()

            for i in range(n_values):
                for j in range(4):
                    self.update()
                prev_lattice[:,:] = self.lattice
                self.update()

                fitness_data[:,n,i] = self.population_size_ratios()

        fitness_means = np.mean(fitness_data, axis=2)
        if fitness_means.shape[1] != n_sims:
            print("ummm bad bad")
        return np.mean(fitness_means, axis=1)

    def simulate_tournament_batch(self, n_sims, iterations, n_team_1):

        fitness_data = np.zeros((2,n_sims))
        population_ratios = np.zeros((self.n_species, n_sims))
        for n in range(n_sims):
            self.init_lattice_2_teams(n_team_1)
            for i in range(iterations):
                self.update()
            population_ratios[:,n] = self.population_size_ratios()
            fitness_data[0,n] = np.sum(population_ratios[0:n_team_1,n])
            fitness_data[1,n] = np.sum(population_ratios[n_team_1:,n])

        #print(np.mean(population_ratios, axis=1).shape)
        return np.mean(fitness_data, axis=1), np.mean(population_ratios, axis=1)




    def entropy_fitness(self, prev_lattice):
        fitnesses = np.zeros(self.n_species)

        is_different = (self.lattice!=prev_lattice)
        # is this a bad way to estimate entropy? should consider more than just 2 points in time - i.e. maybe a history of 5 steps or something with gaussian decaying weights
        for s in range(n_species):
            s_lattice = self.species_lattice==s 
            fitnesses[s] = np.sum(s_lattice&is_different)/(self.size*self.size)
        return fitnesses

    def target_fitness(self, prev_lattice):
        fitnesses = np.zeros(self.n_species)
        for s in range(self.n_species):
            fitnesses[s] = np.sum((self.species_lattice==s) & (self.target_lattice==s)) * (1.0/(self.size**2))
        return fitnesses

    def cooperation_fitness(self, prev_lattice):
        #population_sizes = self.population_size_ratios()*(self.size**2)
        #target = (self.size**2) / self.n_species
        #return -(population_sizes - target)**2
        population_sizes = self.population_size_ratios()
        target = 1.0 / self.n_species
        return 1.0 - np.abs(population_sizes - target)




    def get_metrics(self, n_sims, iterations):
        n_values = iterations//5

        fitness_data = np.zeros((self.n_species,n_sims,n_values))
        population_variances = np.zeros((self.n_species, n_sims))
        lattice_coherences = np.zeros((n_sims, n_values))
        species_coherences = np.zeros((n_sims, n_values))
        

        prev_lattice = np.zeros_like(self.lattice)
        for n in range(n_sims):
            self.init_lattices(self.size, self.init_voronoi)
            for i in range(20):
                # some equilibration time
                self.update()

            population_sizes = np.zeros((n_values, self.n_species))
            for i in range(n_values):
                for j in range(4):
                    self.update()
                prev_lattice[:,:] = self.lattice
                self.update()

                lattice_coherences[n,i] = self.lattice_coherence_estimate(self.lattice, 40)
                species_coherences[n,i] = self.lattice_coherence_estimate(self.species_lattice, 40)
                population_sizes[i,:] = self.population_size_ratios()

                fitness_data[:,n,i] = self.population_size_ratios()
            
            population_variances[:,n] = np.var(population_sizes, axis=0)

        fitness_means = np.mean(fitness_data, axis=2)
        if fitness_means.shape[1] != n_sims:
            print("ummm bad bad")
        return np.mean(fitness_means, axis=1), np.mean(population_variances, axis=1), np.mean(lattice_coherences), np.mean(species_coherences) # shape: (n_species,)


    def save_to_file(self):
        self.n_saves += 1 



def genetic_algorithm_simulation(model, iterations, n_mutated):
    fitnesses = np.zeros((iterations, model.n_species))
    population_variances = np.zeros((iterations, model.n_species))
    lattice_coherences = np.zeros(iterations)
    species_coherences = np.zeros(iterations)
    for i in tqdm(range(iterations)):
        fitness_array = np.zeros((n_mutated, model.n_species))
        mutated_models = [copy.deepcopy(model) for i in range(n_mutated)]
        for m in range(len(mutated_models)):
            mutated_models[m].mutate(2)
            fitness_array[m,:] = mutated_models[m].simulate_batch(5, 20)

        best_CAs = []
        max_fitnesses = []
        for s in range(model.n_species):
            best_mutation_index = np.argmax(fitness_array[:,s])
            max_fitnesses.append(fitness_array[best_mutation_index,s])
            best_CAs.append(mutated_models[best_mutation_index].CAs[s])
        
        # assume worst species goes extinct, and best species takes over it's slot in CA array
        if np.min(max_fitnesses) == 0:
            best_CAs[np.argmin(max_fitnesses)] = copy.deepcopy(best_CAs[np.argmax(max_fitnesses)])

        model.CAs = copy.deepcopy(best_CAs) # deepcopy might not be needed here?
        model.update_CA_matrices()

        fitnesses[i,:], population_variances[i,:], lattice_coherences[i], species_coherences[i] = model.get_metrics(5, 40)

    return model, fitnesses, population_variances, lattice_coherences, species_coherences


def genetic_algorithm_simulation_2(model, iterations, n_mutated):
    fitnesses = np.zeros((iterations, model.n_species))
    population_variances = np.zeros((iterations, model.n_species))
    lattice_coherences = np.zeros(iterations)
    species_coherences = np.zeros(iterations)

    replacement_prob = 0.0#(0.5/model.n_species)
    n_mutate_at_once = 1 # 1 = completely asynchronous genome update, larger value => synchronous update

    for i in tqdm(range(iterations)):
        fitness_array = np.zeros((n_mutated, model.n_species))
        mutated_models = [copy.deepcopy(model) for i in range(n_mutated)]

        
        if rand() < replacement_prob and i > 0:
            worst_CA_idx = np.argmin(fitnesses[i-1,:])
            best_CAs[worst_CA_idx] = copy.deepcopy(best_CAs[np.argmax(fitnesses[i-1,:])])
            species_to_update = np.array([worst_CA_idx])
            #best_CAs[worst_CA_idx].mutate(1)
        else:
            if n_mutate_at_once == 1:
                species_to_update = np.array([int(rand()*model.n_species)])
            else:
                species_to_update = (rand(n_mutate_at_once)*model.n_species).astype('int64')

        for m in range(len(mutated_models)):
            for s in species_to_update:
                mutated_models[m].mutate_species(s, 1)
            fitness_array[m,:] = mutated_models[m].simulate_batch(1, 20)

        best_CAs = []
        max_fitnesses = []
        for s in range(model.n_species):
            best_mutation_index = np.argmax(fitness_array[:,s])
            max_fitnesses.append(fitness_array[best_mutation_index,s])
            best_CAs.append(mutated_models[best_mutation_index].CAs[s])
        
        # assume worst species goes extinct, and best species takes over it's slot in CA array
        #if np.min(max_fitnesses) == 0:
        #    best_CAs[np.argmin(max_fitnesses)] = copy.deepcopy(best_CAs[np.argmax(max_fitnesses)])
        #replacement_prob = (1.0/model.n_species)
        #if rand() < replacement_prob:
        #    worst_CA_idx = np.argmin(max_fitnesses)
        #    best_CAs[worst_CA_idx] = copy.deepcopy(best_CAs[np.argmax(max_fitnesses)])
        #    best_CAs[worst_CA_idx].mutate(1)

        model.CAs = copy.deepcopy(best_CAs) # deepcopy might not be needed here?
        model.update_CA_matrices()

        fitnesses[i,:], population_variances[i,:], lattice_coherences[i], species_coherences[i] = model.get_metrics(5, 20)

    return model, fitnesses, population_variances, lattice_coherences, species_coherences

def genetic_algorithm_with_saves(model, model_name, n_saves, iterations, n_mutated):
    for s in tqdm(range(n_saves)):
        if s == (n_saves-1):
            iters = (iterations // n_saves) + (iterations%n_saves) 
            file_name = model_name
        else:
            iters = (iterations // n_saves)
            file_name = model_name + "_checkpoint_" + str(s)
        model, fitnesses, population_variances, lattice_coherences, species_coherences = genetic_algorithm_simulation_2(model, iters, n_mutated)

        fig, axs = plt.subplots(nrows=2, ncols=2)
        for i in range(model.n_species):
            axs[0,0].plot(fitnesses[:,i])
            axs[0,1].plot(population_variances[:,i])
        axs[0,0].legend([str(i) for i in range(model.n_species)])
        axs[0,0].set_ylabel("fitness")
        axs[0,1].legend([str(i) for i in range(model.n_species)])
        axs[0,1].set_ylabel("population variances")

        axs[1,0].plot(lattice_coherences)
        axs[1,0].set_ylabel("spatial lattice coherence")
        axs[1,1].plot(species_coherences)
        axs[1,1].set_ylabel("spatial species coherence")
        plt.show()

        file = open(file_name, 'wb')
        pickle.dump(model, file)
        file.close()
    return model



def average_functions(size, n_species, radius, n_states, init_voronoi):
    n_runs = 1000
    n_rules = ((2*radius + 1)**2)*n_states
    best_functions = np.zeros((n_runs, n_rules))
    middle_functions = np.zeros((n_runs, n_rules))
    worst_functions = np.zeros((n_runs, n_rules))
    for i in tqdm(range(n_runs)):
        model = Model(size, n_species, radius, n_states, init_voronoi,0.99)
        fitnesses = model.simulate_batch_2(1,50) # is this right??
        sorted_species = np.argsort(fitnesses)
        #print(sorted_species)
        best_functions[i,:] = model.all_CA_rules[sorted_species[n_species-1]]
        middle_functions[i,:] = model.all_CA_rules[sorted_species[n_species//2]]
        worst_functions[i,:] = model.all_CA_rules[sorted_species[0]]
    
    best_visual = np.zeros((20, n_rules))
    middle_visual = np.zeros((20, n_rules))
    worst_visual = np.zeros((20, n_rules))
    for i in range(n_rules):
        best_visual[:,i], bins = np.histogram(best_functions[:,i], bins=20, range=(0,20))
        middle_visual[:,i], bins = np.histogram(middle_functions[:,i], bins=20, range=(0,20))
        worst_visual[:,i], bins = np.histogram(worst_functions[:,i], bins=20, range=(0,20))
    #print(bins)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].plot(np.mean(best_functions, axis=0), color='red')
    axs[1].plot(np.mean(middle_functions, axis=0), color='red')
    axs[2].plot(np.mean(worst_functions, axis=0), color='red')
    xs = np.arange(n_rules)
    ct = axs[0].pcolormesh(xs, bins, best_visual)
    ct = axs[1].pcolormesh(xs, bins, middle_visual)
    ct = axs[2].pcolormesh(xs, bins, worst_visual)
    cb = fig.colorbar(ct, ax=axs[2])
    cb.set_label("count", rotation=0)
    axs[0].set_xlabel("$x$")
    axs[1].set_xlabel("$x$")
    axs[2].set_xlabel("$x$")
    axs[0].set_ylabel("$f(x)$")
    
    plt.show()

