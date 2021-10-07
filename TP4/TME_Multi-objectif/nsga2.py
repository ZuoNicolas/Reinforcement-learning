import numpy as np
from deap import base, creator, benchmarks

from deap import algorithms
from deap.tools._hypervolume import hv


import random
from deap import tools

# ne pas oublier d'initialiser la grane aléatoire (le mieux étant de le faire dans le main))
random.seed()

def my_nsga2(n, nbgen, evaluate, ref_point=np.array([1,1]), IND_SIZE=5, weights=(-1.0, -1.0)):
    """NSGA-2

    NSGA-2
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param ref_point: le point de référence pour le calcul de l'hypervolume
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """

    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    

    toolbox = base.Toolbox()
    paretofront = tools.ParetoFront()

    # à compléter

    toolbox.register("attribute", random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                     
    toolbox.register("mate", tools.cxSimulatedBinary,eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=15.0,low=-5,up=5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2,k=n, nd = 'standard' )
    toolbox.register("evaluate", evaluate)

    # Pour récupérer l'hypervolume, nous nous contenterons de mettre les différentes aleur dans un vecteur s_hv qui sera renvoyé par la fonction.
    pop = toolbox.population(n=n)
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    paretofront.update(pop)
    pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
    s_hv=[hv.hypervolume(pointset, ref_point)]

    # Begin the generational process

    lambda_, cxpb, mutpb = 10, 0.5, 0.5
    for gen in range(1, nbgen):
        # if (gen%10==0):
        #     print("+",end="", flush=True)
        # else:
        #     print(".",end="", flush=True)

        # à completer
        
        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb, mutpb)

        offspring = list(map(toolbox.clone, offspring))
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = toolbox.select(pop + offspring)
        paretofront.update(pop)
        pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
        s_hv.append(hv.hypervolume(pointset, ref_point))
            
    return pop, paretofront, s_hv
