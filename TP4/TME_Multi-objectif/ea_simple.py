import numpy as np
from deap import base, creator, benchmarks

import random
from deap import tools,algorithms

# ne pas oublier d'initialiser la graine aléatoire (le mieux étant de le faire dans le main))
random.seed()


def ea_simple(n, nbgen, evaluate, IND_SIZE, weights=(-1.0,)):
    """Algorithme evolutionniste elitiste

    Algorithme evolutionniste elitiste. 
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """

    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)


    toolbox = base.Toolbox()
    
    

    # à compléter pour sélectionner les opérateurs de mutation, croisement, sélection avec des toolbox.register(...)
    
    toolbox.register("attribute",random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat,creator.Individual,toolbox.attribute,n=IND_SIZE)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    
    toolbox.register("mate",tools.cxSimulatedBinary,eta=15)
    toolbox.register("mutate",tools.mutPolynomialBounded,eta=15,low=-5,up=5,indpb=0.2)
    toolbox.register("select",tools.selBest,k=n,fit_attr='fitness')
    toolbox.register("evaluate",evaluate)


    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()


    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)


    ## à compléter pour initialiser l'algorithme, n'oubliez pas de mettre à jour les statistiques, le logbook et le hall-of-fame.
    pop = toolbox.population(n=n)
    lambda_, cxpb, mutpb = int(n/2), 0.5, 0.5
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for g in range(1,nbgen+1):

        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb, mutpb)
        fitnesses = map(toolbox.clone, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = toolbox.select(offspring+pop)
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # Pour voir l'avancement
        # if (g%10==0):
        #     print("+",end="", flush=True)
        # else:
        #     print(".",end="", flush=True)


        ## à compléter en n'oubliant pas de mettre à jour les statistiques, le logbook et le hall-of-fame
        
        logbook.record( **stats.compile(pop),gen=g)
        hof.update(pop)
    return pop,hof,logbook
