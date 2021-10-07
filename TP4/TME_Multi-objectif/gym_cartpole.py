import cma
import gym
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import array
import random

import math
import matplotlib.pyplot as plt
import time
# La ligne suivante permet de lancer des calculs en parallèle, ce qui peut considérablement accélérer les calculs sur une machine multi-coeur. Pour cela, il vous faut charger le module scoop: python -m scoop gym_cartpole.py
from scoop import futures
# pour que DEAP utilise la parallélisation, il suffit alors d'ajouter toolbox.register("map", futures.map) dans la paramétrisation de l'algorithme évolutionniste. Si vous souhaitez explorer cette possibilité, nous vous conseillons de ne pas mettre l'algorithme évolutionniste dans un fichier séparé, cela peut créer des problèmes avec DEAP.
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

    toolbox.register("attribute", random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                     
    toolbox.register("mate", tools.cxSimulatedBinary,eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=15.0,low=-5,up=5, indpb=0.2)
    toolbox.register("select", tools.selBest,k=n, fit_attr='fitness')
    toolbox.register("evaluate", evaluate)
    
    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()


    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)


    ## à compléter pour initialiser l'algorithme, n'oubliez pas de mettre à jour les statistiques, le logbook et le hall-of-fame.
    
    pop = toolbox.population(n=n)
    lambda_, cxpb, mutpb = 10, 0.5, 0.5
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    for g in range(1,nbgen+1):

        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb, mutpb)

        offspring = list(map(toolbox.clone, offspring))
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = toolbox.select(pop + offspring)
        """
        # Pour voir l'avancement
        if (g%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)
        """
        ## à compléter en n'oubliant pas de mettre à jour les statistiques, le logbook et le hall-of-fame
        logbook.record( gen=g, **stats.compile(pop))
        hof.update(pop)
        
    return pop, hof, logbook

# Pour récupérer le nombre de paramètre. voir fixed_structure_nn_numpy pour la signification des paramètres. Le TME fonctionne avec ces paramètres là, mais vous pouvez explorer des valeurs différentes si vous le souhaitez.
nn=SimpleNeuralControllerNumpy(4,1,2,5)
IND_SIZE=len(nn.get_parameters())

env = gym.make('CartPole-v1')


def eval_nn(genotype, render=False, nbstep=500):
    total_reward=0
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.set_parameters(genotype)

    ## à completer

    # utilisez render pour activer ou inhiber l'affichage (il est pratique de l'inhiber pendant les calculs et de ne l'activer que pour visualiser les résultats. 
    # nbstep est le nombre de pas de temps. Plus il est grand, plus votre pendule sera stable, mais par contre, plus vos calculs seront longs. Vous pouvez donc ajuster cette
    # valeur pour accélérer ou ralentir vos calculs. Utilisez la valeur par défaut pour indiquer ce qui doit se passer pendant l'apprentissage, vous pourrez indiquer une 
    # valeur plus importante pour visualiser le comportement du résultat obtenu.
    observation = env.reset()
    for i in range(nbstep):
        if render:
            env.rend()
        p = nn.predict(observation)[0]
        action = abs(round(p))

        observation, reward, done, info = env.step(int(action)) 
        total_reward += reward
        if done:
            break
    
    if render:
        show_video()
        
    return total_reward,observation[0],observation[2] #0=0 et 2=angle


if (__name__ == "__main__"):

    # faites appel à votre algorithme évolutionniste pour apprendre une politique et finissez par une visualisation du meilleur individu
    #pop, hof, logbook = ea_simple(200, 100, eval_nn, IND_SIZE, weights=(1.0,))
    


    plt.figure()
    plt.title('Evolution de la moyenne de nos fit au cours des itérations')
    plt.xlabel('Nombre d\'itération')
    plt.ylabel('Moyenne des fits')
    nbsteps = 200
    nbIter=10
    time = time.time()
    for p in [5, 10,100,200]:
        data=[ea_simple(p, nbsteps, eval_nn, IND_SIZE,weights=(1.0,))[2]for _ in range(nbIter)]
        env.close()
        #pop,hof,logbook=ea_simple.ea_simple(200, 100, benchmarks.ackley, 10)
        gen=data[0].select("gen")
        genscore=[[data[i].select("avg")[j] for i in range(nbIter)]for j in range(nbsteps)]
        moyenne=[np.median(genscore[i])for i in range(nbsteps)]
        fit_25=[np.quantile(genscore[i],0.25)for i in range(nbsteps)]
        fit_75=[np.quantile(genscore[i],0.75)for i in range(nbsteps)]
        plt.plot(gen,moyenne, label="Fitness moyenne, pop="+str(p))
        plt.fill_between(gen, fit_25, fit_75, alpha=0.25, linewidth=0)
    plt.legend()
    plt.show()
    print(time.time() - time)
