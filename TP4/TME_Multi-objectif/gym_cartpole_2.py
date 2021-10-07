import cma
import gym
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.tools._hypervolume import hv
import array
import random

import math

from nsga2 import my_nsga2

nn=SimpleNeuralControllerNumpy(4,1,2,5)
IND_SIZE=len(nn.get_parameters())

env = gym.make('CartPole-v1')
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
    error=[]

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
        sum_error=[0,0]
        for ind in pop:
            sum_error[0]+=ind.fitness.values[0]
            sum_error[1]+=ind.fitness.values[1]
        sum_error[0]=sum_error[0]/n
        sum_error[1]=sum_error[1]/n
        error.append(sum_error)
        paretofront.update(pop)
        pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
        s_hv.append(hv.hypervolume(pointset, ref_point))  
    return pop, paretofront, s_hv,error

def eval_nn(genotype, render=False, nbstep=500):
    total_x=0 # l'erreur en x est dans observation[0]
    total_theta=0 #  l'erreur en theta est dans obervation[2]
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.set_parameters(genotype)


    # à compléter

    # ATTENTION: vous êtes dans le cas d'une fitness à minimiser. Interrompre l'évaluation le plus rapidement possible est donc une stratégie que l'algorithme évolutionniste peut utiliser pour minimiser la fitness. Dans le cas ou le pendule tombe avant la fin, il faut donc ajouter à la fitness une valeur qui guidera l'apprentissage vers les bons comportements. Vous pouvez par exemple ajouter n fois une pénalité, n étant le nombre de pas de temps restant. Cela poussera l'algorithme à minimiser la pénalité et donc à éviter la chute. La pénalité peut être l'erreur au moment de la chute ou l'erreur maximale.
    observation = env.reset()
    n=0
    for i in range(nbstep):
        p = nn.predict(observation)[0]
        action = abs(round(p))

        observation, reward, done, info = env.step(action) 
        total_x+=np.abs(observation[0])
        total_theta+=np.abs(observation[2])
        if done:
            n=nbstep-i
            break
    
    return (total_x+n, total_theta+n)



if (__name__ == "__main__"):
    plt.figure()
    plt.title('Evolution de la moyenne des erreurs  au cours des itérations')
    plt.xlabel('Nombre d\'itération')
    plt.ylabel('Moyenne des erreur')
    nbsteps = 100
    nbIter=10
    for p in [50]:
        data=[my_nsga2(p, nbsteps, eval_nn,IND_SIZE=IND_SIZE)[1]for _ in range(nbIter)]
        env.close()
        print(data)
    #     #pop,hof,logbook=ea_simple.ea_simple(200, 100, benchmarks.ackley, 10)
    #     error_x=np.array([[data[i][j][0] for j in range(nbsteps-1)]for i in range(nbIter)])
    #     error_theta=np.array([[data[i][j][1] for j in range(nbsteps-1)]for i in range(nbIter)])
    #     gen=range(nbsteps-1)
    #     data=np.array(data)
    #     mediane_x=[np.median(error_x[:,i])for i in range(nbsteps-1)]
    #     fit_25_x=[np.quantile(error_x[:,i],0.25)for i in range(nbsteps-1)]
    #     fit_75_x=[np.quantile(error_x[:,i],0.75)for i in range(nbsteps-1)]
    #     plt.plot(gen,mediane_x, label="error_x mediane, pop="+str(p))
    #     plt.fill_between(gen, fit_25_x, fit_75_x, alpha=0.25, linewidth=0)
    #     mediane_theta=[np.median(error_theta[:,i])for i in range(nbsteps-1)]
    #     fit_25_theta=[np.quantile(error_theta[:,i],0.25)for i in range(nbsteps-1)]
    #     fit_75_theta=[np.quantile(error_theta[:,i],0.75)for i in range(nbsteps-1)]
    #     plt.plot(gen,mediane_theta, label="error_theta mediane, pop="+str(p))
    #     plt.fill_between(gen, fit_25_theta, fit_75_theta, alpha=0.25, linewidth=0)
    # plt.legend()
    # plt.show()
    
