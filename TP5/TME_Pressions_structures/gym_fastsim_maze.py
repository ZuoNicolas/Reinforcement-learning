import cma
import gym, gym_fastsim
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import argparse

import array
import random
import operator
import math

from plot import *

from scoop import futures

from novelty_search import *

from maze_plot import *

# Evaluation d'un réseau de neurones en utilisant gym
def eval_nn(genotype, nbstep=2000, render=False, name="", nn_size=[5,2,2,10]):
    nn=SimpleNeuralControllerNumpy(*nn_size)
    nn.set_parameters(genotype)
    observation = env.reset()
    old_pos=None
    total_dist=0
    if (render):
        f=open("log/traj"+name+".log","w")
    for t in range(nbstep):
        if render:
            env.render()
        action=nn.predict(observation)
        observation, reward, done, info = env.step(action) 
        pos=info["robot_pos"][:2]
        if(render):
            f.write(" ".join(map(str,pos))+"\n")
        if (old_pos is not None):
            d=math.sqrt((pos[0]-old_pos[0])**2+(pos[1]-old_pos[1])**2)
            total_dist+=d
        old_pos=list(pos)
        if(done):
            break
    if (render):
        f.close()
        plot_points_file("log/traj"+name+".log", bg="maze_hard.pbm", title=name, show=False)
    dist_obj=info["dist_obj"]
    #print("End of eval, total_dist=%f"%(total_dist))
    # Remarque: les positions et distances sont arrondis à 2 décimales pour éviter le surapprentissage et le maintien dans le front de pareto de solutions ne différant que
    # de décimales plus éloignées (variante FIT+NS)
    rpos=[round(x,2) for x in pos]
    return round(dist_obj,2), rpos


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


# pour que DEAP ne pose pas de soucis... (à refaire avec la bonne configuration de weights de fitness  dans launch_nsga2)
creator.create("MyFitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
creator.create("Strategy", array.array, typecode="d")


def launch_nsga2(mu=100, lambda_=200, cxpb=0.6, mutpb=0.3, ngen=200, variant="FIT", nn_size=[5,2,2,10], verbose=False):
    
    random.seed()

    nn=SimpleNeuralControllerNumpy(*nn_size)
    params=nn.get_parameters()
    IND_SIZE=len(params)
    MIN_VALUE = -30
    MAX_VALUE = 30
    MIN_STRATEGY = 0.5
    MAX_STRATEGY = 3

    ## à faire: créer MyFitness selon la variante

    if variant == "FIT" or variant == "NS":
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    else:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0))
		
    creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None)
    creator.create("Strategy", array.array, typecode="d")


    ## à faire: définition de la toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, scls=creator.Strategy, size=IND_SIZE, imin=MIN_VALUE, imax=MAX_VALUE, smin=MIN_STRATEGY, smax=MAX_STRATEGY)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    toolbox.register("select",tools.selNSGA2,k=mu, nd='standard')
    toolbox.register("evaluate",eval_nn, nn_size=nn_size)
    toolbox.register("mate", tools.cxSimulatedBinary, eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=15.0,low=MIN_VALUE, up=MAX_VALUE, indpb=0.2)
    
    toolbox.register("map",futures.map)
    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
    


    ## à faire: création de la population
    population = toolbox.population(n=mu)

    paretofront = tools.ParetoFront()
    
    # pour sauvegarder la position finale des politiques explorées
    fbd=open("bd.log","w")

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, (fit, bd) in zip(invalid_ind, fitnesses_bds):
        
        if (variant=="FIT+NS"):
            ind.fitness.values=(fit,0)
        elif (variant=="FIT"):
            ind.fitness.values=(fit,)
        elif (variant=="NS"):
            ind.fitness.values=(0,)
        
        ind.fit = fit
        ind.bd = bd
        fbd.write(" ".join(map(str,bd))+"\n")
        fbd.flush()

    if paretofront is not None:
        paretofront.update(population)

    #print("Pareto Front: "+str(paretofront))

    k=15
    add_strategy="random"
    lambdaNov=6

    # pour les variantes utilisant NS
    archive=updateNovelty(population,population,None,k,add_strategy,lambdaNov)

    for ind in population:
        if (variant=="FIT+NS"):
            ind.fitness.values=(ind.fit,ind.novelty)
        elif (variant=="FIT"):
            ind.fitness.values=(ind.fit,)
        elif (variant=="NS"):
            ind.fitness.values=(ind.novelty,)

        #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))

    # Pour suivre la valeur de l'individu le plus proche de la sortie, quelle que soit l'objectif utilisé
    indexmin, valuemin = min(enumerate([i.fit for i in population]), key=operator.itemgetter(1))

    res_Gen = []
    res_allGen = []
    # Begin the generational process
    for gen in range(1, ngen + 1):
        if (gen%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)

        ## à faire: générer un ensemble de points à partir de la population courante:
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        offspring = list(map(toolbox.clone, offspring))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (fit, bd) in zip(invalid_ind, fitnesses_bds):
            #print("Fit: "+str(fit)) 
            if (variant=="FIT+NS"):
                ind.fitness.values=(fit,0)
            elif (variant=="FIT"):
                ind.fitness.values=(fit,)
            elif (variant=="NS"):
                ind.fitness.values=(0,)
            ind.fit = fit
            ind.bd = bd
            fbd.write(" ".join(map(str,bd))+"\n")
            fbd.flush()

        pq=population+offspring
        
        archive=updateNovelty(pq,offspring,archive,k,add_strategy,lambdaNov)

        for ind in pq:
            if (variant=="FIT+NS"):
                ind.fitness.values=(ind.fit,ind.novelty)
            elif (variant=="FIT"):
                ind.fitness.values=(ind.fit,)
            elif (variant=="NS"):
                ind.fitness.values=(ind.novelty,)

            #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))
        

        ## à faire: choisir la nouvelle population à partir de pq
        population[:] = toolbox.select( population + offspring )

        # Update the hall of fame with the generated individuals
        if paretofront is not None:
            paretofront.update(population)

        # Si vous voulez afficher les valeurs de fitness des différents
#         print("PQ: ")
#         for ind in pq:
#             print(str(ind.fitness.values), end=" ")
#         print("\n=======")
#         print("Population: ")
#         for ind in population:
#             print(str(ind.fitness.values), end=" ")
#         print("\n=======")
#         print("Pareto front: ")
#         for ind in paretofront:
#             print(str(ind.fitness.values), end=" ")
#         print("\n=======")

        # used to track the min value (useful in particular if using only novelty)
        indexmin, newvaluemin = min(enumerate([i.fit for i in pq]), key=operator.itemgetter(1))
        if (newvaluemin<valuemin):
            valuemin=newvaluemin
            
            print("Gen "+str(gen)+", new min ! min fit="+str(valuemin)+" index="+str(indexmin))
            eval_nn(pq[indexmin],render=True,name="gen%04d"%(gen))
        res_Gen.append(valuemin)
        res_allGen.append(newvaluemin)
    fbd.close()
    return population, None, paretofront, res_Gen, res_allGen

env = gym.make('FastsimSimpleNavigation-v0')

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(description='Launch maze navigation experiment.')
    parser.add_argument('--nb_gen', type=int, default=200,
                        help='number of generations')
    parser.add_argument('--mu', type=int, default=100,
                        help='population size')
    parser.add_argument('--lambda_', type=int, default=100,
                        help='number of individuals to generate')
    parser.add_argument('--res_dir', type=str, default="res",
                        help='basename of the directory in which to put the results')
    parser.add_argument('--variant', type=str, default="FIT", choices=['FIT', 'NS', 'FIT+NS'],
                        help='variant to consider')
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='number of hidden layers of the NN controller')
    parser.add_argument('--neurons_per_layer', type=int, default=10,
                        help='number of hidden layers of the NN controller')

    # Il vous est recommandé de gérer les différentes variantes avec cette variable. Les 3 valeurs possibles seront:
    # "FIT+NS": expérience multiobjectif avec la fitness et la nouveauté (NSGA-2)
    # "NS": nouveauté seule
    # "FIT": fitness seule
    # pour les variantes avec un seul objectif, il vous est cependant recommandé d'utiliser NSGA-2 car cela limitera la différence entre les variantes et cela 
    # vous fera gagner du temps pour la suite.
    
    args = parser.parse_args()
    print("Number of generations: "+str(args.nb_gen))
    ngen=args.nb_gen
    print("Population size: "+str(args.mu))
    mu=args.mu
    print("Number of offspring to generate: "+str(args.lambda_))
    lambda_=args.lambda_
    print("Variant: "+args.variant)
    variant=args.variant

    print("NN: %d hidden layers with %d neurons per layer"%(args.hidden_layers, args.neurons_per_layer))
    nn_size=[5,2,args.hidden_layers,args.neurons_per_layer]

    pop, logbook, paretofront, bestGen, allGen= launch_nsga2(mu=mu, lambda_=lambda_, ngen=ngen, variant=variant, nn_size=nn_size)
    #plot_pareto_front(paretofront, "Final pareto front")
    for i,p in enumerate(paretofront):
        print("Visualizing indiv "+str(i)+", fit="+str(p.fitness.values))
        eval_nn(p,render=True)
        
    plot_points_file('log/traj.log', title='traj')
    env.close()
    plt.close()
    plot_points_file('bd.log', title='bd')
    plt.plot(range(args.nb_gen),bestGen,label="Best Gen")
    plt.plot(range(args.nb_gen),allGen, label="All Gen")
    plt.legend()
    plt.savefig(args.variant+'.png')
    plt.show()
