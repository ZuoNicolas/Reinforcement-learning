from deap import creator, gp, base, tools, algorithms
import operator
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import pickle
import datetime
import sys
import os
import pygraphviz as pgv


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def ru():
    return random.uniform(-1,1)

def load_data(file):
    fdata = open(file)
    state=[]
    action=[]

    for l in fdata.readlines():
        d=l.split(" ")
        #print("d: "+str(d))
        fd = list(map(float, d))
        action.append([fd[0]])
        state.append(fd[1:])

    fdata.close()

    return state,action



def evalSymbReg(individual, input, output, nb_obj=1):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    err=0
    for i in range(len(input)):
        res=func(*input[i])+np.random.normal(0,noise)
        err+=(output[i]-res)**2
    return err,
    ## à compléter: calcule l'erreur entre l'entrée et la sortie désirée 
    ## remarque: l'entrée est un verteur, vous pouvez appeler la fonction func de la façon suivante: func(*input[i]) pour tester la i-ième entrée
    ## lorsque la fonction renverra l'erreur seule, faire un renvoi sous la forme 'return err,' (ne pas oublier la virgule car DEAP attend un tuple)
    


def load_data(file, state_dim=0):
    # (option) fonction à utiliser pour lire des données que vous aurez sauvez depuis un simulateur, par exemple le cartpole
    fdata = open(file)
    state=[]
    action=[]

    for l in fdata.readlines():
        d=l.split(" ")
        #print("d: "+str(d))
        fd = list(map(float, d))
        action.append([fd[0]])
        state.append(fd[1:])

    fdata.close()

    input=[]
    output=[]

    for i in range(len(state)-1):
        input.append(state[i]+action[i])
        output.append(state[i+1][state_dim])

    return input, output



if (__name__ == "__main__"):

    random.seed()


    parser = argparse.ArgumentParser(description='Launch symbolic regression run.')
    parser.add_argument('--nb_gen', type=int, default=100,
                        help='number of generations')
    parser.add_argument('--mu', type=int, default=400,
                        help='population size')
    parser.add_argument('--lambda_', type=int, default=400,
                        help='number of individuals to generate')
    parser.add_argument('--res_dir', type=str, default="res",
                        help='basename of the directory in which to put the results')
    parser.add_argument('--selection', type=str, default="elitist", choices=['elitist', 'double_tournament', 'nsga2'],
                        help='selection scheme')
    parser.add_argument('--problem', type=str, default="f1", choices=['f1', 'f2', 'cartpole'], ## à adapter aux problèmes que vous aurez définis
                        help='function to fit')
    parser.add_argument('--state_dim', type=int, default=0,
                        help='dimension to fit (for problems with multiple dimensions, e.g. cartpole)')
    parser.add_argument('--noise', type=float, default="0.",
                        help='noise added to the model to fit (gaussian, mean=0, sigma=noise)')
    
    args = parser.parse_args()
    print("Number of generations: "+str(args.nb_gen))
    ngen=args.nb_gen
    print("Population size: "+str(args.mu))
    mu=args.mu
    print("Number of offspring to generate: "+str(args.lambda_))
    lambda_=args.lambda_
    print("Selection scheme: "+str(args.selection))
    sel=args.selection
    if (sel=="nsga2"):
        nb_obj=2
    else:
        nb_obj=1
    print("Basename of the results dir: "+str(args.res_dir))
    name=args.res_dir

    print("State dim: "+str(args.state_dim)+" (ignored on most problems)")
    state_dim=args.state_dim


    noise=args.noise

    problem=args.problem

    if (problem=="f1"):
        nb_dim=2
        def f(x):
            return x[0]* x[1]+np.cos(x[0])
        input_training=[(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for i in range(30)]
        
        output_training=[f(x) for x in input_training]

        input_testing=[(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for i in range(30)]
        ## à compléter
        output_testing=[f(x) for x in input_testing]
        name_vars={"ARG0": "x1", "ARG1": "x2"}
    elif (problem=="f2"):
        ## à compléter
        nb_dim=3
        def f(x):
            return x[0]**2+ x[1]**2 +x[2]**2

        input_training=[(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0),random.uniform(0.0, 1.0)) for i in range(30)]
        
        output_training=[f(x) for x in input_training]

        input_testing=[(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0),random.uniform(0.0, 1.0)) for i in range(30)]
        ## à compléter
        output_testing=[f(x) for x in input_testing]
        name_vars={"ARG0": "x1", "ARG1": "x2","ARG2": "x3"}
        pass
    elif (problem=="cartpole"):
        ## à compléter (option)
        pass

    ## à compléter: créer pset, l'ensemble de primitives sur lequel la programmation génétique va s'appuyer, cf gp.PrimitiveSet
    pset=gp.PrimitiveSet("MAIN",nb_dim)
    pset.addPrimitive(operator.add,2)
    pset.addPrimitive(operator.sub,2)
    pset.addPrimitive(operator.mul,2)
    pset.addPrimitive(protectedDiv,2,"div")
    pset.addPrimitive(np.cos,1,"cos")
    pset.addPrimitive(np.sin,1,"sin")
    pset.addTerminal(1)
    pset.addEphemeralConstant("ru",ru) 
    pset.renameArguments(**name_vars)

    ## à compléter, définir les fitness (prendre en compte nb_obj) et Individual, cf creator dans DEAP
    if(nb_obj==1):
        creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
        creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMin,pset=pset)
    else:
        creator.create("FitnessMin",base.Fitness,weights=(-1.0,-1.0))
        creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMin,pset=pset)
    # valeur typiques de mutation et de croisement en GP
    cxpb=0.5
    mutpb=0.1


    d=datetime.datetime.today()
    if(name!=""):
        sep="_"
    else:
        sep=""
    run_name=name+"_"+d.strftime(name+sep+"%Y_%m_%d-%H-%M-%S")
    try:
        os.makedirs(run_name)
    except OSError:
        pass
    print("Putting the results in : "+run_name)

        
    ## à compléter: définir la toolbox à utiliser pour la GP
    ## notamment, la fonction d'évaluation (faire appel à evalSymbReg) et la stratégie de sélection 
    ## en fonction de la valeur de la variable sel.
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    toolbox.register("compile",gp.compile, pset=pset)
    toolbox.register("mate",gp.cxOnePoint)
    toolbox.register("mutate",gp.mutShrink)
    toolbox.register("evaluate",evalSymbReg,input=input_training,output=output_training)
    if(sel=="elitist"):
        toolbox.register("select",tools.selBest,fit_attr="fitness")
    if(sel=="double_tournament"):
        toolbox.register("select",tools.selDoubleTournament,fitness_size=10,parsimony_size=2,fitness_first=True,fit_attr="fitness")
    if(sel=="nsga2"):
        toolbox.register("select",tools.selNSGA2,nd='standard')
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    
    ## à compléter, initialisation de la population: 
    ## pop = ...

    if (nb_obj==1):
        print("Hall-of-fame: best solution")
        hof = tools.HallOfFame(1)
    else:
        print("Hall-of-fame: Pareto front")
        hof=tools.ParetoFront()

    ## à compléter: faire appel à l'algorithme évolutionniste algorithms.eaMuPlusLambda
    population=toolbox.population(n=mu)
    pop, log = algorithms.eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, mstats, hof, verbose=True)

    
    # Tracés de quelques courbes de résultats
    avg,dmin,dmax=log.chapters['fitness'].select("avg", "min", "max")
    gen=log.select("gen")

    plt.figure()
    plt.yscale("log")
    plt.plot(gen[1:],dmin[1:])
    plt.title("Minimum error")
    plt.savefig(run_name+"/min_error_gen%d.pdf"%(ngen))
    
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average error")
    plt.savefig(run_name+"/avg_error_gen%d.pdf"%(ngen))

    avg,dmin,dmax=log.chapters['size'].select("avg", "min", "max")
    gen=log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average size")
    plt.savefig(run_name+"/avg_size_gen%d.pdf"%(ngen))

    avg,dmin,dmax=log.chapters['height'].select("avg", "min", "max")
    gen=log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average height")
    plt.savefig(run_name+"/avg_height_gen%d.pdf"%(ngen))
        

    # Affichage de quelques résultats sur les meilleurs individus générés ainsi que de l'arbre généré
    # Remarque: pour voir l'affichage de l'arbre, il faut installer pygraphviz (pip install pygraphviz)
    # sur certains OS, il faut au préalable installer graphviz
    for i,ind in enumerate(hof):
        print("=========")
        print("HOF %d, len=%d"%(i,len(ind)))
        print("Error on the training dataset: %f"%(evalSymbReg(ind, input_training, output_training, nb_obj=1)))
        print("Error on the testing dataset: %f"%(evalSymbReg(ind, input_testing, output_testing, nb_obj=1)))

        nodes, edges, labels = gp.graph(ind)


        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for ni in nodes:
            n = g.get_node(ni)
            n.attr["label"] = labels[ni]


        g.draw(run_name+"/hof%d_tree_gen%d.pdf"%(i,ngen))
