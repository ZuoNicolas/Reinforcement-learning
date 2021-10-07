#!/usr/local/bin/python

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from sample_generator import SampleGenerator
import matplotlib.pyplot as plt
import copy

class Main:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50

    def reset_batch(self):
        self.x_data = []
        self.y_data = []

    def make_nonlinear_batch_data(self):
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self):
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def approx_linear_batch(self):
        model = Line(self.batch_size)

        self.make_linear_batch_data()
        
        start = time.process_time()
        model.train(self.x_data, self.y_data)
        print("LLS time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

        start = time.process_time()
        model.train_from_stats(self.x_data, self.y_data)
        print("LLS from scipy stats:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)
        
        coeficient=0.0001
        for i in range(1):
            start = time.process_time()
            model.train_regularized(self.x_data, self.y_data, coef=coeficient)
            print("coeficient :",coeficient)
            print("regularized LLS :", time.process_time() - start)
            model.plot(self.x_data, self.y_data,coeficient)
            coeficient*=10
        
    def approx_rbfn_batch(self):
        model = RBFN(nb_features=500)
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_ls(self.x_data, self.y_data)
        print("RBFN LS time:", time.process_time() - start," Features =",model.nb_features)
        model.plot(self.x_data, self.y_data, model.nb_features)

        start = time.process_time()
        model.train_ls2(self.x_data, self.y_data)
        print("RBFN LS2 time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data,model.nb_features)

    def approx_rbfn_iterative(self):
        max_iter = 100
        alpha=0.5
        it=1

        temps=0
        for x in range(it):
            model = RBFN(nb_features=20)
            #model1= RBFN(nb_features=20)
            #model2= RBFN(nb_features=20)
            start = time.process_time()
            # Generate a batch of data and store it
            self.reset_batch()
            g = SampleGenerator()

            for i in range(max_iter):
                # Draw a random sample on the interval [0,1]
                x = np.random.random()
                y = g.generate_non_linear_samples(x)
                self.x_data.append(x)
                self.y_data.append(y)
                
                # Comment the ones you don't want to use
                model.train_gd(x, y, alpha)
                #model1.train_rls(x, y)
                #model2.train_rls_sherman_morrison(x, y)
            temps+=time.process_time() - start
        print("RBFN Incr time:", temps/it, "Features =",model.nb_features)
        model.plot(self.x_data, self.y_data, model.nb_features,max_iter,alpha,"gd")
        #model1.plot(self.x_data, self.y_data, model1.nb_features,max_iter,None,"rls")
        #model2.plot(self.x_data, self.y_data, model2.nb_features,max_iter,None,"sherman")
        """
        it=1
        temps=0
        for x in range(it):
            model = LWR(nb_features=20)
    
            start = time.process_time()
            model.train_lwls(self.x_data, self.y_data)
            temps+=time.process_time() - start
        print("LWR time:", temps/it)
        model.plot(self.x_data, self.y_data,model.nb_features,self.batch_size,"Lwr")
        """

    def approx_lwr_batch(self):
        it=1
        temps=0
        for x in range(it):
            model = LWR(nb_features=20)
            self.make_nonlinear_batch_data()
    
            start = time.process_time()
            model.train_lwls(self.x_data, self.y_data)
            temps+=time.process_time() - start
        print("LWR time:", temps/it)
        model.plot(self.x_data, self.y_data)

if __name__ == '__main__':
    m = Main()
    m.approx_linear_batch()
    #m.approx_rbfn_batch()
    #m.approx_rbfn_iterative()
    #m.approx_lwr_batch()
