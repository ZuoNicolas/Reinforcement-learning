import numpy as np
import matplotlib.pyplot as plt
from gaussians import Gaussians


class RBFN(Gaussians):
    def __init__(self, nb_features):
        super().__init__(nb_features)
        self.theta = np.random.random(self.nb_features)
        self.a = np.zeros(shape=(self.nb_features, self.nb_features))
        self.a_inv = np.matrix(np.identity(self.nb_features))
        self.b = np.zeros(self.nb_features)

    def f(self, x, theta=None):
        """
        Get the FA output for a given input vector
    
        :param x: A vector of dependent variables of size N
        :param theta: A vector of coefficients to apply to the features. 
        :If left blank the method will default to using the trained thetas in self.theta.
        
        :returns: A vector of function approximator outputs with size nb_features
        """
        if not hasattr(theta, "__len__"):
            theta = self.theta
        value = np.dot(self.phi_output(x).transpose(), theta.transpose())
        return value

    def feature(self, x, idx):
        """
         Get the output of the idx^th feature for a given input vector
         This is function f() considering only one feature
         Used mainly for plotting the features

         :param x: A vector of dependent variables of size N
         :param idx: index of the feature

         :returns: the value of the feature for x
         """
        phi = self.phi_output(x)
        return phi[idx] * self.theta[idx]

    # ----------------------#
    # # Training Algorithms ##
    # ----------------------#

    # ------ batch least squares (projection approach) ---------
    def train_ls(self, x_data, y_data):
        x = np.array(x_data)
        y = np.array(y_data)
        X = self.phi_output(x)


        self.theta=np.linalg.inv(X@X.transpose())@X@y
        print("\n-------- Methode 1, avec boucle: --------")
        print("Theta =", self.theta)
        print("-----------------------------------------\n")
        
        #TODO: Fill this

    # ------ batch least squares (calculation approach) ---------
    def train_ls2(self, x_data, y_data):
        a = np.zeros(shape=(self.nb_features, self.nb_features))
        b = np.zeros(self.nb_features)
        
        #TODO: Fill this
        
        #Le reshape pour faire apparaitre le 1, car np initialise en (50,), 
        #se qui pose probleme apres quand je fait des produits de matrice
        y = np.array(y_data).reshape(len(y_data),1) 
        x=[]    #stoackage pour a
        x2=[]   #stockage pour b
        
        #Recuperation des données
        for i in range(len(x_data)):
            phi=self.phi_output(x_data[i])
            x.append((phi@phi.T))
            x2.append(phi@y[i].T)#J'ai mis y[i].T pour avoir une matrice : (10,1)*(1,50) = (10,50)
        
        #Calcul somme des a
        for i in range(self.nb_features):
            for j in range(self.nb_features):
                sum=0
                for v in x:
                    sum+=v[i][j]
                    
                a[i][j]=sum
        
        #Calcul somme des b
        for i in range(self.nb_features):
            sum=0
            for v in x2:
                sum+=v[i]
            b[i]=sum

        self.a=a
        self.b=b
        self.theta=np.linalg.solve(a,b)
        
        print("\n-------- Methode 2, avec boucle: --------")
        print("Theta =", self.theta)
        print("-----------------------------------------\n")
        
        
    # -------- gradient descent -----------------
    def train_gd(self, x, y, alpha):
        
        self.theta=self.theta+alpha*(y-(self.phi_output(x).transpose()@self.theta))@self.phi_output(x).transpose()
        #TODO: Fill this

    # -------- recursive least squares -----------------
    def train_rls(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.dot(phi, phi.transpose())
        self.b = self.b + y * phi.transpose()[0]

        result = np.dot(np.linalg.pinv(self.a), self.b)
        self.theta = np.array(result)

    # -------- recursive least squares (other version) -----------------
    def train_rls2(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.outer(phi,phi)
        self.b = self.b + y * phi.transpose()[0]

        self.theta = np.dot(np.linalg.pinv(self.a), self.b)

    # -------- recursive least squares with Sherman-Morrison -----------------
    def train_rls_sherman_morrison(self, x, y):
        u = self.phi_output(x)
        v = self.phi_output(x).transpose()

        value = (v * self.a_inv * u)[0, 0]
        tmp_mat = self.a_inv * np.dot(u, v)* self.a_inv

        self.a_inv = self.a_inv - (1.0 / (1 + value)) * tmp_mat
        self.b = self.b + y * u.transpose()[0]

        result = np.dot(self.a_inv, self.b)
        self.theta = np.array(result)[0]

    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data,features=None,MaxIter=None,alpha=None,name=""):
        xs = np.linspace(0.0, 1.0, 1000)
        z = []
        for i in xs:
            z.append(self.f(i))

        z2 = []
        for i in range(self.nb_features):
            temp = []
            for j in xs:
                temp.append(self.feature(j, i))
            z2.append(temp)
        
        plt.title(name+" Features :"+str(features)+" MaxIter :"+str(MaxIter)+" Alpha :"+str(alpha))
        plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
        plt.plot(xs, z, lw=3, color='red')
        for i in range(self.nb_features):
            plt.plot(xs, z2[i])
        plt.show()
