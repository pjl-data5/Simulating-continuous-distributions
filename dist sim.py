# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:39:56 2020

@author: p*****
"""

# Importing relevant libraries
import numpy as np
from numpy.random import uniform
import math
import matplotlib.pyplot as plt
import seaborn as sns 


# BEGIN
# Class "continuous_distributions" defines various simulated distributions

class continuous_distributions():
    
    def __uniform(self, **kwargs):
        """
        simulating uniform rvs using numpy
        :param size: array length of rvs
        :return: array of uniform rvs 
        """ 
        return uniform(**kwargs)

    
    def exponential(self, beta,size):
        """
        Simulating expoential rvs using 
        the inverse probability transfom method 
        :params beta: shape 
        :param size: array length of rvs 
        :return: array of exponential rvs 
        """ 
        u = self.__uniform(size=size) 
        return -np.log(u)/beta
    
    # The uniform and exponential classes above were given
    
    def normal_distr(self, mu, sigma, size): 
        get_norm = np.random.normal(mu, sigma**2, size) 
        return get_norm   
    
    def gamma_distr(self, beta, n, size):
        """
        The sum of exponential distributions with parameter beta
        will yield the gamma distribution with parameters n and beta
        """
        g = 0
        g += self.exponential(beta, size)
        return g
    
    def chisq_distr(self, alpha, beta, size):
        """
        If Y ~Gamma(alpha,beta) then (2*Y/alpha) yields a 
        Chi squared distribution with parameter beta
        """
        y = self.gamma_distr(alpha, beta, size)
        chi = ((2*y)/alpha)
        return chi
    
    def beta_distr(self, alpha1, alpha2, beta, size):
        """
        Can be obtained with two gamma distributions and a bit
        of arithmetic
        """
        g1 = self.gamma_distr(alpha1, beta, size)
        g2 = self.gamma_distr(alpha2, beta, size)
        b = (g1/(g1+g2))
        return b
    
    def cauchy_distr(self, mu, std, size):
        """
        Can obtain cauchy distribution from normal distribution:
        """
        a = continuous_distributions().normal_distr(mu, std, size) 
        b = continuous_distributions().normal_distr(mu, std, size) 
        c = a/b      # a divided by b yields c ~ Cauchy(0,1) as a and b are both N(0,1)
        return c   

    def F_distr(self, d1, d2, b1, b2, size):
        """
        Dividing two Chi-Squared distributions will yield the F distribution
        """
        x1 = self.chisq_distr(d1, b1, size)
        x2 = self.chisq_distr(d2, b1, size)
        f = x1/x2
        return f
    
    def frechet_distr(self, alpha, s, m, size):
        """
        If X ~ Unif(0,1) then a Frechet distribution can be obtained the 
        transformation technique
        """
        X = self.__uniform(size=size) #given
        fre = m+s*(-np.log(X))**(-1/alpha)
        return fre
    
    def weibull_distr(self, lamb, k, size):
        """
        transformation technique from uniform again
        """
        X = self.__uniform(size=size)
        w = lamb*(-np.log(X))**(1/k)
        return w
    
    def rayleigh_distr(self, lamb, k, size):
        """
        can be easily simulated from the weibull distribution
        """
        r = self.weibull_distr(lamb, k, size)
        return r
    
    def lomax_distr(self, beta, lamb, size):
        """
        If Y ~ Exp(beta) then M*exp(Y) yields a paretoII(M, beta)
        Distribution from which lomax can be obtained. The paretoII distribution
        can be simulated from the exponential.
        """
        y = self.exponential(beta, size)
        pareto = beta*np.exp(y)
        l = -(y - pareto)
        return l
# END      
        
    
    # Exponential
sns.distplot(continuous_distributions().exponential(beta=2, size=10000), bins=35, color="g")
plt.title("Exponential Distribution", fontsize = 15)
plt.show()

# Normal
sns.distplot(continuous_distributions().normal_distr(size=10000, mu=0, sigma=1), color="r")
plt.title("Normal Distribution", fontsize = 15)
plt.show()

# Gamma 
sns.distplot(continuous_distributions().gamma_distr(size=10000, beta=2, n=10), color="g")
plt.title("Gamma Distribution", fontsize = 15)
plt.show()

# Chi-Squared
sns.distplot(continuous_distributions().chisq_distr(size=10000, beta=2, alpha=2), color="g")
plt.title("Chi-Squared Distribution", fontsize = 15)
plt.show()

# Beta
sns.distplot(continuous_distributions().beta_distr(size=10000, beta=1, alpha1=2, alpha2=1), color='g')
plt.title("Beta Distribution", fontsize = 15)
plt.show()

# Cauchy
sns.distplot(continuous_distributions().cauchy_distr(size=100, mu=0, std=1), color = "orange")
plt.title("Cauchy Distribution", fontsize = 15)
plt.show()

# F
sns.distplot(continuous_distributions().F_distr(size=100, b1=2, d1=2, b2=1, d2=1), color='b')
plt.title(" F Distribution", fontsize = 15)
plt.show()

# Frechet
sns.distplot(continuous_distributions().frechet_distr(size=10000, m=3, s=2, alpha=3), color = 'r')
plt.title("Frechet Distribution", fontsize = 15)
plt.show()

# Weibull
sns.distplot(continuous_distributions().weibull_distr(size=1000, lamb=4, k=1), color = 'r')
plt.title("Weibull Distribution", fontsize = 15)
plt.show()

# Rayleigh 
sns.distplot(continuous_distributions().rayleigh_distr(size=10000, lamb=2, k=2), color='purple')
plt.title("Rayleigh Distribution", fontsize = 15)
plt.show()

# Lomax
sns.distplot(continuous_distributions().lomax_distr(size=10000, beta=3, lamb=2))
plt.title("Lomax Distribution", fontsize = 15)
plt.show()

    
    
    
    
    
    
    
    
    
