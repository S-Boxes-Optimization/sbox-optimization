import random
import numpy as np
from numpy.linalg import inv


class MarkovChain:

    pass

    def __init__(self, a, P, tanamio=10):

        self._TAMANIO = tanamio
        self.A = a
        self.P = P
        
        np.random.seed(42)
        self.U = [np.random.uniform() for _ in range(self._TAMANIO)]

        self.X = [0 for _ in range(self._TAMANIO)]
        

    def g(self, u):
        cumulative_sum = np.cumsum(self.A)
        return np.where(u <= cumulative_sum)[0][0]


    def f(self, i, u):
        cumulative_sum = np.cumsum(self.P[i])
        return np.where(u <= cumulative_sum)[0][0]
    

    def correr(self):
        self.X[0] = self.g(self.U[0])

        for n in range(self._TAMANIO -1):  
            self.X[n+1] = self.f(self.X[n], self.U[n+1])

        print("X = ",self.X)
    
    @property
    def getX(self) -> list[str]: 
        return self.X
        
    @property
    def TAMANIO(self) -> int: 
        return self._TAMANIO
    
    @TAMANIO.setter
    def TAMANIO(self, newTamanio):
        self._TAMANIO = newTamanio  
       
        
    def details(self) -> str:
        return f'Markov chain [ {self.TAMANIO} steps and {len(self.A)} nodes]'
    
    def __str__ (self) -> str:
        return f'Markov chain [ {self.TAMANIO} steps and {len(self.A)} nodes]'
    
    """
    def g (self, u: float) -> int:
        suma = 0
        for i in range(self.SIZE):
            inferior = 0
            for k in range(i):
                inferior += self.A[k]   
            superior = inferior + self.A[i]    
            if (u > inferior) and (u <= superior): suma += i
        return suma

    def f (self, i: int, u:float) -> int:
        suma = 0
        for j in range(self.SIZE):
            inferior = 0
            for k in range(j):
                inferior += self.P[i][k]
            superior = inferior + self.P[i][j]
            if (u > inferior) and (u <= superior): suma += j
        return suma
    """

if __name__ == '__main__':
    
    print(0.2 + 0.1)
    cosas = MarkovChain([1],[[1]],1)
    print(cosas.TAMANIO)
    cosas.TAMANIO = 2
    print(cosas.TAMANIO)