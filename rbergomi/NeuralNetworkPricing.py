# -*- coding: utf-8 -*-
import numpy as np
import json
from scipy.interpolate import CubicSpline

def elu(x):
# Implements the elu activation function.
    val = x
    val[x < 0] = np.exp(x[x < 0]) - 1
    return(val)

class NeuralNetwork:    
# Implements a basic neural network.
    def __init__(self,fileName):
    # Constructor, network is loaded from a json file.
        self.weights = []
        self.biases = []
        self.scaleMeanIn = []
        self.scaleStdIn = []
        self.scaleMeanOut = []
        self.scaleStdOut = []
        with open(fileName) as json_file:
            tmp = json.load(json_file)
        nLayers = int((len(tmp)-4)/2)
        for i in range(nLayers):
            self.weights.append(np.transpose(tmp[2*i]))
            self.biases.append(np.array(tmp[2*i+1]).reshape(-1,1))
        self.scaleMeanIn = np.array(tmp[-4]).reshape(-1,1)
        self.scaleStdIn = np.sqrt(np.array(tmp[-3]).reshape(-1,1))
        self.scaleMeanOut = np.array(tmp[-2]).reshape(-1,1)
        self.scaleStdOut = np.sqrt(np.array(tmp[-1]).reshape(-1,1))
        
    def Eval(self,x):
    # Evaluates the network.
        nLayers = len(self.weights)
        val = (x - self.scaleMeanIn)/self.scaleStdIn
        for i in range(0,nLayers - 1):
            val = elu(np.dot(self.weights[i],val) + self.biases[i])
        val = np.dot(self.weights[nLayers-1],val) + self.biases[nLayers-1]
        return self.scaleStdOut*val + self.scaleMeanOut
    
class NeuralNetworkPricer:    
# Implements a neural network pricer based on multiple sub networks.
    def __init__(self,contracts_folder,weights_folder,model_name):
    # Constructor.
        self.nn = []
        self.idx_in = []
        self.idx_out = []
        self.lb = []
        self.ub = []
        self.label = model_name
        
        # Load contract grid:
        self.T = np.loadtxt(contracts_folder + "\\expiries.txt").reshape(-1,1)
        self.k = np.loadtxt(contracts_folder + "\\logMoneyness.txt").reshape(-1,1)
        
        # Set the forward variance curve grid points (in case relevant):
        Txi = np.concatenate((np.arange(0.0025,0.0175,0.0025),
                              np.arange(0.02,0.14,0.02),
                              np.arange(0.16,1,0.12),
                              np.arange(1.25,2,0.25),
                              np.array([3])))
        
        # Basic naming of json files:
        json_files = ["_weights_1.json",
                      "_weights_2.json",
                      "_weights_3.json",
                      "_weights_4.json",
                      "_weights_5.json",
                      "_weights_6.json"]

        # Load each sub-network:
        idxOutStart = 0
        for i in range(len(json_files)):
            self.nn.append(NeuralNetwork(weights_folder + "\\" + model_name + json_files[i]))
            self.idx_in.append(np.arange(0,self.nn[i].scaleMeanIn.shape[0]))
            idxOutEnd = idxOutStart + self.nn[i].scaleMeanOut.shape[0]
            self.idx_out.append(np.arange(idxOutStart,idxOutEnd))
            idxOutStart = idxOutEnd
        
        # Set parameter bounds (and more):
        if (model_name == "rheston"):
            self.lb = np.concatenate((np.array([0,0.1,-1]),pow(0.05,2)*np.ones(27))).reshape(-1,1)
            self.ub = np.concatenate((np.array([0.5,1.25,0]),np.ones(27))).reshape(-1,1)
            self.Txi = Txi
        elif (model_name == "rbergomi"):
            self.lb = np.concatenate((np.array([0,0.75,-1]),pow(0.05,2)*np.ones(27))).reshape(-1,1)
            self.ub = np.concatenate((np.array([0.5,3.50,0]),np.ones(27))).reshape(-1,1)
            self.Txi = Txi
        elif (model_name == "rbergomi_extended"):
            self.lb = np.concatenate((np.array([0.75,-1,-0.5,-0.5]),pow(0.05,2)*np.ones(27))).reshape(-1,1)
            self.ub = np.concatenate((np.array([3.50,0,0.5,0.5]),np.ones(27))).reshape(-1,1)
            self.Txi = Txi
        elif (model_name == "heston"):
            self.lb = np.array([0,pow(0.05,2),0,-1,pow(0.05,2)]).reshape(-1,1)
            self.ub = np.array([25,1,10,0,1]).reshape(-1,1)
        else:
            raise Exception('NeuralNetworkPricer:__init__: Invalid model name.')
        
    def EvalInGrid(self,x):
    # Evaluates the model in the grid points only.
        # Check bounds:
        if (any(x < self.lb) or any(x > self.ub)):
            raise Exception('NeuralNetworkPricer:EvalInGrid: Parameter bounds are violated.')
        
        nNetworks = len(self.nn)
        nPts = self.k.shape[0]
        iv = np.zeros(nPts).reshape(-1,1)
        for i in range(0,nNetworks):
            iv[self.idx_out[i]] = self.nn[i].Eval(x[self.idx_in[i]])
        
        return(iv)
        
    def AreContractsInDomain(self,kq,Tq):
    # Checks if the contracts are within the supported domain.
        if not kq.shape == Tq.shape:
            raise Exception('NeuralNetworkPricer:AreContractsInDomain: Shape of input vectors are not the same.')
    
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        minTGrid = np.min(uniqTGrid)
        maxTGrid = np.max(uniqTGrid)
        idxValid = np.ones((len(kq), 1), dtype=bool)
        for i in range(0,len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] > maxTGrid or uniqT[i] < minTGrid:
                idxValid[idxT] = False
            else:
                if uniqT[i] == maxTGrid:
                    idxAbove = len(uniqTGrid) - 1
                else:
                    idxAbove = np.argmax(uniqTGrid > uniqT[i])
                idxBelow = idxAbove - 1
                idxGridBelow = self.T == uniqTGrid[idxBelow]
                idxGridAbove = self.T == uniqTGrid[idxAbove]
                idxValid[idxT] =   (kq[idxT] >= np.max([np.min(self.k[idxGridBelow]),np.min(self.k[idxGridAbove])])) \
                                 & (kq[idxT] <= np.min([np.max(self.k[idxGridBelow]),np.max(self.k[idxGridAbove])]))
        return(np.ravel(idxValid))        
        
    def Eval(self,x,kq,Tq):
    # Evaluates the model in arbitrary contracts (within the supported domain).
        ivGrid = self.EvalInGrid(x)
        if (not all(self.AreContractsInDomain(kq,Tq))):
            raise Exception('NeuralNetworkPricer:Eval: Some contracts violate the neural network domain.')

        ivGrid = self.EvalInGrid(x)
        nPts = kq.shape[0]
        iv = np.zeros((nPts,1))
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        maxTGrid = max(uniqTGrid)
        for i in range(0,len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] == maxTGrid:
                idxAbove = len(uniqTGrid) - 1
            else:
                idxAbove = np.argmax(uniqTGrid > uniqT[i])
            idxBelow = idxAbove - 1
            T_above = uniqTGrid[idxAbove]
            T_below = uniqTGrid[idxBelow]
            idxGridBelow = self.T == uniqTGrid[idxBelow]
            idxGridAbove = self.T == uniqTGrid[idxAbove]

            iv_below_grid = ivGrid[idxGridBelow]
            iv_above_grid = ivGrid[idxGridAbove]
            k_below_grid = self.k[idxGridBelow]
            k_above_grid = self.k[idxGridAbove]

            # Fit splines:
            idxSort_below = np.argsort(k_below_grid)
            idxSort_above = np.argsort(k_above_grid)
            spline_lower  = CubicSpline(k_below_grid[idxSort_below],iv_below_grid[idxSort_below],bc_type='natural')
            spline_upper  = CubicSpline(k_above_grid[idxSort_above],iv_above_grid[idxSort_above],bc_type='natural')

            # Evaluate spline:
            iv_below = spline_lower(kq[idxT])
            iv_above = spline_upper(kq[idxT])

            # Interpolate
            frac = (uniqT[i] - T_below) / (T_above - T_below)
            iv[idxT] = np.sqrt(((1-frac)*T_below*pow(iv_below,2) + frac*T_above*pow(iv_above,2))/uniqT[i])

        return(iv)
            