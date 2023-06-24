import pandas as pd
import numpy as np
import common_math
from common_math import activationMap, activationGradMap
from numpy import matlib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from scipy import optimize

class Layer:
    def __init__(self, **kwargs) -> None:
        activation = kwargs.get('activation')
        self.nr_units = len(activation) if isinstance(activation, list) else kwargs.get('nr_units', 0)
        self.nr_inputs = kwargs.get('nr_inputs')
        self.W = matlib.rand(self.nr_units, self.nr_inputs)-0.5
        self.b = matlib.zeros((self.nr_units,1))
        
        if(isinstance(activation,str)):
            activation_fn_list = [activationMap[activation],]*self.nr_units
            activation_grad_list = [activationGradMap[activation],]*self.nr_units
        elif(isinstance(activation,list)):
            activation_fn_list = [activationMap[x] for x in activation]
            activation_grad_list = [activationGradMap[x] for x in activation]
        else:
            raise Exception("Invalid activation")
        
        self.activation_fn_list = activation_fn_list
        self.activation_grad_list = activation_grad_list

        # Activation function vector assumed to apply element-wise. Can be revised later.
        self.activation_fn = lambda x: [activation_fn_list[i](np.array(x)[i].item()) for i in range(len(activation_fn_list))]
        self.activation_grad_fn = lambda z: np.diagflat([activation_grad_list[i](np.array(z)[i].item()) for i in range(len(activation_grad_list))])

    def forward_pass(self, input):
        self.Z = matlib.matmul(self.W, input) + self.b
        self.fwd_input = input
        data = pd.DataFrame(self.Z) # Convert Z to dataframe - each sample forms a column.
        # Apply the activation functions to each sample.
        A = data.apply(self.activation_fn, axis=0)
        self.A = np.matrix(A.values)
        return self.A

    def backward_prop(self, **kwargs):
        # Compute the derivatives in the backward propagation.
        # Assume we are given the derivative vector dJ/dA for this layer.
        # Also, we get the A vector from downstream to compute dW
        self.dA = kwargs.get('dA') # 2D Matrix if multiple samples.
        df_z = pd.DataFrame(self.Z) # Each column represents a sample.
        ds_dG = df_z.apply(self.activation_grad_fn, axis=0, result_type='reduce') # Apply to each sample, each of which is now a row of a 2D matrix.
        ds_dA = pd.DataFrame(self.dA).apply(lambda x: np.matrix(x).T, axis=0, result_type='reduce')
        ds_dZ = ds_dG * ds_dA # Each row is now a vector of dZ (one per sample)
        # Transform to a matrix so each column is dZ for one sample
        self.dZ = ds_dZ.apply(lambda x: pd.Series(x.tolist())).applymap(lambda x: x[0]).transpose().to_numpy()
        # Compute dW; the A provided is a matrix, each column being one sample.
        ds_Aprev = pd.DataFrame(kwargs.get('A')).apply(lambda x: np.matrix(x), axis=0, result_type='reduce')
        ds_dW = ds_dZ * ds_Aprev
        # Each row of ds_dW is a matrix, so it would have to be converted to a 3D matrix for numpy representation (1 per sample)
        self.ds_dW = ds_dW
        self.db = self.dZ
        self.dA_ds = np.matmul(self.W.T, self.dZ)
        return self.dA_ds


class NeuralNet:
    def __init__(self, **kwargs) -> None:
        self.layers = kwargs.get('layers',[])
        self.loss = kwargs.get('loss')
        self.loss_grad = kwargs.get('loss_grad')
        self.loss_iter = []
    
    def one_pass_update(self, x, y, step_size=0.01):
        grads = []
        # First do a forward pass on the input to get final output.
        network_output = self.forward_pass(x)
        # Compute the cost, and the derivative for final output (each column is one sample)
        cost = self.loss(y, network_output.T)
        dA = self.loss_grad(y, network_output) # First gradient.
        # Do a backward pass.
        self.backward_prop(dA = dA)

        # Update the weights for each layer (first layer is just an identity matrix)
        for layer in self.layers[1:]:
            layer.W = layer.W - step_size * layer.ds_dW.mean()
            layer.b = layer.b - step_size * np.mean(layer.db, axis=1)
            grads += [ii.item() for ii in layer.ds_dW.mean().reshape(layer.W.size,1)]
            grads += [ii.item() for ii in np.mean(layer.db, axis=1).reshape(layer.b.size,1)]
        return cost, grads
    
    def __add__(self, obj2):
        # Missing checks on whether the two networks are compatible.
        if(isinstance(obj2, NeuralNet)):
            return NeuralNet(layers = self.layers + obj2.layers, loss = self.loss, loss_grad = self.loss_grad)
        elif(isinstance(obj2, Layer)):
            return NeuralNet(layers = self.layers + [obj2,])

    def forward_pass(self, x):
        curr_input = x
        for layer in self.layers:
            curr_input = layer.forward_pass(curr_input)
        return curr_input # The final output.
    
    def backward_prop(self, **kwargs):
        dA = kwargs.get('dA')
        for indx in reversed(range(1, len(self.layers))):
            # 0th layer is an input layer that just provides an identity weight. 
            dA = self.layers[indx].backward_prop(dA = dA, A = self.layers[indx-1].A)

    def networkParamsUpdate(self, params): # Update the model parameters.
        running_count = 0
        for layer in self.layers[1:]:
            W_new = np.matrix(params[running_count:running_count+layer.W.size])
            running_count += W_new.size
            b_new = np.matrix(params[running_count:running_count+layer.b.size])
            running_count += b_new.size

            layer.W = W_new.reshape(layer.W.shape)
            layer.b = b_new.reshape(layer.b.shape)

    def fit(self, x, y, max_iterations):
        def optimFn(p):
            self.networkParamsUpdate(p)
            return self.one_pass_update(x,y,0.0)
        
        cost, grad = self.one_pass_update(x,y,0.0)
        self.optim_res = optimize.minimize(
            optimFn, np.random.randn(len(grad)), jac=True, 
            options={'maxiter':max_iterations}
            )
        


if __name__ == "__main__":
    nn_case = 'iris'

    if(nn_case == 'layer_test1'):
        nn = Layer(nr_inputs = 3, activation=['relu', 'sigmoid'], nr_units = 2)
        m = nn.forward_pass(input=np.mat('1;0;0'))
        n = nn.forward_pass(input=np.mat('0;1;0'))
        o = nn.forward_pass(input=np.mat('0;0;1'))
        output = matlib.bmat([m, n, o])
        print(output)
        print(nn.W)
    if(nn_case == 'layer_test2'):
        nn = Layer(nr_inputs = 3, activation=['relu', 'sigmoid'], nr_units = 2)
        m = nn.forward_pass(input=np.mat('1 0;0 1;1 1'))
        print(m)

    if(nn_case == 'iris'):
        data = load_iris()
        x = data.data[0:100,:].T
        y = data.target[0:100]

        # Build the neural network.
        layer0 = Layer(nr_inputs=4, activation='identity',nr_units=4)
        layer0.W = np.eye(layer0.W.shape[0])
        layer0.b = layer0.b*0.0
        layer1 = Layer(nr_inputs=4, activation='sigmoid',nr_units=1)
        nn = NeuralNet(layers=[layer0,layer1], loss=common_math.log_loss, loss_grad = common_math.log_loss_grad)
        nn.fit(x=x, y=y, max_iterations=1000)
        print(nn.layers[1].W)
        print(nn.layers[1].b)

        sk_mdl = LogisticRegression(penalty='none', fit_intercept=True)
        sk_mdl.fit(x.T, y)
        sk_mdl.get_params()
        print(nn)
        