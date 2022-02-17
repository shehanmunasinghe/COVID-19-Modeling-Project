from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import math
import sys

### Custom PyTorch Modules
class SimplifiedLinear(nn.Module):
    '''
    Implementation of a single layer of a Neural Network mode, without the batch size dimension in PyTorch 

    Reference: https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6
    '''
    
    def __init__(self, in_features, out_features, bias=True, weight_tensor = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_tensor = weight_tensor
        if not self.weight_tensor==None:
            self.weight = torch.nn.Parameter(self.weight_tensor)
        else: 
            self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.weight = self.weight * 0.01
        
    def reset_parameters(self):
        if self.weight_tensor == None:
            # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            torch.nn.init.constant_(self.weight, 0.001)
        

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
         
        if x.shape[-1] != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            
        #output = input @ self.weight.t() + self.bias
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

## Interface for NNs used with the forward model
class NNModel(nn.Module):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__()

        self.past_n_days = past_n_days
        self.activation = activation

        # Initialize different activation functions
        if self.activation == 'leaky_relu':
            self.leaky_relu = nn.LeakyReLU(0.01)    
        elif self.activation == 'elu':
            self.elu = nn.ELU()

    def act_fn(self,r, fn='sigmoid'):
        if self.activation == 'leaky_relu':
            r = self.leaky_relu(r)
        elif self.activation == 'elu':
            r = self.elu(r)
        elif self.activation =='sigmoid':
            r = torch.sigmoid(r)
            
        return r



### NNs used with the forward model
class NN_1Layer_Fixed(NNModel):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__(past_n_days,activation)

        self.nn1 = SimplifiedLinear(self.past_n_days * 4, 3, bias=True, 
            weight_tensor=torch.Tensor([
                [0.0001,0.0001,0.0001],
                [0.001,0.002,0.001],
                [0.0002,0.032,0.002],
                [0.003,0.0015,0.003],
            ]))

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''
        
        features = torch.flatten(input_sird_values)
                
        r = self.nn1(features)
        r = self.act_fn(r, self.activation)

        r = torch.clamp(r, min=0.0, max=1)
        
        if self.activation == 'elu':
            r = r + 1.0

        return r

class NN_1Layer(NNModel):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__(past_n_days,activation)

        self.nn1 = SimplifiedLinear(self.past_n_days * 4, 3, bias=True)

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''
        
        features = torch.flatten(input_sird_values)
                
        r = self.nn1(features)
        r = self.act_fn(r, self.activation)

        r = torch.clamp(r, min=0.0, max=1)
        
        if self.activation == 'elu':
            r = r + 1.0

        return r


class NN_2Layer_FixedFirstLayer(NNModel):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__(past_n_days,activation)

        self.nn1 = SimplifiedLinear(self.past_n_days * 4, 3, bias=True, 
            weight_tensor=torch.Tensor([
                [0.0001,0.0001,0.0001],
                [0.001,0.002,0.001],
                [0.0002,0.032,0.002],
                [0.003,0.0015,0.003],
            ]))
        self.nn2 = SimplifiedLinear(3, 3) 

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''
        
        features = torch.flatten(input_sird_values)
                
        r = self.nn1(features)
        r = self.act_fn(r, self.activation)
        
        r = self.nn2(r)
        r = self.act_fn(r, self.activation)


        r = torch.clamp(r, min=0.0, max=1)
        
        if self.activation == 'elu':
            r = r + 1.0

        return r



class NN_2Layer(NNModel):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__(past_n_days,activation)

        self.nn1 = SimplifiedLinear(self.past_n_days * 4, 3)
        self.nn2 = SimplifiedLinear(3, 3) 

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''
        
        features = torch.flatten(input_sird_values)
                
        r = self.nn1(features)
        r = self.act_fn(r, self.activation)
        
        r = self.nn2(r)
        r = self.act_fn(r, self.activation)


        r = torch.clamp(r, min=0.0, max=1)
        
        if self.activation == 'elu':
            r = r + 1.0

        return r

class NN_3Layer(NNModel):
    
    def __init__(self,
            past_n_days,
            activation
        ):
        super().__init__(past_n_days,activation)

        self.nn1 = SimplifiedLinear(self.past_n_days * 4, 3)
        self.nn2 = SimplifiedLinear(3, 3) 
        self.nn3 = SimplifiedLinear(3, 3) 

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''
        
        features = torch.flatten(input_sird_values)
                
        r = self.nn1(features)
        r = self.act_fn(r, self.activation)
        
        r = self.nn2(r)
        r = self.act_fn(r, self.activation)

        r = self.nn3(r)
        r = self.act_fn(r, self.activation)


        r = torch.clamp(r, min=0.0, max=1)
        
        if self.activation == 'elu':
            r = r + 1.0

        return r







def build_nn_model_from_config(conf):
    past_n_days=conf["MODEL"]["PAST_N_DAYS"]
    activation = conf["MODEL"]["ACTIVATION"]

    if conf['MODEL']['NN_MODEL'] == 'NN_1Layer':
        return NN_1Layer(past_n_days, activation)
    elif conf['MODEL']['NN_MODEL'] == 'NN_1Layer_Fixed':
        return NN_1Layer_Fixed(past_n_days, activation)
    elif conf['MODEL']['NN_MODEL'] == 'NN_2Layer_FixedFirstLayer':
        return NN_2Layer_FixedFirstLayer(past_n_days, activation)
    elif conf['MODEL']['NN_MODEL'] == 'NN_2Layer':
        return NN_2Layer(past_n_days, activation)
    elif conf['MODEL']['NN_MODEL'] == 'NN_3Layer':
        return NN_3Layer(past_n_days, activation)

    return None
