import torch
import torch.nn as nn

### SIRD Model
class SIRDModel(nn.Module):
    '''SIRDModel 

    Forward model which implements the set of differential equations
    '''

    def __init__(self, 
            nn_model,
            population=1000, 
            
        ):
        super(SIRDModel, self).__init__()

        self.N=population
        
        self.nn_model = nn_model
        

    def forward(self, input_sird_values):
        '''
           input_sird_values : (past_n_days,4) 
        '''

        r = self.nn_model(input_sird_values)
                
        beta, gamma, delta = r[0], r[1], r[2]
        
        sir_params = {'beta':beta.item(), 'gamma':gamma.item(),'delta':delta.item()}
        
        
        S_t, I_t, R_t, D_t = input_sird_values[-1] 
        
        ## Calculate next day values
        S_t1 = S_t - beta * (S_t * I_t)/self.N    
        I_t1 = I_t + beta * (S_t * I_t)/self.N - gamma*I_t - delta * I_t        
        R_t1 = R_t + gamma * I_t
        D_t1 = D_t + delta * I_t

        out = torch.stack([ S_t1, I_t1, R_t1, D_t1])
        
        return out, sir_params
