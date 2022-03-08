import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


def build_optimizer_from_config(conf, model):
    lr=float(conf["OPTIMIZATION"]["LR"])

    if conf['OPTIMIZATION']['OPTIMIZER']=='Adam':
        return torch.optim.Adam( model.parameters(), lr=lr ) 

    return None

def push_to_tensor_queue(tensor, x):
    return torch.cat((tensor[1:], x ))


## Training Loop
def do_training(
        optimizer,
        num_days_iter, 
        trainset, 
        past_n_days,
        model2,
        criterion,

        epochs=10
    ):

    gt = torch.cat([ trainset[:].squeeze()]) # shape (5,1,4) ->(5,4)

    for epoch in range(epochs):
        
        optimizer.zero_grad()
        loss = 0

        out_pred2 = []
        for i in range(0,num_days_iter):
            if i==0:
                x_n_days = gt[i:i+past_n_days]
            else:
                x_n_days = push_to_tensor_queue(x_n_days, torch.unsqueeze(pred_y, 0))

            pred_y, _ = model2(x_n_days)    
            out_pred2.append(pred_y)

        out_pred2 = torch.stack(out_pred2)

        gt_y = gt[past_n_days:]

        loss = criterion(out_pred2 , gt_y )  

        loss.backward()
        optimizer.step()

        if epoch%1 ==0:
            print('epoch=%d  loss=%f'%(epoch,loss.item()))

def do_inference(
    model2, 
    dataset,
    num_days_iter,
    past_n_days
    ):
    with torch.no_grad():
        # Run the model forward pass

        # x_in = dset[0] # (1,4)

        # out_pred = []
        # sir_params_list = []
        # for i in range(num_days_iter):
        #     pred_y, sir_params = model2(x_in) #(4)
        #     pred_y = torch.unsqueeze(pred_y, 0)
        #     x_in = pred_y
        #     out_pred.append(pred_y)
        #     sir_params_list.append(sir_params)
        #     # x_in = pred_y
        # out_pred = torch.stack(out_pred)


        # x_in = dset[0:past_n_days] # (7,4)

        dset = torch.cat([ dataset[:].squeeze()]) # shape (5,1,4) ->(5,4)

        out_pred = []
        sir_params_list = []
        for i in range(num_days_iter):
            if i==0:
                x_in = dset[i:i+past_n_days] # (7,4)
                # x_in = torch.squeeze(x_in, 1)   # (7,1,4) -> (7,4)
            else:
                x_in = push_to_tensor_queue(x_in, torch.unsqueeze(pred_y, 0))
            
            # print('x_in.shape',x_in.shape)
            pred_y, sir_params = model2(x_in) #(4)
            # pred_y = torch.unsqueeze(pred_y, 0)
            out_pred.append(pred_y)
            sir_params_list.append(sir_params)
        out_pred = torch.stack(out_pred)

        # print('out_pred.shape',out_pred.shape) #[num_days_iter,4]

        # Predicted SIR params and values
        d_sir_params = {"beta":[],"gamma":[],"delta":[]}
        d_pred_sir_values = {"S":[],"I":[],"R":[],"D":[]}
        for i in range(len(sir_params_list)) :
            sir_params = sir_params_list[i]
            d_sir_params["beta"].append((np.abs(sir_params["beta"])))
            d_sir_params["gamma"].append((np.abs(sir_params["gamma"])))
            d_sir_params["delta"].append((np.abs(sir_params["delta"])))

            d_pred_sir_values["S"].append(out_pred[i][0].detach().numpy())
            d_pred_sir_values["I"].append(out_pred[i][1].detach().numpy())
            d_pred_sir_values["R"].append(out_pred[i][2].detach().numpy())
            d_pred_sir_values["D"].append(out_pred[i][3].detach().numpy())
        v_x = [i for i in range(len(d_sir_params["beta"]))]

        # Ground truth SIR values
        d_gt_sir_values = {"S":[],"I":[],"R":[],"D":[]}
        for i in range(past_n_days,len(dset)) :
            d_gt_sir_values["S"].append(dset[i][0].detach().numpy())
            d_gt_sir_values["I"].append(dset[i][1].detach().numpy())
            d_gt_sir_values["R"].append(dset[i][2].detach().numpy())
            d_gt_sir_values["D"].append(dset[i][3].detach().numpy())       

        # Plot SIR params 
        fig, ax = plt.subplots(3)
        fig.suptitle('Predicted SIR Parameters')
        ax[0].plot(v_x, d_sir_params["beta"] , label = "beta")
        ax[0].legend()
        ax[1].plot(v_x, d_sir_params["gamma"] , label = "gamma")
        ax[1].legend()
        ax[2].plot(v_x, d_sir_params["delta"] , label = "delta")
        ax[2].legend()

        # Plot SIR values
        fig2, ax2 = plt.subplots(4)
        fig2.suptitle('SIRD Values')
        ax2[0].plot(v_x, d_pred_sir_values["S"] , label = "S")
        ax2[0].plot(v_x, d_gt_sir_values["S"] , label = "S(real)")
        ax2[0].legend()
        ax2[1].plot(v_x, d_pred_sir_values["I"] , label = "I")
        ax2[1].plot(v_x, d_gt_sir_values["I"] , label = "I(real)")
        ax2[1].legend()
        ax2[2].plot(v_x, d_pred_sir_values["R"] , label = "R")
        ax2[2].plot(v_x, d_gt_sir_values["R"] , label = "R(real)")
        ax2[2].legend()
        ax2[3].plot(v_x, d_pred_sir_values["D"] , label = "D")
        ax2[3].plot(v_x, d_gt_sir_values["D"] , label = "D(real)")
        ax2[3].legend()

        # Plot Daily reported cases (Difference of I+R+D)
        total_reported_pred = np.array(d_pred_sir_values["I"]) + np.array(d_pred_sir_values["R"]) + np.array(d_pred_sir_values["D"])
        total_reported_gt   = np.array(d_gt_sir_values["I"]) + np.array(d_gt_sir_values["R"]) + np.array(d_gt_sir_values["D"])
        
        diff_pred = total_reported_pred[1:]-total_reported_pred[:-1]
        diff_gt = total_reported_gt[1:]-total_reported_gt[:-1]

        fig3, ax3 = plt.subplots(1)
        fig3.suptitle('Daily Reported Cases')
        ax3.plot(diff_pred , label = "Daily Reported (Predicted)")
        ax3.plot(diff_gt , label = "Daily Reported (Real)")
        ax3.legend()

        # Plot Daily deaths (Difference of D)
        total_dead_pred =  np.array(d_pred_sir_values["D"])
        total_dead_gt   =  np.array(d_gt_sir_values["D"])
    
        daily_dead_pred = total_dead_pred[1:] - total_dead_pred[:-1]
        daily_dead_gt = total_dead_gt[1:] - total_dead_gt[:-1]

        fig3, ax3 = plt.subplots(1)
        fig3.suptitle('Daily Deaths')
        ax3.plot(daily_dead_pred , label = "Daily Deaths (Predicted)")
        ax3.plot(daily_dead_gt , label = "Daily Deaths (Real)")
        ax3.legend()
        

