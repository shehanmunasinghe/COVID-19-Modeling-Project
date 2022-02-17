import torch
import torch.nn as nn

def DifferenceAccumulatedReported(out_pred,gt_y):
    # confirmed
    pred_c = out_pred[:,1]+out_pred[:,2]+out_pred[:,3]
    gt_c = gt_y[:,1]+gt_y[:,2]+gt_y[:,3]
    loss_c = nn.functional.mse_loss(pred_c,gt_c) /100

    # deaths
    loss_d = nn.functional.mse_loss(out_pred[:,3], gt_y[:,3])

    return (loss_c + loss_d)/1000


class DifferenceDailyReported:
    '''  Calculate loss based on difference in daily reported cases'''

    def __init__(self, population_denormalization_factor = 1) -> None:
        self.population_denormalization_factor = population_denormalization_factor
        

    def __call__(self, pred_y, gt_y):
        '''
        Args: 
            pred_y: (num_days, 4)
            gt_y: (num_days, 4)
        '''
        # Total reported
        total_reported_pred = pred_y[:,1]+pred_y[:,2]+pred_y[:,3]
        total_reported_gt   = gt_y[:,1]+gt_y[:,2]+gt_y[:,3]

        # forward difference (daily change)
        daily_reported_pred = total_reported_pred[1:] - total_reported_pred[:-1]
        daily_reported_gt   = total_reported_gt[1:] - total_reported_gt[:-1]

        # diff_pred   =  pred_y[1:,:] - pred_y[:-1,:]
        # diff_gt     =  gt_y[1:,:] - gt_y[:-1,:]

        # I+R+D
        # daily_reported_pred = diff_pred[:,1]+diff_pred[:,2]+diff_pred[:,3]
        # daily_reported_gt   = diff_gt[:,1]+diff_gt[:,2]+diff_gt[:,3]

        loss = nn.functional.mse_loss(daily_reported_pred,daily_reported_gt) * self.population_denormalization_factor #/1000 * self.population

        return loss

def build_loss_from_config(conf, population_denormalization_factor=1):
    if conf["OPTIMIZATION"]["CRITERION"] == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss()
    elif conf["OPTIMIZATION"]["CRITERION"] == 'DifferenceDailyReported':
        return DifferenceDailyReported(population_denormalization_factor)
        
    return None