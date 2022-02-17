import covid19dh
import pandas as pd
from datetime import date, datetime, timedelta

def getCountryDataDF(country="Sri Lanka",start=None,end=None, normalize = True):
    '''
    Returns:
        population
        population_denormalization_factor
        df
    '''
    if start:
        if end==None:
            end = date.today() - timedelta(days=1)
        print("Data Frame : start= ",start,"    end= ",end)
        df, src = covid19dh.covid19(country,start=start, end=end, verbose = False) 
    else:
        df, src = covid19dh.covid19(country, verbose = False) 
    population = int(df[["population"]].iloc[0])
    # print('population',population)
    df = df[['date','confirmed','recovered','deaths']]
    df.fillna(0, inplace=True)
    df.rename(columns={"date": "Date", "confirmed": "Confirmed","deaths":"Deaths","recovered":"Recovered"}, inplace=True)
    df.set_index('Date',inplace=True)
    df = df.loc[~(df==0).all(axis=1)]
    #df = df.loc[(df != 0.0).any(axis=1)]
    df["Infected"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
    df["Susceptible"] = population - df["Confirmed"] 
    
    if normalize:
        pdnf = population/1000 
        return 1000 ,pdnf, df/population * 1000 # S,I,R,D per thousand
    else:
        pdnf = 1
        return population , pdnf, df

import torch

class SplitDataset():
    def __init__(self, 
                    country="Sri Lanka", 
                    start = date(2020,10,6),
                    end = date(2021,7,31), 
                    normalize = True, 
                    unsqueeze = False 
        ):

        #self.split_size = split_size
        self.population , self.population_denormalization_factor, self.real_data_df = getCountryDataDF(
            country=country , 
            start = start,
            end = end,
            normalize = normalize
        )
        
        #Dataframe to tensor
        S   = torch.tensor(self.real_data_df['Susceptible'].values)
        I   = torch.tensor(self.real_data_df['Infected'].values)
        R   = torch.tensor(self.real_data_df['Recovered'].values)
        D   = torch.tensor(self.real_data_df['Deaths'].values)
        
        self.data_tensor = torch.stack((S,I,R,D))
        self.data_tensor = torch.transpose(self.data_tensor, 0, 1) #shape = (split_size,4) 
        if unsqueeze:
            self.data_tensor= torch.unsqueeze(self.data_tensor, 1)
        self.data_tensor = self.data_tensor.float()

    def __len__(self):
        return len(self.real_data_df)

    def __getitem__(self, idx):       
        return self.data_tensor[idx] 

            

def getTorchDataset(country, start, end, normalize = True):

    dset = SplitDataset(
        country=country,
        start = start,
        end = end,

        normalize = normalize,
        unsqueeze=True
    )

    population = dset.population
    pdnf = dset.population_denormalization_factor

    return population,pdnf, dset


def build_dataset_from_config(conf):
    population,pdnf, trainset = getTorchDataset(
        country=conf['DATA']['COUNTRY'], 
        start = conf['DATA']['START_DATE'],
        end = conf['DATA']['END_DATE'],
        normalize = True
    )


    return population,pdnf, trainset
