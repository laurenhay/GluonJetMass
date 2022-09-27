### This file houses plugins
import pandas as pd

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

#reads in files and adds redirector
def handleData(jsonFile, redirector, testing = True, data = False):
    if data == True:
        inputs = 'JetHT_data'
    else: inputs = 'QCD_sim'
    df = pd.read_json(jsonFile)   
    dict = {} 
    if testing:
        for year in df[inputs].keys():
            dict[year] = [redirector + df[inputs][year][0]]
    else:
        dict = df[inputs].to_dict()
        for key in dict.keys():
            dict[key] = [redirector + dict[key][i] for i in range(len(dict[key]))]
    return dict