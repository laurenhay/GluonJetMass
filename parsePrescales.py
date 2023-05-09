import json
import argparse
import pandas
import numpy as np
from correctionlib import convert
import correctionlib.schemav2 as cs
## run,cmsls,prescidx,totprescval,hltpath/prescval,logic,l1bit/prescval                                                          
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input') ##input JSON file                                                                          
parser.add_argument('-t', '--hltpath') ##desired hlt path                                                                       

makeJSON = False                                                                                                                                 
args = parser.parse_args()

if "Collisions16" in args.input:
    year = 2016
    HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500']
elif "Collisions17" in args.input:
    year = 2017
    HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
else:
    year = 2018
    HLT_paths = ['AK8PFJet15', 'AK8PFJet25', 'AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']

print(args.input)

info = open(args.input)
cert_jsonData = pandas.read_json(info, orient = 'index')
#print("Cert ", year, " data: \n", cert_jsonData)
print(year, " runs: \n", cert_jsonData)
#allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style csv' 
ps_csvData = pandas.read_csv("allRunsAK8HLT_skimmed.csv")

print("Prescale csv data: \n")
#print(ps_csvData)
#print("PS columns: ", ps_csvData.columns)
#print('String contains:', ps_csvData[ps_csvData['hltpath/prescval'].str.contains('HLT_AK8PFJet400')])
#print('String contains: \n')
runs = [362628, 362485, 274953, 274949]
#print(np.any(ps_csvData['# run']==run for run in runs))
#ps = ps_csvData[ps_csvData['hltpath/prescval'].str.contains('HLT_AK8PFJet400_v')][ps_csvData['# run']==np.any(runs)]['totprescval'].to_numpy()
#print(ps)
print(len(ps_csvData['totprescval'][ps_csvData['hltpath/prescval'].str.contains('AK8PFJet40_v')]))
#print(ps_csvData[ps_csvData['hltpath/prescval'].str.contains('HLT_AK8PFJet400_v')][ps_csvData['# run']==274953])


### convert ps csv into correctionlib json following along from github.com/cms-nanoAOD/correctionlib/blob/master/data/conversion.py
def get_ps(ps):
    #after previous selections, length of ps should be 1
    if len(ps) != 1:
        raise ValueError(ps)
    return ps.iloc[0]["totprescal"]

def build_lumibins(ps):
    edges = sorted(set(ps["cmsls"]))
    print("Lumi block edges ", edges)
    if len(edges) == 1:
        return get_ps
    else:
        return cs.Binning.parse_obj({
            "nodetype": "binning",
            "input": "lumi",
            "edges": edges,
            "content":[get_ps(ps[(ps["cmsls"]>=lo) & (ps["cmsls"]<hi)]) for lo,hi in zip(edges[:-1],edges[1:])
            
def build_

def psOutput(psData, certData, hltpaths):
    output = cs.Category.parse_obj({
        "nodetype": "category",
        "input":"run",
        "content":[cs.CategoryItem.parse_obj({
            "key": run,
            "value": cs.Category.parse_obj({
                "nodetype":"category",
                "input":"lumi",
                "content":[cs.CategoryItem.parse_obj({
                    "key":lumi,
                    "value":[cs.Categor.parse_obj({


psCorr = cs.Correction(
    name = "prescaleWeight",
    #version?
    inputs = [
        #cs.Variable(name="pt",   type="real",   description="Leading jet pt"), # try to figure out how to assign path inside correctionlib later 
        cs.Variable(name="run",  type="int",    descriptiion="Run number"),
        cs.Variable(name="path", type="string", desciption="HLT path"),
        cs.Variable(name="lumi", type="int",    description="Luminosity block"),
    ],
    output = cs.Variable(name="weight", type="real", descripiton="Prescale event weight"),
    data = cs.Category.parse_obj({
        nodetype:"category",
        input: "run",
        content:cs.Category.parse_obj({
            nodetype:"category",
            input:"lumi",
            content:cs.Category.parse_obj({
                nodetype:"category"
        } 
    ]
if makeJSON:
    with open("prescales_AK8HLT.json", "w") as f:
        #### get run numbers from Golden JSON                                                                                        
        for run in list(jsonData.keys()):
            for runColumn in ps_csvData[:,1]:
                f.write(runColumn)



