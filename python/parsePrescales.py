import json
import argparse
import pandas
import numpy as np
from correctionlib import convert
import correctionlib.schemav2 as cs
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input') ##input JSON file                                                                          
parser.add_argument('-t', '--hltpath') ##desired hlt path                                                                       
                                                                                                                                 
args = parser.parse_args()

if args.hltpath != None:
    HLT_paths = [args.hltpath+'40', args.hltpath+'60', args.hltpath+'80', args.hltpath+'140', args.hltpath+'200', args.hltpath+'260', args.hltpath+'320', args.hltpath+'400', args.hltpath+'450', args.hltpath+'500']
    if "Collisions16" in args.input:
        year = 2016
    elif "Collisions17" in args.input:
        year = 2017
        HLT_paths = HLT_paths + [args.hltpath+'550']
    else:
        year = 2018
        HLT_paths = ['AK8PFJet15', 'AK8PFJet25'] + HLT_paths + [args.hltpath+'550']
else:
    if "Collisions16" in args.input:
        year = 2016
        HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500']
    elif "Collisions17" in args.input:
        year = 2017
        HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
    else:
        year = 2018
        HLT_paths = ['AK8PFJet15', 'AK8PFJet25', 'AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']

#print(args.input)

info = open(args.input)
cert_jsonData = pandas.read_json(info, orient = 'index')
#print("Cert ", year, " data: \n", cert_jsonData)
#print(year, " runs: \n", cert_jsonData.iloc[:,0].keys().to_list())
golden_runs = cert_jsonData.iloc[:,0].keys().to_list()

#### allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style csv' 
#### run,cmsls,prescidx,totprescval,hltpath/prescval,logic,l1bit/prescval  
if args.hltpath == None or args.hltpath == "AK8PFJet":
    ps_csvData = pandas.read_csv("allRunsAK8HLT_skimmed.csv")
elif args.hltpath == "PFJet" and year == 2016:
    ps_csvData = pandas.read_csv("allRunsAK4HLT2016_skim.csv")
else:
    print("invalid HLT inputs")
#### Removing "None" values from ps data
#print(ps_csvData[ps_csvData['# run'].isin(golden_runs)])
ps_csvData = ps_csvData[ps_csvData['cmsls']!='None']

### convert ps csv into correctionlib json following along from github.com/cms-nanoAOD/correctionlib/blob/master/data/conversion.py
def get_ps(ps):
    ### after previous selections, length of ps should be 1
    if len(ps) != 1:
        print("Length of ps after selection ", len(ps)) 
        raise ValueError(ps)
    print("Final prescale weight: ", ps.iloc[0]["totprescval"])
    return float(ps.iloc[0]["totprescval"])

def build_lumibins(ps):
    ##### to sort as bin edges properly, starting lumi sections need to be stored as floats 
    print("Path: ", ps["hltpath/prescval"].iloc[0], type(ps["hltpath/prescval"].iloc[0]))
    edges = sorted(set(ps["cmsls"].astype(float)))
    if len(edges)==1:
        return get_ps(ps)
    elif len(edges) > 1:
        #print("Prescale changed")
        edges.append(float('inf'))
        print("Lumi bin edges: ", list(zip(edges[:-1],edges[1:])))
        content = [get_ps(ps[(ps["cmsls"].astype(float)>=lo) & (ps["cmsls"].astype(float)<hi)]) for lo,hi in zip(edges[:-1],edges[1:])]
        print("Prescales: ", content)
        return cs.Binning.parse_obj({
            "nodetype": "binning",
            "input": "lumi",
            "edges": edges,
            "content":content,
            "flow":"clamp"
            })

def build_paths(ps, HLT_paths):
    print("Run: ", ps["# run"].iloc[0], type(ps["# run"].iloc[0]))
    #####  paths are unique bc of hltpath/lumi --> must make array of path name separate as done above
    paths = HLT_paths
    print("Type of path key: ", type(paths[0])) 
    return cs.Category.parse_obj({
        "nodetype": "category",
        "input":"path",
        "content":[{"key":str(path), "value":build_lumibins(ps[ps['hltpath/prescval'].str.contains(path+"_v")])} for path in paths],
    })
            
def build_runs(ps, golden_runs):
    json_runs = golden_runs
    runs = sorted(ps["# run"][ps['# run'].isin(json_runs)].unique())
    print("Selected ", len(runs), ": ", runs)
    return cs.Category.parse_obj({
        "nodetype":"category",
        "input":"run",
        "content":[{"key":int(run), "value":build_paths(ps[ps['# run']==run], HLT_paths)} for run in runs],
    })


psCorr = cs.Correction.parse_obj(
    {
        "version": 2,
        "name": "prescaleWeight",
        "inputs": [
            #cs.Variable(name="pt",   type="real",   description="Leading jet pt"), # try to figure out how to assign path inside correctionlib later 
            {"name":"run", "type":"int"},
            {"name":"path", "type":"string"},
            {"name":"lumi", "type":"real"},
        ],
        "output":{"name":"weight", "type":"real"},
        "data":build_runs(ps_csvData, golden_runs),
    })

cset = cs.CorrectionSet(
    schema_version = 2,
    corrections = [psCorr],
    description = "Jet prescales for " + str(year)
    )

with open("ps_weight_JSON_"+args.hltpath+str(year)+".json", "w") as f:
    f.write(cset.json(exclude_unset=True))
