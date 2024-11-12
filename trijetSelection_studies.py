import awkward as ak
import numpy as np
import coffea
import os

print(coffea.__version__)
from coffea import util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import hist


from python.plugins import *
from python.trijetProcessor import makeTrijetHists
import pickle
import argparse

parser = argparse.ArgumentParser()

environmentGroup = parser.add_mutually_exclusive_group(required=False)
environmentGroup.add_argument('--casa', action='store_true', help='Use Coffea-Casa redirector: root://xcache/')
environmentGroup.add_argument('--lpc', action='store_true', help='Use CMSLPC redirector: root://cmsxrootd.fnal.gov/')
environmentGroup.add_argument('--winterfell', action='store_true', help='Get available files from UB Winterfell /mnt/data/cms')

parser.add_argument('--btag', choices=['bbloose', 'bloose', 'bbmed', 'bmed', None], default="None") 
parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--mctype', choices=['herwig', 'pythia', 'MG'], default="MG", help="MC generator running on")
parser.add_argument('--data', action='store_true') 
parser.add_argument('--dask', action='store_true', help='Run on dask')
parser.add_argument('--testing', action='store_true', help='Testing; run on only a subset of data')
parser.add_argument('--verbose', type=bool, help='Have processor output status; set false if making log files', default='True')
parser.add_argument('--allUncertaintySources', action='store_true', help='Run processor for each unc. source separately')
parser.add_argument('--jetSyst', default=['nominal', 'HEM'], nargs='+')
parser.add_argument('--syst', default=['PUSF', 'L1PreFiringWeight'], nargs='+')
parser.add_argument('--datasetRange', default=None, help="Run on subset of available datasets")
parser.add_argument('--jk', action='store_true', help="Run jackknife processor")
arg = parser.parse_args()

environments = [arg.casa, arg.lpc, arg.winterfell]

if not np.any(environments): #if user forgets to assign something here
    print('Default environment -> lpc')
    arg.lpc = True

#### WE'RE MISSING 2016B ver2 -- AK8 PF HLT is missing need to use AK4 trigger isntead
### Run coffea processor and make plots
def runTrijetAnalysis(data=arg.data, jet_syst=arg.jetSyst, year=arg.year, casa=arg.casa, winterfell=arg.winterfell, testing=arg.testing, dask=arg.dask, verbose=arg.verbose, syst=arg.syst, range=arg.datasetRange, mctype = arg.mctype, jk=arg.jk):
    processor = makeTrijetHists(data = arg.data, btag = arg.btag, jet_systematics = jet_syst, systematics = syst, jk=jk)
    jkstring = "JK_" if jk else ""
    if year == 2016 or year == 2017 or year == 2018:
        year_str = str(year)
    elif year == "2016" or year == "2016APV" or year == "2017" or year == "2018":
        year_str = year
    else:
        year_str = "All"
    datastring = "JetHT" if processor.do_gen == False else "QCDsim"
    if processor.do_gen==True and arg.winterfell:
        filename = "QCD_flat_files.json"
    elif processor.do_gen==True:
        # filename = "fileset_QCD.json"
        if mctype == "MG":
            filename = "fileset_MG_pythia8_wRedirs.json"
        elif mctype == "herwig":
            filename = "fileset_HERWIG_wRedirs.json"
        elif mctype == "pythia":
            filename = "fileset_QCD_wRedirs.json"
    else:
        # filename = "datasets_UL_NANOAOD.json"
        filename = "fileset_JetHT_wRedirs.json"

    if arg.testing and not arg.data:
        fname = 'coffeaOutput/trijet/trijetHistsTest_ak4corr_{}_rap{}_{}{}_{}{}.pkl'.format(datastring, processor.ycut, jet_syst[0],mctype, jkstring, year_str)
    elif arg.testing and arg.data:
        fname = 'coffeaOutput/trijet/trijetHistsTest{}_rap{}_{}{}_{}{}.pkl'.format(datastring, processor.ycut,jet_syst[0],mctype, jkstring, year_str)
    elif not arg.testing and arg.data:
        fname = 'coffeaOutput/trijet/trijetHists_{}_rap{}_{}{}_{}{}.pkl'.format(datastring, processor.ycut,jet_syst[0], mctype, jkstring, year_str)
    else:
        fname = 'coffeaOutput/trijet/trijetHists_ak4corr_{}_rap{}_{}{}_{}{}.pkl'.format(datastring, processor.ycut,jet_syst[0],mctype, jkstring, year_str)
    if range!=None:
        print("Range input: ", range)
        fname=fname[:-4]+"_"+range[0]+"_"+range[1]+".pkl"
        print("New ranged fname ", fname)
        result = runCoffeaJob(processor, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor.do_gen, verbose = verbose, year=year, datasetRange = range)
    else:
        result = runCoffeaJob(processor, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor.do_gen, verbose = verbose, year=year)
    with open(fname, "wb") as f:
        pickle.dump( result, f)

if arg.allUncertaintySources:
    
    unc_srcs =["nominal","AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","JER","JMR","JMS","HEM", "Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef","FlavorQCD","JER","JMR","JMS","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF""RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    if arg.mctype!='pythia':
        unc_srcs.append(["Q2", "PDF"])
else:
    unc_srcs = arg.jetSyst
for src in unc_srcs:
    print("Running processor for ", src)
    runTrijetAnalysis(jet_syst=[src])

#Make plots
import matplotlib.pyplot as plt
os_path = 'plots/selectionStudies/trijet/'

plots = {}

plt.rcParams["figure.figsize"] = (10,10)
fig, axs = plt.subplots(2, 2)
fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[0,0])
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('mreco').plot1d(ax=axs[0,1])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[1,0])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('mreco').plot1d(ax=axs[1,1])
plt.savefig(os_path+'/pt_m_reco_u_g.png')

if not arg.data:
    plt.rcParams["figure.figsize"] = (20,15)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
    result['jet_pt_mass_gen_u'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[0,0])
    result['jet_pt_mass_gen_u'][{'dataset':sum}].project('mgen').plot1d(ax=axs[0,1])
    result['jet_pt_mass_gen_g'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[1,0])
    result['jet_pt_mass_gen_g'][{'dataset':sum}].project('mgen').plot1d(ax=axs[1,1])
    plt.savefig(os_path+"/pt_m_gen_u_g.png")

    response_matrix_u_values = result['response_matrix_u'].project("ptreco", "mreco", "ptgen", "mgen").values()
    response_matrix_g_values = result['response_matrix_g'].project("ptreco", "mreco", "ptgen", "mgen").values()
    response_matrix_g_final = response_matrix_g_values.reshape( (nptreco)*(nmassreco), (nptgen)*(nmassgen) )
    response_matrix_u_final = response_matrix_u_values.reshape( (nptreco)*(nmassreco), (nptgen)*(nmassgen) ) 
    plt.figure(figsize = (40,80))
    plt.imshow( np.log(response_matrix_u_final+1), vmax=10, aspect="equal", cmap="Blues" )
    plt.xlabel("GEN", fontsize=50)
    plt.ylabel("RECO", fontsize=50)
    plt.tick_params(labelsize=40)
    plt.savefig(os_path+'response_matrix_u.png')
    plt.figure(figsize = (40,80))
    plt.imshow( np.log(response_matrix_g_final+1), vmax=10, aspect="equal", cmap="Blues" )
    plt.xlabel("GEN", fontsize=50)
    plt.ylabel("RECO", fontsize=50)
    plt.tick_params(labelsize=40)
    plt.savefig(os_path+'response_matrix_g.png')

