import awkward as ak
import numpy as np
import coffea
import os

print(coffea.__version__)
from coffea import util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import hist
import argparse

parser = argparse.ArgumentParser()

environmentGroup = parser.add_mutually_exclusive_group(required=False)
environmentGroup.add_argument('--casa', action='store_true', help='Use Coffea-Casa redirector: root://xcache/')
environmentGroup.add_argument('--lpc', action='store_true', help='Use CMSLPC redirector: root://cmsxrootd.fnal.gov/')
environmentGroup.add_argument('--winterfell', action='store_true', help='Get available files from UB Winterfell /mnt/data/cms')

parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--data', action='store_true', help="Run on data") 
parser.add_argument('--dask', action='store_true', help='Run on dask')
parser.add_argument('--testing', action='store_true', help='Testing; run on only a subset of data')
parser.add_argument('--verbose', type=bool, help='Have processor output status; set false if making log files', default='True')
parser.add_argument('--allUncertaintySources', action='store_true', help='Run processor for each unc. source separately')
parser.add_argument('--jetSyst', default=['nominal', 'jer'], nargs='+')
parser.add_argument('--syst', default=['PUSF', 'L1PreFiringWeight'], nargs='+')

arg = parser.parse_args()

environments = [arg.casa, arg.lpc, arg.winterfell]

if not np.any(environments): #if user forgets to assign something here
    print('Default environment -> lpc')
    arg.lpc = True

from python.plugins import *
from python.dijetProcessor import makeDijetHists
import pickle

#### WE'RE MISSING 2016B ver2 -- AK8 PF HLT is missing need to use AK4 trigger isntead
### Run coffea processor and make plots
        
def runDijetAnalysis(data=arg.data, jet_syst=arg.jetSyst, year=arg.year, casa=arg.casa, winterfell=arg.winterfell, testing=arg.testing, dask=arg.dask, verbose=arg.verbose, syst=arg.syst):
    processor = makeDijetHists(data = data, jet_systematics = jet_syst, systematics = syst)
    datastring = "JetHT" if processor.do_gen == False else "QCDsim"
    if year == 2016 or year == 2017 or year == 2018:
        year_str = str(year)
    elif year == "2016" or year == "2016APV" or year == "2017" or year == "2018":
        year_str = year
    else:
        year_str = "All"
    if processor.do_gen==True and arg.winterfell:
        filename = "QCD_flat_files.json"
    elif processor.do_gen==True:
        filename = "fileset_QCD.json"
    else:
        filename = "datasets_UL_NANOAOD.json"
    if arg.testing and not data:
        fname = 'coffeaOutput/dijet/dijetHistsTest_wXSscaling_{}_pt{}_rapidity{}_{}{}.pkl'.format(datastring, processor.ptcut, processor.ycut, jet_syst[0]+jet_syst[-1],year_str)
    elif arg.testing and data:
        fname = 'coffeaOutput/dijet/dijetHistsTest{}_pt{}_rapidity{}_{}{}.pkl'.format(datastring, processor.ptcut, processor.ycut, jet_syst[0]+jet_syst[-1],year_str)
    elif not arg.testing and data:
        fname = 'coffeaOutput/dijet/dijetHists_{}_pt{}_rapidity{}_{}{}.pkl'.format(datastring, processor.ptcut, processor.ycut, jet_syst[0]+jet_syst[-1], year_str)
    else:
        fname = 'coffeaOutput/dijet/dijetHists_wXSscaling_{}_pt{}_rapidity{}_{}{}.pkl'.format(datastring, processor.ptcut, processor.ycut, jet_syst[0], year_str)
    result = runCoffeaJob(processor, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, year=year, data = not processor.do_gen, verbose=False)
    with open(fname, "wb") as f:
        pickle.dump( result, f)
if arg.allUncertaintySources:
    unc_srcs = ["nominal", "jer", "AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample",
"RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
else:
    unc_srcs = arg.jetSyst
for src in unc_srcs:
    print("Running processor for ", src)
    runDijetAnalysis(jet_syst=[src])

#Make plots
import matplotlib.pyplot as plt
os_path = 'plots/selectionStudies/dijet/'
# result=result[0]
plt.rcParams["figure.figsize"] = (10,10)
fig, axs = plt.subplots(2, 2)
fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[0,0])
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('mreco').plot1d(ax=axs[0,1])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[1,0])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('mreco').plot1d(ax=axs[1,1])
plt.savefig(os_path+'pt_m_reco_u_g.png')

if not data:
    plt.rcParams["figure.figsize"] = (20,15)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
    result['jet_pt_mass_u_gen'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[0,0])
    result['jet_pt_mass_u_gen'][{'dataset':sum}].project('mgen').plot1d(ax=axs[0,1])
    result['jet_pt_mass_g_gen'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[1,0])
    result['jet_pt_mass_g_gen'][{'dataset':sum}].project('mgen').plot1d(ax=axs[1,1])
    plt.savefig(os_path+"pt_m_gen_u_g.png")

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

