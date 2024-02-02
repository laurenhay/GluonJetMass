import awkward as ak
import numpy as np
import coffea
import os

print(coffea.__version__)
from coffea import util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import hist


from plugins import *
from trijetProcessor import makeTrijetHists
import pickle
import argparse

parser = argparse.ArgumentParser()

environmentGroup = parser.add_mutually_exclusive_group(required=True)
environmentGroup.add_argument('--casa', action='store_true', help='Use Coffea-Casa redirector: root://xcache/')
environmentGroup.add_argument('--lpc', action='store_true', help='Use CMSLPC redirector: root://cmsxrootd.fnal.gov/')
environmentGroup.add_argument('--winterfell', action='store_true', help='Get available files from UB Winterfell /mnt/data/cms')

parser.add_argument('--btag', choices=['bbloose', 'bloose', 'bbmed', 'bmed', None], default="bbloose") 
parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--data', action='store_true', help="Run on data") 
parser.add_argument('--dask', action='store_true', help='Run on dask')
parser.add_argument('--runProcessor', type=bool, help='Run processor; if True run the processor; if False, only make plots', default='True')
parser.add_argument('--testing', action='store_true', help='Testing; run on only a subset of data')
arg = parser.parse_args()

environments = [arg.casa, arg.lpc, arg.winterfell]

if not np.any(environments): #if user forgets to assign something here
    print('Default environment -> lpc')
    args.lpc = True

#### WE'RE MISSING 2016B ver2 -- AK8 PF HLT is missing need to use AK4 trigger isntead
### Run coffea processor and make plots
run_bool = arg.runProcessor
data_bool = arg.data
btag_str = arg.btag
year = arg.year
processor = makeTrijetHists(data = data_bool, btag = btag_str)
datastring = "JetHT" if processor.do_gen == False else "QCDsim"
if processor.do_gen==True and arg.winterfell:
    filename = "QCD_flat_files.json"
elif processor.do_gen==True:
    filename = "fileset_QCD.json"
else:
    filename = "datasets_UL_NANOAOD.json"
if year == 2016 or year == 2017 or year == 2018:
    year_str = str(year)
elif year == "2016" or year == "2016APV" or year == "2017" or year == "2018":
    year_str = year
else:
    year_str = "All"

if arg.testing and not data:
    fname = 'coffeaOutput/trijetHistsTest_wXSscaling_{}_pt{}_eta{}_{}jesjec.pkl'.format(datastring, processor.ptcut, processor.etacut, processor.btag, year_str)
elif arg.testing and data:
    fname = 'coffeaOutput/trijetHistsTest{}_pt{}_eta{}_{}jesjec.pkl'.format(datastring, processor.ptcut, processor.etacut, processor.btag, year_str)
elif not arg.testing and data:
    fname = 'coffeaOutput/trijetHists_{}_pt{}_eta{}_{}jesjec{}.pkl'.format(datastring, processor.ptcut, processor.etacut, processor.btag, year_str)
else:
    fname = 'coffeaOutput/trijetHists_wXSscaling_{}_pt{}_eta{}_{}jesjec{}.pkl'.format(datastring, processor.ptcut, processor.etacut, processor.btag, year_str)

if run_bool:
    result = runCoffeaJob(processor, jsonFile = filename, casa = arg.casa, winterfell = arg.winterfell, testing = arg.testing, dask = arg.dask, data = not processor.do_gen, verbose = False)
    with open(fname, "wb") as f:
        pickle.dump( result, f)

else:
    with open(fname, "rb") as f:
        result = pickle.load( f )
#Make plots
import matplotlib.pyplot as plt
os_path = 'plots/selectionStudies/dijet/'

plots = {}

plt.rcParams["figure.figsize"] = (10,10)
fig, axs = plt.subplots(2, 2)
fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[0,0])
result['jet_pt_mass_reco_u'][{'dataset':sum}].project('mreco').plot1d(ax=axs[0,1])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[1,0])
result['jet_pt_mass_reco_g'][{'dataset':sum}].project('mreco').plot1d(ax=axs[1,1])
plt.savefig(os_path+'/pt_m_reco_u_g.png')

if not data:
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

