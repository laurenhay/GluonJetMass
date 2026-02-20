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

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
unc_srcs = ['nominal']
# , 'JERUp', 'JERDown', 'HEM',
#  'JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown', 
#  'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp', 
#  'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown', 
#  'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
#  'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up', 'JES_RelativeJEREC1Down',
#  'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown', 'JES_RelativePtBBUp', 'JES_RelativePtBBDown',
#  'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 
#  'JES_RelativePtEC2Down', 'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 
#  'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 'JES_RelativeStatECUp', 'JES_RelativeStatECDown',
#  'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown', 'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 
#  'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']
parser.add_argument('--bg', choices=['TTjets', 'Wjets', 'Zjets', None], default="None", help="Which background sample to run of. If none run all successively")
parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--mctype', choices=['herwig', 'pythia', 'MG'], default="MG", help="MC generator running on")
parser.add_argument('--data', action='store_true', help="Run on data") 
parser.add_argument('--dask', action='store_true', help='Run on dask')
parser.add_argument('--dijetOnly', action='store_true', help='Only run dijet')
parser.add_argument('--trijetOnly', action='store_true', help='Only run trijet')
parser.add_argument('--testing', action='store_true', help='Testing; run on only a subset of data')
parser.add_argument('--verbose', type=bool, help='Have processor output status; set false if making log files', default='True')
parser.add_argument('--allUncertaintySources', action='store_true', help='Run processor for each unc. source separately')
parser.add_argument('--jetSyst', default=unc_srcs, nargs='+')
parser.add_argument('--datasetRange', default=None, help="Run on subset of available datasets", type=list_of_ints)
parser.add_argument('--jk', action='store_true', help="Run jackknife processor")
parser.add_argument('--jkRange', default=None, help="Run on subset of jk indices", type=list_of_ints)

arg = parser.parse_args()

environments = [arg.casa, arg.lpc, arg.winterfell]

if not np.any(environments): #if user forgets to assign environment
    print('Default environment -> lpc')
    arg.lpc = True
if arg.data:
    arg.jetSyst = ['nominal']
if arg.mctype == 'herwig':
    arg.jetSyst = ['nominal']
    

from python.plugins import *
from python.dijetProcessor import makeDijetHists
from python.trijetProcessor import makeTrijetHists
import pickle

#### WE'RE MISSING 2016B ver2 -- AK8 PF HLT is missing need to use AK4 trigger isntead
### Run coffea processor and make plots
        
def runBackgrounds(jet_syst=arg.jetSyst, year=arg.year, bg = arg.bg, casa=arg.casa, winterfell=arg.winterfell, testing=arg.testing, dask=arg.dask, verbose=arg.verbose, range=arg.datasetRange, mctype = arg.mctype, jk=arg.jk, jk_range = arg.jkRange, dijetOnly=arg.dijetOnly, trijetOnly=arg.trijetOnly):
    print("Running background ", bg)
    processor_dijet = makeDijetHists(data = False, jet_systematics = jet_syst, jk = jk, jk_range = jk_range)
    processor_trijet = makeTrijetHists(data = False, jet_systematics = jet_syst, jk = jk, jk_range = jk_range)
    if jk:
        if jk_range != None:
            jkstring = "JK" + str(jk_range[0]) + "_" + str(jk_range[1])
        else:
            jkstring = "JK" 
    else: jkstring = ""
    if year == 2016 or year == 2017 or year == 2018:
        year_str = str(year)
    elif year == "2016" or year == "2016APV" or year == "2017" or year == "2018":
        year_str = year
    else:
        year_str = "All"
    if bg == "TTjets":
        bg_files = ['fileset_TTbar_MG_wRedirs.json']
    elif bg == "Zjets":
        bg_files = ['fileset_Zjets_MG_wRedirs.json']
    elif bg == "Wjets":
        bg_files = ['fileset_Wjets_MG_wRedirs.json']
    else:
        bg_files = ['fileset_Zjets_MG_wRedirs.json', 'fileset_Wjets_MG_wRedirs.json', 'fileset_TTbar_MG_wRedirs.json']
    if testing:
        test_str = "Test"
    else:
        test_str = ""
    for filename in bg_files:
        print(filename)
        print(filename[8:13])
        fname_dijet = 'coffeaOutput/dijet/{}_dijetSel_{}_fixXSandpt_{}_{}.pkl'.format(filename[8:13], test_str, mctype, year_str)
        fname_trijet = 'coffeaOutput/trijet/{}_trijetSel_{}_fixXSandpt_{}_{}.pkl'.format(filename[8:13], test_str, mctype, year_str)
        if dijetOnly:
            result_dijet = runCoffeaJob(processor_dijet, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor_dijet.do_gen, year=year)
            with open(fname_dijet, "wb") as f:
                pickle.dump( result_dijet, f)
            del result_dijet
        elif trijetOnly:
            result_trijet = runCoffeaJob(processor_trijet, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor_trijet.do_gen, year=year)
            with open(fname_trijet, "wb") as f:
                pickle.dump( result_trijet, f)
            del result_trijet
        else: #### default is to do both
            result_dijet = runCoffeaJob(processor_dijet, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor_dijet.do_gen, year=year)
            with open(fname_dijet, "wb") as f:
                pickle.dump( result_dijet, f)
            del result_dijet
            result_trijet = runCoffeaJob(processor_trijet, jsonFile = filename, casa = casa, winterfell = winterfell, testing = testing, dask = dask, data = not processor_trijet.do_gen, year=year)
            with open(fname_trijet, "wb") as f:
                pickle.dump( result_trijet, f)
            del result_trijet
if arg.allUncertaintySources:
    # unc_srcs =["nominal","AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","JER","JMR","JMS","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF""RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    unc_srcs = ['nominal', 'JERUp', 'JERDown', 'HEM',
 'JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown', 
 'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp', 
 'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown', 
 'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
 'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up', 'JES_RelativeJEREC1Down',
 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown', 'JES_RelativePtBBUp', 'JES_RelativePtBBDown',
 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 
 'JES_RelativePtEC2Down', 'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 
 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 'JES_RelativeStatECUp', 'JES_RelativeStatECDown',
 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown', 'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 
 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown', 'JMRUp', 'JMRDown', 'JMSUp', 'JMSDown']
    for src in unc_srcs:
        print("Running processor for ", src)
        runBackgrounds(data=arg.data, jet_syst=[src])
else:
    runBackgrounds()
#Make plots
# import matplotlib.pyplot as plt
# os_path = 'plots/selectionStudies/dijet/'
# # result=result[0]
# plt.rcParams["figure.figsize"] = (10,10)
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
# result['jet_pt_mass_reco_u'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[0,0])
# result['jet_pt_mass_reco_u'][{'dataset':sum}].project('mreco').plot1d(ax=axs[0,1])
# result['jet_pt_mass_reco_g'][{'dataset':sum}].project('ptreco').plot1d(ax=axs[1,0])
# result['jet_pt_mass_reco_g'][{'dataset':sum}].project('mreco').plot1d(ax=axs[1,1])
# plt.savefig(os_path+'pt_m_reco_u_g.png')

# if not data:
#     plt.rcParams["figure.figsize"] = (20,15)
#     fig, axs = plt.subplots(2, 2)
#     fig.suptitle('Ungroomed (top) and groomed (bottom) reco jets')
#     result['jet_pt_mass_u_gen'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[0,0])
#     result['jet_pt_mass_u_gen'][{'dataset':sum}].project('mgen').plot1d(ax=axs[0,1])
#     result['jet_pt_mass_g_gen'][{'dataset':sum}].project('ptgen').plot1d(ax=axs[1,0])
#     result['jet_pt_mass_g_gen'][{'dataset':sum}].project('mgen').plot1d(ax=axs[1,1])
#     plt.savefig(os_path+"pt_m_gen_u_g.png")

#     response_matrix_u_values = result['response_matrix_u'].project("ptreco", "mreco", "ptgen", "mgen").values()
#     response_matrix_g_values = result['response_matrix_g'].project("ptreco", "mreco", "ptgen", "mgen").values()
#     response_matrix_g_final = response_matrix_g_values.reshape( (nptreco)*(nmassreco), (nptgen)*(nmassgen) )
#     response_matrix_u_final = response_matrix_u_values.reshape( (nptreco)*(nmassreco), (nptgen)*(nmassgen) ) 
#     plt.figure(figsize = (40,80))
#     plt.imshow( np.log(response_matrix_u_final+1), vmax=10, aspect="equal", cmap="Blues" )
#     plt.xlabel("GEN", fontsize=50)
#     plt.ylabel("RECO", fontsize=50)
#     plt.tick_params(labelsize=40)
#     plt.savefig(os_path+'response_matrix_u.png')
#     plt.figure(figsize = (40,80))
#     plt.imshow( np.log(response_matrix_g_final+1), vmax=10, aspect="equal", cmap="Blues" )
#     plt.xlabel("GEN", fontsize=50)
#     plt.ylabel("RECO", fontsize=50)
#     plt.tick_params(labelsize=40)
#     plt.savefig(os_path+'response_matrix_g.png')


