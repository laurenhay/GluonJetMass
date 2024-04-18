import ROOT
import numpy as np
import array as array
import math
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS, hep.style.firamath])
import uproot
import hist
import coffea
from coffea import processor
#import statistics as st
ROOT.gStyle.SetOptStat(000000)
ROOT.gInterpreter.ProcessLine('#include "MyTUnfoldDensity.h"')
print(coffea.__version__)
print(uproot.__version__)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--syst', default=['jer', 'PUSF', 'L1PreFiringWeight'], nargs='+')
parser.add_argument('--allUncSrcs', action='store_true', help='Run processor for each unc. source separately')
parser.add_argument('-i', '--input', required=False, help='MC input pkl file')
parser.add_argument('-d', '--dataInput', default=None,  help='Data input pkl file; if none provided only run closure test')                    
arg = parser.parse_args()
#### define plotting functions
def plotinputsROOT(matrix, truth, reco, groom="", syst="", year=""):
            histMCReco_M=matrix.ProjectionY("MCReco "+groom)
            histMCTruth_M=matrix.ProjectionX("MCTruth "+groom)
            c1 = ROOT.TCanvas("c1","Plot MC input "+groom+" binned by pt (outer) and mass (inner)",1200,400)
            c1.Divide(2,1)
            c1.cd(1)
            histMCTruth_M.SetMarkerStyle(21)
            histMCTruth_M.SetLineColor(ROOT.kRed)
            histMCTruth_M.SetMarkerColor(ROOT.kRed)
            histMCTruth_M.Draw()
            truth.SetMarkerStyle(24)
            truth.SetLineColor(ROOT.kBlue)
            truth.SetMarkerColor(ROOT.kBlue)
            truth.Draw("SAME")
            leg1 = ROOT.TLegend(0.7, 0.7, 0.86, 0.86)
            leg1.AddEntry(histMCTruth_M, "MC Gen from M", "p")
            leg1.AddEntry(truth, "MC Grn", "p")
            leg1.Draw()
            c1.cd(2)
            histMCReco_M.SetMarkerStyle(21)
            histMCReco_M.SetLineColor(ROOT.kRed)
            histMCReco_M.SetMarkerColor(ROOT.kRed)
            histMCReco_M.Draw()
            reco.SetMarkerStyle(24)
            reco.SetLineColor(ROOT.kBlue)
            reco.SetMarkerColor(ROOT.kBlue)
            reco.Draw("SAME")
            leg1_2 = ROOT.TLegend(0.7, 0.7, 0.86, 0.86)
            leg1_2.AddEntry(histMCReco_M, "MC Reco from M", "p")
            leg1_2.AddEntry(reco, "MC Reco", "p")
            leg1_2.Draw()

            c1.SaveAs("MCInput_"+groom+"_"+syst+"flatmatrix"+year+".png")
            c1.Close()

#### open files
fname = "coffeaOutput/dijetHists_wXSscaling_QCDsim_pt200.0_rapidity2.5jesjecL1PU2016.pkl"
with open(fname, "rb") as f:
    result = pickle.load( f )
fname = "coffeaOutput/dijetHists_JetHT_pt200.0_rapidity2.5jesjecL1PU2016.pkl"
with open(fname, "rb") as f:
    result_data = pickle.load( f )
if "dijet" in fname:
    os_path = "plots/unfolding/dijet/"
else:
    os_path = "plots/unfolding/trijet/"
year = fname[-8:-4]
print(year)
#### for data get results once
data_pt_m_u =  result_data["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
data_pt_m_g = result_data["jet_pt_mass_reco_g"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
#### get 
axis_names = [ax.name for ax in result['jet_pt_mass_reco_u'].axes]
print("avail hists ", result.keys())
print(axis_names)
cats = [cat for cat in result['jet_pt_mass_reco_u'][{'ptreco':sum, 'dataset':sum, 'mreco':sum}].axes[0]]
availSysts = [ax for ax in result['jet_pt_mass_reco_u'].project("syst").axes[0]]
print("Available systs ", availSysts)
print("Non nominal systs ", [syst for syst in availSysts if ("Up" in syst)])
syst_hist_dict = {}
for syst in [syst for syst in availSysts if ("Up" in syst)]:
    syst = syst[:-2]
    print("Running unfolding for systematic ", syst)
    ptgen_mgen_u = result["jet_pt_mass_gen_u"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
    ptgen_mgen_u_up = result["jet_pt_mass_gen_u"][{'dataset':sum, 'syst':syst+"Up"}].values(flow=True)
    ptgen_mgen_u_dn = result["jet_pt_mass_gen_u"][{'dataset':sum, 'syst':syst+"Down"}].values(flow=True)
    ptgen_mgen_g = result["jet_pt_mass_gen_g"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
    ptgen_mgen_g_up = result["jet_pt_mass_gen_g"][{'dataset':sum, 'syst':syst+"Up"}].values(flow=True)
    ptgen_mgen_g_dn = result["jet_pt_mass_gen_g"][{'dataset':sum, 'syst':syst+"Down"}].values(flow=True)
    fakes_ptreco_mreco = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["fakes"][{'dataset':sum, 'syst':"nominal"}]))
    misses_ptgen_mgen = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["misses"][{'dataset':sum, 'syst':"nominal"}]))
    # open data root file and get hists
    ptreco_mreco_uu  = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':"nominal"}]))
    # ptreco_mreco_u_up  = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':syst+"Up"}]))
    # ptreco_mreco_u_dn  = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':syst+"Down"}]))
    ptreco_mreco_u  = result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
    ptreco_mreco_u_up  = result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':syst+"Up"}].values(flow=True)
    ptreco_mreco_u_dn  = result["jet_pt_mass_reco_u"][{'dataset':sum, 'syst':syst+"Down"}].values(flow=True)
    ptreco_mreco_g = result["jet_pt_mass_reco_g"][{'dataset':sum, 'syst':"nominal"}].values(flow=True)
    ptreco_mreco_g_up  = result["jet_pt_mass_reco_g"][{'dataset':sum, 'syst':syst+"Up"}].values(flow=True)
    ptreco_mreco_g_dn  = result["jet_pt_mass_reco_g"][{'dataset':sum, 'syst':syst+"Down"}].values(flow=True)
    response_matrix_u, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'][{'syst':"nominal"}].project("ptreco", "mreco", "ptgen", "mgen").to_numpy(flow=True)
    response_matrix_g = result['response_matrix_g'][{'syst':"nominal"}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    response_matrix_u_up = result['response_matrix_u'][{'syst':syst+"Up"}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    response_matrix_u_dn = result['response_matrix_u'][{'syst':syst+"Down"}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    response_matrix_g_up = result['response_matrix_g'][{'syst':syst+"Up"}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    response_matrix_g_dn = result['response_matrix_g'][{'syst':syst+"Down"}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    n_ptgen_bin = len(ptgen_edges)-1
    n_mgen_bin= len(mgen_edges)-1
    mreco_edges = mreco_edges[1:-1]
    ptreco_edges = ptreco_edges[1:-1]
    mgen_edges = mgen_edges[1:-1]
    ptgen_edges = ptgen_edges[1:-1]
    print( "Size of ptreco bins: " , n_ptreco_bin)
    print( "Size of mreco bins: ",n_mreco_bin)
    print( "Size of ptgen bins: ", n_ptgen_bin)
    print( "Size of mgen bins: ", n_mgen_bin)
    print( "Size of response matrix ", len(response_matrix_u.flatten()))
    
    #### make TUnfold binning axes
    
    print("Nbins ptreco ", n_ptreco_bin, " Nbins mreco", n_mreco_bin)
    print("Bins ptreco ", ptreco_edges, " bins mreco", mreco_edges)
    detectorBinning = ROOT.TUnfoldBinning("detector")
    recoBinning = detectorBinning.AddBinning("reco")
    #recoBinning.AddAxis(ptreco_mreco_uu.GetYaxis(), False, False) # mreco
    #recoBinning.AddAxis(ptreco_mreco_uu.GetXaxis(), False, False) # ptreco
    recoBinning.AddAxis("m_{RECO}", n_mreco_bin-2, mreco_edges, False, False) # mreco
    recoBinning.AddAxis("pt_{RECO}", n_ptreco_bin-2, ptreco_edges, False, False) # ptreco

    print("Nbins ptgen ", n_ptgen_bin, " Nbins mgen", n_mgen_bin)
    print("Bins ptgen ", ptgen_edges, " bins mgen", mgen_edges)
    generatorBinning = ROOT.TUnfoldBinning("generator")
    fakeBinning=generatorBinning.AddBinning("fakesBin", 1)
    genBinning = generatorBinning.AddBinning("gen")
#    genBinning.AddAxis(ptgen_mgen_u.GetYaxis(), False, False) # mgen
#    genBinning.AddAxis(ptgen_mgen_u.GetXaxis(), False, False) # ptgen
    genBinning.AddAxis("m_{GEN}", n_mgen_bin-2, mgen_edges, False, False) #mgen
    genBinning.AddAxis("pt_{GEN}", n_ptgen_bin-2, ptgen_edges, False, False) #ptgen  
    print("Test binning for ptreco = 250, mreco = 500", recoBinning.GetGlobalBinNumber(250,500))
  
    # create histogram of migrations and gen hists
    histMCGenRec_u=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Ungroomed")
    histMCGenRec_u_up=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec+jerUp Ungroomed")
    histMCGenRec_u_dn=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec-jerDn Ungroomed")
    histMCGenRec_g=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Groomed")
    histMCGenRec_g_up=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec+jerUp Groomed")
    histMCGenRec_g_dn=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec-jerDn Groomed")
    #### create data histograms
    histDataReco_u=recoBinning.CreateHistogram("histDataReco Ungroomed")  #data values in reco binnning --> input to unfolding
    histDataReco_g=recoBinning.CreateHistogram("histDataReco Groomed")  #data values in reco binnning --> input to unfolding

    #### create MC hists for comparison
    histMCReco_u=recoBinning.CreateHistogram("histMCReco Ungroomed") #gen values in reco binning --> htruef in Sal's example
    histMCReco_u_up=recoBinning.CreateHistogram("histMCReco w/ unc up Ungroomed")
    histMCReco_u_dn=recoBinning.CreateHistogram("histMCReco w/ unc dn Ungroomed")
    histMCTruth_u=genBinning.CreateHistogram("histMCTruth Ungroomed")  #gen values in gen binning
    histMCTruth_u_up=genBinning.CreateHistogram("histMCTruth w/ unc up Ungroomed")
    histMCTruth_u_dn=genBinning.CreateHistogram("histMCTruth w/ unc dn Ungroomed")
    
    histMCReco_g=recoBinning.CreateHistogram("histMCReco Groomed") #gen values in reco binning --> htruef in Sal's example
    histMCReco_g_up=recoBinning.CreateHistogram("histMCReco w/ unc up Groomed")
    histMCReco_g_dn=recoBinning.CreateHistogram("histMCReco w/ unc dn Groomed")
    histMCTruth_g=genBinning.CreateHistogram("histMCTruth Groomed")  #gen values in gen binning 
    histMCTruth_g_up=genBinning.CreateHistogram("histMCTruth w/ unc up Groomed")
    histMCTruth_g_dn=genBinning.CreateHistogram("histMCTruth w/ unc dn Groomed")

    histMCTruth_RecoBinned_u=recoBinning.CreateHistogram("histMCTruth Reco Binned, Ungroomed") #gen values in reco binning --> htruef in Sal's example  
    histMCTruth_RecoBinned_g=recoBinning.CreateHistogram("histMCTruth Reco Binned, Groomed") #gen values in reco binning --> htruef in Sal's example  
    #### Loop through reco and gen bins of MC input and fill hist of migrations
    #### reco loop: i is ptreco, j is mreco
    for i in range(n_ptreco_bin-1):
        for j in range(n_mreco_bin-1):
            glob_recobin=(i)*(n_mreco_bin)+j
            print("Bin j = " , j , " has edge mreco " , mreco_edges[j] , "and  bin i " , i , " has ptreco edge " , ptreco_edges[i])
            recoBin=recoBinning.GetGlobalBinNumber(mreco_edges[j],ptreco_edges[i])
	    #### only fill fakes in fake genBin
            fake_weight = fakes_ptreco_mreco.GetBinContent(i,j)
            #print("Fake weight ", fake_weight," for i == ",i, " and j == ",j)
            fakeBin=fakeBinning.GetStartBin()
            histMCGenRec_u.SetBinContent(fakeBin,recoBin,fake_weight)
            histMCGenRec_u_up.SetBinContent(fakeBin,recoBin,fake_weight)
            histMCGenRec_u_dn.SetBinContent(fakeBin,recoBin,fake_weight)
	    #### fill data hist
            data_weight_u=data_pt_m_u[i][j]
            histDataReco_u.Fill(recoBin, data_weight_u)
            #print("Data weight ", data_weight_u, " or ", ptreco_mreco_uu.GetBinContent(i, j) , " for matrix reco bin ", recoBin)
            #### fill MC reco hist for comparison of inputs
            reco_weight_u=ptreco_mreco_u[i][j]
            histMCReco_u.Fill(recoBin, reco_weight_u)
            #print("Reco weight ungroomed ", reco_weight_u ," for matrix reco bin " , recoBin)
            reco_weight_g=ptreco_mreco_g[i][j]
            histMCReco_g.Fill(recoBin, reco_weight_g)
            #print("Reco weight groomed " , reco_weight_g , " for matrix reco bin " , recoBin)
            histMCReco_u_up.Fill(recoBin, reco_weight_u)
            ####  do same for up and down uncertainties for checks
            reco_weight_g_up=ptreco_mreco_g_up[i][j]
            histMCReco_g_up.Fill(recoBin, reco_weight_g_up)
            reco_weight_u_dn=ptreco_mreco_u_dn[i][j]
            histMCReco_u_dn.Fill(recoBin, reco_weight_u_dn)
            reco_weight_g_dn=ptreco_mreco_g_dn[i][j]
            histMCReco_g_dn.Fill(recoBin, reco_weight_g_dn)
	    #### gen loop: k is ptgen, l is mgen
            for k in range(n_ptgen_bin-1):
                for l in range(n_mgen_bin-1):
                    glob_genbin=(k)*(n_mgen_bin)+l
                    genBin=genBinning.GetGlobalBinNumber(mgen_edges[l],ptgen_edges[k])
     	            #### fill MC truth for closure test
	            #### ONLY FILL ONCE INSIDE i,j LOOP
                    if(i==0 and j==0):
                        #### fill Gen weights
                        #print("Bin l = ", l, " has mgen edge ", mgen_edges[l], " and bin k = ", k, " has ptgen edge ", ptgen_edges[k])
                        truth_weight_u = ptgen_mgen_u[k][l]
                        histMCTruth_u.Fill(genBin, truth_weight_u)
                        #print("Truth weight ", truth_weight_u, " for matrix gen bin ", genBin)
                        truth_weight_g = ptgen_mgen_g[k][l]
                        histMCTruth_g.Fill(genBin, truth_weight_g)
                        #### do same for uncertainties                                                                                                                    
                        truth_weight_u_up = ptgen_mgen_u_up[k][l]
                        histMCTruth_u_up.Fill(genBin, truth_weight_u_up)
                        truth_weight_g_up = ptgen_mgen_g_up[k][l]
                        histMCTruth_g_up.Fill(genBin, truth_weight_g_up)
                        truth_weight_u_dn = ptgen_mgen_u_dn[k][l]
                        histMCTruth_u_dn.Fill(genBin, truth_weight_u_dn)
                        truth_weight_g_dn = ptgen_mgen_g_dn[k][l]
                        histMCTruth_g_dn.Fill(genBin, truth_weight_g_dn)

                        #### Fill truth but binned in reco for comparison                                                                                                  
                        recoBin_genObj=recoBinning.GetGlobalBinNumber(mgen_edges[l],ptgen_edges[k])                                                                     
                        #### print("With pt edge " , ptgen_edges[k] , " for k == " , k , " and m edge " , mgen_edges[l]  ," for l ==  " , l , "and reco bin " , recoBin_genObj)
                        histMCTruth_RecoBinned_u.Fill(recoBin_genObj, truth_weight_u)
                        histMCTruth_RecoBinned_g.Fill(recoBin_genObj, truth_weight_g)
                    #### Get global bin number and ill response matrices
                    glob_bin = glob_recobin*((n_mgen_bin)*(n_ptgen_bin))+glob_genbin
	            #print("Global bin ", glob_bin, " for reco bin ", glob_recobin, " and gen bin ", glob_genbin)
	            #print("TUnfold gen bin ", genBin, " and reco bin ", recoBin, "for value", response_matrix_u[glob_bin])
	            #### fill ungroomed resp. matrices
                    resp_weight_u = response_matrix_u[i][j][k][l]
                    #resp_weight_u = response_matrix_u.flatten()[glob_bin]
                    print("Response weight for index i ",  i, " j ", j, " k ", k, " l ", l, " in unflattened matrix ", response_matrix_u[i][j][k][l])
                    print("and weight for global index ", glob_bin, " in flattened matrix ", response_matrix_u.flatten()[glob_bin])
                    histMCGenRec_u.Fill(genBin,recoBin,resp_weight_u)
                    resp_weight_u_up = response_matrix_u_up[i][j][k][l]
                    histMCGenRec_u_up.Fill(genBin,recoBin,resp_weight_u_up)
                    resp_weight_u_dn = response_matrix_u_dn[i][j][k][l]
                    histMCGenRec_u_dn.Fill(genBin,recoBin,resp_weight_u_dn)
                    #### fill groomed resp. matrices
                    resp_weight_g = response_matrix_g[i][j][k][l]
                    histMCGenRec_g.Fill(genBin,recoBin,resp_weight_g)
                    resp_weight_g_up = response_matrix_g_up[i][j][k][l]
                    histMCGenRec_g_up.Fill(genBin,recoBin,resp_weight_g_up)
                    resp_weight_g_dn = response_matrix_g_dn[i][j][k][l]
                    histMCGenRec_g_dn.Fill(genBin,recoBin,resp_weight_g_dn)
    print("MC Reco (filled with MC gen but reco bins)")
    histMCReco_u.Print("base")
    print("MC Truth (filled with MC gen and gen bins)")
    histMCTruth_u.Print("base")
    print("Response matrix")
    histMCGenRec_u.Print("base")
    
    #### check that response matrix has been filled properly
    histMCReco_u_M=histMCGenRec_u.ProjectionY("MCReco ungroomed")
    histMCTruth_u_M=histMCGenRec_u.ProjectionX("MCTruth ungroomed")
    histMCReco_u_M.Print("all")
    histMCReco_g_M=histMCGenRec_g.ProjectionY("MCReco groomed")
    histMCTruth_g_M=histMCGenRec_g.ProjectionX("MCTruth groomed")
    
    plotinputsROOT(histMCGenRec_u, histMCTruth_u, histMCReco_u, groom="ungroomed", syst=syst, year=year)
    plotinputsROOT(histMCGenRec_g, histMCTruth_g, histMCReco_g, groom="groomed", syst=syst, year=year)
    
    #### convert from pyroot to hist for easy plotting
    resp_u=uproot.pyroot.from_pyroot(histMCGenRec_u)
    print(type(resp_u), resp_u.title)
    print("Hist object: ", resp_u.to_hist())
    print("Axes names ", [ax.name for ax in resp_u.to_hist().axes])
    print("Gen bins: ", [bin for bin in resp_u.to_hist().project("xaxis").axes[0]])
    print("Reco bins: ", [bin for bin in resp_u.to_hist().project("yaxis").axes[0]])
    resp_g = uproot.pyroot.from_pyroot(histMCGenRec_g).to_hist()
    mctruth_u= uproot.pyroot.from_pyroot(histMCTruth_u).to_hist()
    mctruth_g= uproot.pyroot.from_pyroot(histMCTruth_g).to_hist()
    mcreco_u= uproot.pyroot.from_pyroot(histMCReco_u).to_hist()
    mcreco_g = uproot.pyroot.from_pyroot(histMCReco_g).to_hist()
    ugly_axes = [ax.name for ax in resp_u.to_hist().axes]
    fig, ax = plt.subplots()
    hep.hist2dplot(resp_g, ax=ax)
    plt.savefig(os_path+"/histrespmatrix_g"+syst+".png")
    fig, ax = plt.subplots()
    hep.histplot(mctruth_g, ax=ax, label="MC truth")
    hep.histplot(resp_g.project('xaxis'), ax=ax, label="MC truth from Matrix")
    leg = ax.legend(loc='best', labelspacing=0.25)
    leg.set_visible(True)
    plt.savefig(os_path+"/histmcinput_truth_g"+syst+year+".png")
    fig, ax = plt.subplots()
    hep.histplot(mcreco_g, ax=ax, label="MC reco")
    hep.histplot(resp_g.project('yaxis'), ax=ax, label="MC reco from Matrix")
    leg = ax.legend(loc='best', labelspacing=0.25)
    leg.set_visible(True)
    plt.savefig(os_path+"/histmcinput_reco_g"+syst+year+".png")
    plt.show()
    #### plot same checks for each bin
    for i in range(n_ptreco_bin):
        print("IDK YET")
