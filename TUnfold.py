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
#ROOT.gInterpreter.ProcessLine('#include "MyTUnfoldDensity.h"')
print(coffea.__version__)
print(uproot.__version__)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--year', choices=['2016', '2017', '2018', '2016APV', None], default="None", help="Year to run on")
parser.add_argument('--syst', default=['jer', 'PUSF', 'L1PreFiringWeight'], nargs='+')
parser.add_argument('--allUncSrcs', action='store_true', help='Run processor for each unc. source separately')
parser.add_argument('-i', '--input', required=False, help='MC input pkl file')
parser.add_argument('-d', '--dataInput', default=None,  help='Data input pkl file; if none provided only run closure test')                    
arg = parser.parse_args("")

import os
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#### define plotting functions
def plotinputsROOT(matrix, truth, reco, groom="", syst="", year="", ospath=''):
            histMCReco_M=matrix.ProjectionY("MCReco "+groom)
            histMCTruth_M=matrix.ProjectionX("MCTruth "+groom)
            c1 = ROOT.TCanvas("c1","Plot MC input "+groom+" binned by pt (outer) and mass (inner)",1200,400)
            c1.Divide(2,1)
            c1.cd(1)
            histMCTruth_M.SetMarkerStyle(21)
            histMCTruth_M.SetLineColor(ROOT.kRed)
            histMCTruth_M.SetMarkerColor(ROOT.kRed)
            histMCTruth_M.Draw("E")
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
            histMCReco_M.Draw("E")
            reco.SetMarkerStyle(24)
            reco.SetLineColor(ROOT.kBlue)
            reco.SetMarkerColor(ROOT.kBlue)
            reco.Draw("SAME")
            leg1_2 = ROOT.TLegend(0.7, 0.7, 0.86, 0.86)
            leg1_2.AddEntry(histMCReco_M, "MC Reco from M", "p")
            leg1_2.AddEntry(reco, "MC Reco", "p")
            leg1_2.Draw()
            c1.Draw()
            c1.SaveAs(ospath+"MCInput_"+groom+"_"+syst+"flatmatrix"+year+".png")
            c1.Close()
def CompareCoffeaROOT(result, syst_hist_dict, os_path, groomed=False, syst="nominal"):
    if groomed:
        end = "_g"
    else:
        end = "_u"
    m_coffea = result["response_matrix"+end][{'syst':syst}]
    
    h_coffea = result["ptreco_mreco"+end][{'syst':syst}]
    htrue_coffea = result["ptgen_mgen"+end][{'syst':syst}]
    
    h_root = syst_hist_dict[syst]["MCReco"+end]
    htrue_root = syst_hist_dict[syst]["MCTruth"+end]
    
    h_m_root = syst_hist_dict[syst]['MCGenRec'+end].ProjectionY("MCReco")
    htrue_m_root = syst_hist_dict[syst]['MCGenRec'+end].ProjectionX("MCTruth")
    
    ptreco_edges = [bin[0] for bin in result["ptreco_mreco"+end].project('ptreco').axes[0]] + [result["ptreco_mreco"+end].project('ptreco').axes[0][-1][1]]
    ptgen_edges = [bin[0] for bin in result["ptgen_mgen"+end].project('ptgen').axes[0]] + [result["ptgen_mgen"+end].project('ptgen').axes[0][-1][1]]
    mreco_edges = [bin[0] for bin in result["ptreco_mreco"+end].project('mreco').axes[0]] + [result["ptreco_mreco"+end].project('mreco').axes[0][-1][1]]
    mgen_edges = [bin[0] for bin in result["ptgen_mgen"+end].project('mgen').axes[0]]+ [result["ptgen_mgen"+end].project('mgen').axes[0][-1][1]]
    for ipt in range(len(ptreco_edges)-1):
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7,7),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True)
        root_reco_vals = np.zeros(len(mreco_edges)-1)
        root_m_reco_vals = np.zeros(len(mreco_edges)-1)
        root_reco_errs = np.zeros(len(mreco_edges)-1)
        root_m_reco_errs = np.zeros(len(mreco_edges)-1)
        for im in range(len(mreco_edges)-1):
            root_reco_vals[im] = h_root.GetBinContent(im+2+(ipt+1)*(len(mreco_edges)-1))
            root_m_reco_vals[im] = h_m_root.GetBinContent(im+2+(ipt+1)*(len(mreco_edges)-1))
            root_reco_errs[im] = h_root.GetBinError(im+2+(ipt+1)*(len(mreco_edges)-1))
            root_m_reco_errs[im] = h_m_root.GetBinError(im+2+(ipt+1)*(len(mreco_edges)-1))
        ratio_vals = np.divide(m_coffea[{'ptreco':ipt}].project('mreco').values(), root_m_reco_vals,
                out=np.empty(np.array(root_m_reco_vals.shape)).fill(np.nan),
                where=root_m_reco_vals!= 0)
        ratio_errs = np.divide(m_coffea[{'ptreco':ipt}].project('mreco').variances()**0.5, root_m_reco_vals,
                out=np.empty(np.array(root_m_reco_vals.shape)).fill(np.nan),
                where=root_m_reco_vals!= 0)
        hep.histplot(h_coffea[{'ptreco':ipt}].project('mreco'), stack=False, histtype='errorbar',
                         ax=ax, density=False, marker =["o"], color = 'blue', binwnorm=True,
                         label=['Coffea hist ' + str(ptreco_edges[ipt])])
        hep.histplot(root_reco_vals, mreco_edges, stack=False, histtype='errorbar', yerr= root_reco_errs,
                         ax=ax, density=False, marker =["*"], color = 'green', binwnorm=True,
                         label=['Root hist ' + str(ptreco_edges[ipt])])
        hep.histplot(m_coffea[{'ptreco':ipt}].project('mreco'), stack=False, histtype='errorbar',
                         ax=ax, density=False, marker =["."], color = 'red', binwnorm=True,
                         label=['Coffea matrix ' + str(ptreco_edges[ipt])])
        hep.histplot(root_m_reco_vals, mreco_edges, stack=False, histtype='step', yerr= root_m_reco_errs,
                         ax=ax, density=False, color = 'magenta', binwnorm=True,
                         label=['Root hist ' + str(ptreco_edges[ipt])])
        hep.histplot(ratio_vals,mreco_edges, stack=False, histtype='errorbar',yerr=ratio_errs,
                         ax=rax, density=False, marker =["."], color = 'red',
                         label=['Coffea/Root matrix ' + str(ptreco_edges[ipt])])
        rax.set_ylabel("Coffea/Root")
        ax.set_ylabel("Events/Bin Width (Gev^-1)")
        leg = ax.legend(loc='best', labelspacing=0.25)
        leg.set_visible(True)
        fig.savefig(os_path+"CoffeaROOT_reco_"+syst+'_pt'+str(ptreco_edges[ipt])+end+".png")
    for ipt in range(len(ptgen_edges)-1):
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7,7),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True)
        root_gen_vals = np.zeros(len(mgen_edges)-1)
        root_m_gen_vals = np.zeros(len(mgen_edges)-1)
        root_gen_errs = np.zeros(len(mgen_edges)-1)
        root_m_gen_errs = np.zeros(len(mgen_edges)-1)
        for im in range(len(mgen_edges)-1):
            root_gen_vals[im] = htrue_root.GetBinContent(im+2+(ipt+1)*(len(mgen_edges)-1))
            root_m_gen_vals[im] = htrue_m_root.GetBinContent(im+2+(ipt+1)*(len(mgen_edges)-1))
            root_gen_errs[im] = htrue_root.GetBinError(im+2+(ipt+1)*(len(mgen_edges)-1))
            root_m_gen_errs[im] = htrue_m_root.GetBinError(im+2+(ipt+1)*(len(mgen_edges)-1))
        ratio_vals = np.divide(m_coffea[{'ptgen':ipt}].project('mgen').values(), root_m_gen_vals,
                out=np.empty(np.array(root_m_gen_vals.shape)).fill(np.nan),
                where=root_m_gen_vals!= 0)
        ratio_errs = np.divide(m_coffea[{'ptgen':ipt}].project('mgen').variances()**0.5, root_m_gen_vals,
                out=np.empty(np.array(root_m_gen_vals.shape)).fill(np.nan),
                where=root_m_gen_vals!= 0)
        hep.histplot(htrue_coffea[{'ptgen':ipt}].project('mgen'), stack=False, histtype='errorbar',
                         ax=ax, density=False, marker =["o"], color = 'blue', binwnorm=True,
                         label=['Coffea hist ' + str(ptgen_edges[ipt])])
        hep.histplot(root_gen_vals, mgen_edges, stack=False, histtype='errorbar', yerr= root_gen_errs,
                         ax=ax, density=False, marker =["*"], color = 'green', binwnorm=True,
                         label=['Root hist ' + str(ptgen_edges[ipt])])
        hep.histplot(m_coffea[{'ptgen':ipt}].project('mgen'), stack=False, histtype='errorbar',
                         ax=ax, density=False, marker =["."], color = 'red', binwnorm=True,
                         label=['Coffea matrix ' + str(ptgen_edges[ipt])])
        hep.histplot(root_m_gen_vals, mgen_edges, stack=False, histtype='step', yerr= root_m_gen_errs,
                         ax=ax, density=False, color = 'magenta', binwnorm=True,
                         label=['Root hist ' + str(ptgen_edges[ipt])])
        hep.histplot(ratio_vals,mgen_edges, stack=False, histtype='errorbar',yerr=ratio_errs,
                         ax=rax, density=False, marker =["."], color = 'red',
                         label=['Coffea/Root matrix ' + str(ptreco_edges[ipt])])
        rax.set_ylabel("Cfea/ROOT")
        ax.set_ylabel("Events/Bin Width (Gev^-1)")
        leg = ax.legend(loc='best', labelspacing=0.25)
        leg.set_visible(True)
        fig.savefig(os_path+"CoffeaROOT_GEN_"+syst+'_pt'+str(ptgen_edges[ipt])+end+".png")
#### functions for setting up response matrix
def setupBinning(result):
    binning_dict = {}
    response_matrix_u, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'][{'syst':"nominal"}].project("ptreco", "mreco", "ptgen", "mgen").to_numpy(flow=True)
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    n_ptgen_bin = len(ptgen_edges)-1
    n_mgen_bin= len(mgen_edges)-1
    mreco_edges = mreco_edges[1:-1]
    ptreco_edges = ptreco_edges[:-1]
    ptreco_edges[0] = 0.
    mgen_edges = mgen_edges[1:-1]
    ptgen_edges = ptgen_edges[:-1]
    ptgen_edges[0] = 0.
    # print("ptgen edges ", ptgen_edges)
    # print( "Size of response matrix ", len(response_matrix_u.flatten()))
    #### make TUnfold binning axes
    # print("Nbins ptreco ", n_ptreco_bin, " Nbins mreco", n_mreco_bin)
    # print("Bins ptreco ", ptreco_edges, " bins mreco", mreco_edges)
    detectorBinning = ROOT.TUnfoldBinning("detector")
    recoBinning = detectorBinning.AddBinning("reco")
    # recoBinning.AddAxis(ptreco_mreco_u.GetYaxis(), False, False) # mreco
    # recoBinning.AddAxis(ptreco_mreco_u.GetXaxis(), False, False) # ptreco
    recoBinning.AddAxis("m_{RECO}", n_mreco_bin-2, mreco_edges, False, False) # mreco
    recoBinning.AddAxis("pt_{RECO}", n_ptreco_bin-1, ptreco_edges, False, False) # ptreco
    # print("Nbins ptgen ", n_ptgen_bin, " Nbins mgen", n_mgen_bin)
    # print("Bins ptgen ", ptgen_edges, " bins mgen", mgen_edges)
    generatorBinning = ROOT.TUnfoldBinning("generator")
    fakeBinning=generatorBinning.AddBinning("fakesBin", 1)
    genBinning = generatorBinning.AddBinning("gen")
    # genBinning.AddAxis(ptgen_mgen_u.GetYaxis(), False, False) # mgen
    # genBinning.AddAxis(ptgen_mgen_u.GetXaxis(), False, False) # ptgen
    genBinning.AddAxis("m_{GEN}", n_mgen_bin-2, mgen_edges, False, False) #mgen
    genBinning.AddAxis("pt_{GEN}", n_ptgen_bin-1, ptgen_edges, False, False) #ptgen 
    return detectorBinning, generatorBinning, mreco_edges, ptreco_edges, mgen_edges, ptgen_edges

def fillData(result, detectorBinning, mreco_edges, ptreco_edges, new=True):
    #### create data histograms
    if new:
        reco_str = "ptreco_mreco"
        gen_str = "ptgen_mgen"
    else:
        reco_str = "jet_pt_mass_reco"
        gen_str = "jet_pt_mass_reco"
    data_pt_m_u =  result[reco_str+"_u"][{'syst':"nominal"}].project('ptreco', 'mreco').values(flow=True)
    data_pt_m_g = result[reco_str+"_g"][{'syst':"nominal"}].project('ptreco', 'mreco').values(flow=True)
    recoBinning = detectorBinning.FindNode("reco")
    DataReco_u=recoBinning.CreateHistogram("histDataReco Ungroomed")  #data values in reco binnning --> input to unfolding
    DataReco_g=recoBinning.CreateHistogram("histDataReco Groomed")  #data values in reco binnning --> input to unfolding
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    for i in range(n_ptreco_bin-1):
        for j in range(n_mreco_bin-1):
    	    #### fill data hist
            recoBin=recoBinning.GetGlobalBinNumber(mreco_edges[j],ptreco_edges[i])
            data_weight_u=data_pt_m_u[i][j]
            DataReco_u.SetBinContent(recoBin, data_weight_u)
            data_weight_g=data_pt_m_g[i][j]
            DataReco_g.SetBinContent(recoBin, data_weight_g)
            #print("Data weight ", data_weight_u, " or ", ptreco_mreco_uu.GetBinContent(i, j) , " for matrix reco bin ", recoBin)
    return DataReco_u, DataReco_g
def getHists(result, syst, detectorBinning, generatorBinning, new=True):
    if new:
        reco_str = "ptreco_mreco"
        gen_str = "ptgen_mgen"
    else:
        reco_str = "jet_pt_mass_reco"
        gen_str = "jet_pt_mass_gen"
    hist = {}
    ptgen_mgen_u=result[gen_str+"_u"][{'syst':syst}].project('ptgen', 'mgen').values(flow=True)
    ptgen_mgen_g=result[gen_str+"_g"][{'syst':syst}].project('ptgen', 'mgen').values(flow=True)
    genErr_u=np.sqrt(result[gen_str+"_u"][{'syst':syst}].project('ptgen', 'mgen').variances(flow=True))
    genErr_g=np.sqrt(result[gen_str+"_g"][{'syst':syst}].project('ptgen', 'mgen').variances(flow=True))
    #### in future datasets input for fakes and misses depends on syst as well
    fakes_ptreco_mreco = result["fakes"][{'syst':syst}].project('ptreco', 'mreco').values(flow=True)
    fakesErr = np.sqrt(result["fakes"][{'syst':syst}].project('ptreco', 'mreco').variances(flow=True))
    hist['misses_ptgen_mgen'] = uproot.pyroot.to_pyroot(uproot.writing.identify.to_writable(result["misses"][{'syst':"nominal"}]))
    ptreco_mreco_u  = result[reco_str+"_u"][{'syst':syst}].project('ptreco', 'mreco').values(flow=True)
    ptreco_mreco_g = result[reco_str+"_g"][{'syst':syst}].project('ptreco', 'mreco').values(flow=True)
    recoErr_u=np.sqrt(result[reco_str+"_u"][{'syst':syst}].project('ptreco', 'mreco').variances(flow=True))
    recoErr_g=np.sqrt(result[reco_str+"_g"][{'syst':syst}].project('ptreco', 'mreco').variances(flow=True))
    response_matrix_u, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").to_numpy(flow=True)
    response_matrix_g = result['response_matrix_g'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").values(flow=True)
    respErr_u = np.sqrt(result['response_matrix_u'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").variances(flow=True))
    respErr_g = np.sqrt(result['response_matrix_g'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").variances(flow=True))
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    n_ptgen_bin = len(ptgen_edges)-1
    n_mgen_bin= len(mgen_edges)-1
    mreco_edges = mreco_edges[1:-1]
    ptreco_edges = ptreco_edges[:-1]
    ptreco_edges[0] = 0.
    mgen_edges = mgen_edges[1:-1]
    ptgen_edges = ptgen_edges[:-1]
    ptgen_edges[0] = 0.
    # print( "Size of response matrix ", len(response_matrix_u.flatten()))
    # create histogram of migrations and gen hists
    hist['MCGenRec_u']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Ungroomed "+syst)
    hist['MCGenRec_g']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Groomed "+syst)
    #### get gen, reco, and fake binning
    recoBinning = detectorBinning.FindNode("reco")
    genBinning = generatorBinning.FindNode("gen")
    fakeBinning = generatorBinning.FindNode("fakesBin")
    #### create MC hists for comparison
    hist['MCReco_u']=recoBinning.CreateHistogram("histMCReco Ungroomed "+syst) #gen values in reco binning --> htruef in Sal's example
    hist['MCTruth_u']=genBinning.CreateHistogram("histMCTruth Ungroomed "+syst)  #gen values in gen binning    
    hist['MCReco_g']=recoBinning.CreateHistogram("histMCReco Groomed "+syst) #gen values in reco binning --> htruef in Sal's example
    hist['MCTruth_g']=genBinning.CreateHistogram("histMCTruth Groomed "+syst)  #gen values in gen binning
    # hist['MCTruth_RecoBinned_u']=recoBinning.CreateHistogram("histMCTruth Reco Binned, Ungroomed") #gen values in reco binning --> htruef in Sal's example 
    # hist['MCTruth_RecoBinned_g']=recoBinning.CreateHistogram("histMCTruth Reco Binned, Groomed") #gen values in reco binning --> htruef in Sal's example  
    #### Loop through reco and gen bins of MC input and fill hist of migrations
    #### reco loop: i is ptreco, j is mreco
    for i in range(n_ptreco_bin-1):
        for j in range(n_mreco_bin-1):
            glob_recobin=(i)*(n_mreco_bin)+j
            # print("Bin j = " , j , " has edge mreco " , mreco_edges[j] , "and  bin i " , i , " has ptreco edge " , ptreco_edges[i])
            recoBin=recoBinning.GetGlobalBinNumber(mreco_edges[j],ptreco_edges[i])
	    #### only fill fakes in fake genBin
            fake_weight = fakes_ptreco_mreco[i][j]
            #print("Fake weight ", fake_weight," for i == ",i, " and j == ",j)
            fakeBin=fakeBinning.GetStartBin()
            hist['MCGenRec_u'].SetBinContent(fakeBin,recoBin,fake_weight)
            hist['MCGenRec_u'].SetBinError(fakeBin,recoBin,fakesErr[i][j])
            #### fill MC reco hist for comparison of inputs
            reco_weight_u=ptreco_mreco_u[i][j]
            hist['MCReco_u'].SetBinContent(recoBin, reco_weight_u)
            hist['MCReco_u'].SetBinError(recoBin, recoErr_u[i][j])
            #print("Reco weight ungroomed ", reco_weight_u ," for matrix reco bin " , recoBin)
            reco_weight_g=ptreco_mreco_g[i][j]
            hist['MCReco_g'].SetBinContent(recoBin, reco_weight_g)
            hist['MCReco_g'].SetBinError(recoBin, recoErr_g[i][j])
            #print("Reco weight groomed " , reco_weight_g , " for matrix reco bin " , recoBin)            
	    #### gen loop: k is ptgen, l is mgen
            for k in range(n_ptgen_bin-1):
                for l in range(n_mgen_bin-1):
                    glob_genbin=(k)*(n_mgen_bin)+l
                    genBin=genBinning.GetGlobalBinNumber(mgen_edges[l],ptgen_edges[k])
     	            #### fill MC truth for closure test
	            #### ONLY FILL ONCE INSIDE i,j LOOP
                    if(i==0 and j==0):
                        #### fill Gen weights
                        # print("Bin l = ", l, " has mgen edge ", mgen_edges[l], " and bin k = ", k, " has ptgen edge ", ptgen_edges[k])
                        truth_weight_u = ptgen_mgen_u[k][l+1]
                        hist['MCTruth_u'].SetBinContent(genBin, truth_weight_u)
                        hist['MCTruth_u'].SetBinError(genBin, genErr_u[k][l+1])
                        #print("Truth weight ", truth_weight_u, " for matrix gen bin ", genBin)
                        truth_weight_g = ptgen_mgen_g[k][l+1]
                        hist['MCTruth_g'].SetBinContent(genBin, truth_weight_g)
                        hist['MCTruth_g'].SetBinError(genBin, genErr_g[k][l+1])
                        #### SetBinContent truth but binned in reco for comparison
                        recoBin_genObj=recoBinning.GetGlobalBinNumber(mgen_edges[l],ptgen_edges[k])
                        #### print("With pt edge " , ptgen_edges[k] , " for k == " , k , " and m edge " , mgen_edges[l]  ," for l ==  " , l , "and reco bin " , recoBin_genObj)
                        # hist['MCTruth_RecoBinned_u'].SetBinContent(recoBin_genObj, truth_weight_u)
                        # hist['MCTruth_RecoBinned_g'].SetBinContent(recoBin_genObj, truth_weight_g)
                    #### Get global bin number and ill response matrices
                    glob_bin = glob_recobin*((n_mgen_bin)*(n_ptgen_bin+1))+glob_genbin
	            #print("Global bin ", glob_bin, " for reco bin ", glob_recobin, " and gen bin ", glob_genbin)
	            #print("TUnfold gen bin ", genBin, " and reco bin ", recoBin, "for value", response_matrix_u[glob_bin])
                    
	            #### fill ungroomed resp. matrices
                    resp_weight_u = response_matrix_u[i][j][k][l+1]
                    #resp_weight_u = response_matrix_u.flatten()[glob_bin]
                    # print("Response weight for index i ",  i, " j ", j, " k ", k, " l ", l, " in unflattened matrix ", response_matrix_u[i][j][k][l])
                    # print("and weight for global index ", glob_bin, " in flattened matrix ", response_matrix_u.flatten()[glob_bin])
                    hist['MCGenRec_u'].SetBinContent(genBin,recoBin,resp_weight_u)
                    hist['MCGenRec_u'].SetBinError(genBin,recoBin,respErr_u[i][j][k][l+1])
                    #### fill groomed resp. matrices
                    resp_weight_g = response_matrix_g[i][j][k][l+1]
                    hist['MCGenRec_g'].SetBinContent(genBin,recoBin,resp_weight_g)
                    hist['MCGenRec_g'].SetBinError(genBin,recoBin,respErr_g[i][j][k][l+1])
    # print("Response matrix")
    # hist['MCGenRec_u'].Print("base")
    hist['fakes_ptreco_mreco'] = fakes_ptreco_mreco
    hist['ptreco_mreco_u'] = ptreco_mreco_u
    hist['ptreco_mreco_g'] = ptreco_mreco_g
    hist['ptgen_mgen_u'] = ptgen_mgen_u
    hist['ptgen_mgen_g'] = ptgen_mgen_g
    hist['response_matrix_u'] = response_matrix_u
    hist['response_matrix_g'] = response_matrix_g
    del fakes_ptreco_mreco, ptreco_mreco_u, ptreco_mreco_g, ptgen_mgen_u, ptgen_mgen_g, response_matrix_u, response_matrix_g
    return hist
#### open files
if __name__ == "__main__":
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
    checkdir(os_path)
    year = fname[-8:-4]
    print(year)
    #### get 
    axis_names = [ax.name for ax in result['jet_pt_mass_reco_u'].axes]
    print("avail hists ", result.keys())
    print(axis_names)
    cats = [cat for cat in result['jet_pt_mass_reco_u'].project('ptreco', 'mreco').axes[0]]
    availSysts = [ax for ax in result['jet_pt_mass_reco_u'].project("syst").axes[0]]
    print("Available systs ", availSysts)
    print("Non nominal systs ", [syst for syst in availSysts if ("Up" in syst)])
    detectorBinning, generatorBinning, mreco_edges, ptreco_edges, mgen_edges, ptgen_edges = setupBinning(result)
    #### for data get results once
    DataReco_u, DataReco_g = fillData(result_data, detectorBinning, mreco_edges, ptreco_edges)
    syst_hist_dict = {}
    #hist = getHists(result, "nominal", detectorBinning, generatorBinning)
    #syst_hist_dict["nominal"]=hist
    #for syst in [syst for syst in availSysts if ("Up" in syst)]:
    for syst in ["nominal"]:
        ROOT.TH1.SetDefaultSumw2(False)
        #syst = syst[:-2]
        hist = getHists(result, syst, detectorBinning, generatorBinning)
        syst_hist_dict[syst]=hist
        print("Building respone matrices for systematic ", syst)
        #### check that response matrix has been filled properly
        MCReco_u_M=hist['MCGenRec_u'].ProjectionY("MCReco ungroomed")
        MCTruth_u_M=hist['MCGenRec_u'].ProjectionX("MCTruth ungroomed")
        #MCReco_u_M.Print("all")
        MCReco_g_M=hist['MCGenRec_g'].ProjectionY("MCReco groomed")
        MCTruth_g_M=hist['MCGenRec_g'].ProjectionX("MCTruth groomed")
        plotinputsROOT(hist['MCGenRec_u'], hist['MCTruth_u'], hist['MCReco_u'], groom="ungroomed", syst=syst, year=year)
        plotinputsROOT(hist['MCGenRec_g'], hist['MCTruth_g'], hist['MCReco_g'], groom="groomed", syst=syst, year=year)
        hist['MCTruth_u'].Print("all")
        #### plot inputs and check that they match matrix
        for ipt in range(len(ptgen_edges)-1): 
            ci = ROOT.TCanvas("c" + str(ipt), "c" + str(ipt))
            ci.cd()
            hcopy = ROOT.TH1D("inputGenPt" + str(ipt), "Input Gen pT " + str(ptgen_edges[ipt]) +" - " +str(ptgen_edges[ipt+1]), len(mgen_edges)-1, mgen_edges )
            hcopy_M = ROOT.TH1D("inputMATRIXGenPt" + str(ipt), "InputMATRIX Gen pT " + str(ptgen_edges[ipt]) +" - " +str(ptgen_edges[ipt+1]), len(mgen_edges)-1, mgen_edges )
            # print("Length of mgen edges: ", len(mgen_edges), "mgen edges", mgen_edges)
            #        hcopy.Print("all")
            for im in range(len(mgen_edges)):
                # print("Bin value: ", hist['MCTruth_u'].GetBinContent(im+1+ipt*(len(mgen_edges)-1)), " for pt bin ", ipt, " and mass bin ", im, " ", mgen_edges[im])
                # print("Error value: ", np.sqrt(hist['MCTruth_u'].GetBinError(im+1+ipt*(len(mgen_edges)-1))), " from truth bin ", im+2+ipt*(len(mgen_edges)-1))
                hcopy.SetBinContent(im, hist['MCTruth_u'].GetBinContent(im+1+ipt*(len(mgen_edges)-1))) 
                hcopy.SetBinError(im, hist['MCTruth_u'].GetBinError(im+1 + ipt*(len(mgen_edges)-1)))
                # hcopy_M.SetBinContent(im, MCTruth_u_M.GetBinContent(im+1+ipt*(len(mgen_edges)-1)))
                # hcopy_M.SetBinError(im, np.sqrt(MCTruth_u_M.GetBinError(im+1 + ipt*(len(mgen_edges)-1))))
                # hcopy.SetMarkerStyle(24)
            hcopy.SetLineColor(ROOT.kBlue)
            hcopy.SetMarkerColor(ROOT.kBlue)
            hcopy.GetXaxis().SetTitle("Gen Mass TeV")
            hcopy.Draw()
            # hcopy_M.SetMarkerStyle(21)
            # hcopy_M.SetLineColor(ROOT.kRed)
            # hcopy_M.SetMarkerColor(ROOT.kRed)
            # hcopy_M.GetXaxis().SetTitle("Gen Mass TeV")
            # hcopy_M.Draw("SAME")
            leg1 = ROOT.TLegend(0.7, 0.7, 0.86, 0.86)
            # leg1.AddEntry(hcopy_M, "MC Gen from M", "p")
            leg1.AddEntry(hcopy, "MC Gen", "p")
            leg1.Draw()
            #ci.Draw()
            hist[hcopy.GetName()]=hcopy
            ci.SaveAs(os_path+"MCInput_ungroomed_"+syst+'_pt'+str(ptgen_edges[ipt])+"_"+year+".png")
            ci.Close()
            #### add resp matrices and other hists to dictionary for final result
            syst_hist_dict[syst]=hist
    #### convert from pyroot to hist for easy plotting
    # resp_u=uproot.pyroot.from_pyroot(histMCGenRec_u)
    # print(type(resp_u), resp_u.title)
    # print("Hist object: ", resp_u.to_hist())
    # print("Axes names ", [ax.name for ax in resp_u.to_hist().axes])
    # print("Gen bins: ", [bin for bin in resp_u.to_hist().project("xaxis").axes[0]])
    # print("Reco bins: ", [bin for bin in resp_u.to_hist().project("yaxis").axes[0]])
    # resp_g = uproot.pyroot.from_pyroot(histMCGenRec_g).to_hist()
    # mctruth_u= uproot.pyroot.from_pyroot(histMCTruth_u).to_hist()
    # mctruth_g= uproot.pyroot.from_pyroot(histMCTruth_g).to_hist()
    # mcreco_u= uproot.pyroot.from_pyroot(histMCReco_u).to_hist()
    # mcreco_g = uproot.pyroot.from_pyroot(histMCReco_g).to_hist()
    # ugly_axes = [ax.name for ax in resp_u.to_hist().axes]
    # fig, ax = plt.subplots()
    # hep.hist2dplot(resp_g, ax=ax)
    # plt.savefig(os_path+"/histrespmatrix_g"+syst+".png")
    # fig, ax = plt.subplots()
    # hep.histplot(mctruth_g, ax=ax, label="MC truth")
    # hep.histplot(resp_g.project('xaxis'), ax=ax, label="MC truth from Matrix")
    # leg = ax.legend(loc='best', labelspacing=0.25)
    # leg.set_visible(True)
    # plt.savefig(os_path+"/histmcinput_truth_g"+syst+year+".png")
    # fig, ax = plt.subplots()
    # hep.histplot(mcreco_g, ax=ax, label="MC reco")
    # hep.histplot(resp_g.project('yaxis'), ax=ax, label="MC reco from Matrix")
    # leg = ax.legend(loc='best', labelspacing=0.25)
    # leg.set_visible(True)
    # plt.savefig(os_path+"/histmcinput_reco_g"+syst+year+".png")
    # plt.show()
