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
            print(histMCTruth_M.GetName())
            print(matrix.GetName())
            c1 = ROOT.TCanvas("c"+groom,"Plot MC input "+groom+" binned by pt (outer) and mass (inner)",1200,400)
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
            leg1.AddEntry(truth, "MC Gen", "p")
            leg1.Draw()
            c1.cd(2)
            reco.SetMarkerStyle(24)
            reco.SetLineColor(ROOT.kBlue)
            reco.SetMarkerColor(ROOT.kBlue)
            histMCReco_M.SetMarkerStyle(21)
            histMCReco_M.SetLineColor(ROOT.kRed)
            histMCReco_M.SetMarkerColor(ROOT.kRed)
            histMCReco_M.Draw("E")
            reco.Draw("SAME")
            leg1_2 = ROOT.TLegend(0.7, 0.7, 0.86, 0.86)
            leg1_2.AddEntry(histMCReco_M, "MC Reco from M", "p")
            leg1_2.AddEntry(reco, "MC Reco", "p")
            leg1_2.Draw()
            c1.Draw()
            c1.SaveAs(ospath+"MCInput_"+groom+"_"+syst+"flatmatrix"+year+".png")
            return c1
        
        
def plotUnfoldOutputHist(htrue, u, groomed=False, norm=True, os_path='', channel='', IOV=''): #, oMat, oSys, oTotal):                                                 
    #### plotting options                                                                                                                                     
    tot_error_opts = {
            'label': 'Stat. Unc.',
            'facecolor': 'powderblue',
            'linewidth': 0
        }
    stat_error_opts = {
            'label': 'Stat. Unc.',
                    'hatch': '///',
                    'edgecolor': 'black',
            'facecolor': 'none',
            'linewidth': 0
        }
    data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 10.,
            'color': 'k',
            'elinewidth': 1,
        }
    #### u is total unfolding object, htrue is truth hist from coffea                                                                                         
    o = u.GetOutput("u")
    mCovInput = u.GetEmatrixInput("Input unc")
    inputErrTot = np.array([mCovInput.GetBinContent(i,i)**0.5 for i in range(0, o.GetNbinsX()+1)])
    ptedges = [bin[0] for bin in htrue.project("ptgen").axes[0]] + [htrue.project("ptgen").axes[0][-1][1]]
    print(ptedges)
    medges = [bin[0] for bin in htrue.project('mgen').axes[0]]+ [htrue.project('mgen').axes[0][-1][1]]
    widths = htrue.project("mgen").axes[0].widths
    xlim = medges[-1]
    if groomed: os_path=os_path+"Groomed"
    for ipt in range(1,len(ptedges)):
        j = ipt-1 #index for coffea bc no underflow bin                                                                                                       
        inputErr = np.array([inputErrTot[(im+2+ipt*(len(medges)-1))] for im in range(0, len(medges)-1)])
        oErr = np.array([o.GetBinError(im+2+ipt*(len(medges)-1)) for im in range(0, len(medges)-1)])
        oVals = np.array([o.GetBinContent(im+2+ipt*(len(medges)-1)) for im in range(0, len(medges)-1)])
        oHistVals = np.array([o.GetBinContent(im+2+ipt*(len(medges)-1)) for im in range(0, len(medges)-1)])
        oHistErr = oErr
        hist = htrue[{'ptgen':j, 'syst':"nominal"}].project("mgen")
        #### set up figure                                                                                                                                    
        fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
        ax.set_ylabel(r'$\frac{Events}{Bin Size} (GeV^{-1})$', loc = 'top')
        ax.text(0.60, 0.70, str(ptedges[j])+r"$<p_{T}<$" +str(ptedges[j+1]) + " GeV",
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='green', fontsize=14)
        rax.set_ylim(0.5, 2)
        ax.set_xlim(5, 1000.)
        ax.set_yscale('log')
        rax.set_xlim(5, 1000.)
        hep.cms.label("Preliminary", year=IOV, data = True, loc=0, ax=ax, fontsize=18);
        if groomed:
            ax.set_xlabel(r'$m_{SD, RECO} (GeV)$' )
        if norm:
            print("Check that sum of values ", np.sum(hist.values())," is same as integrate ", hist.integrate("mgen").value)
            hist = hist*1.0/hist.integrate("mgen").value
            oVals_sum = np.sum(oVals)
            oHistVals = oVals*1.0/oVals_sum
            oHistErr = np.sqrt(oVals)/oVals_sum
            #inputErr = inputErr*1.0/oVals_sum
            print("oVals after norm ", oVals, " by value ", np.sum(oVals))
        ratio = np.divide(hist.values(),oHistVals,
                      out=np.empty(np.array(hist.project("mgen").values()).shape).fill(1),
                      where= oHistVals!=0,)
        # ratio_staterr_up = np.divide(mcvals.values()+stat_unc_up+syst_unc_up,datavals.values(),                                                             
        #               out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),                                                                           
        #               where=datavals.values()!= 0,)                                                                                                         
        # ratio_staterr_down = np.divide(mcvals.values()-stat_unc_down-syst_unc_down,datavals.values(),                                                       
        #               out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),                                                                           
        #               where=datavals.values()!= 0,)         
        ratio_toterr_up = np.divide(oHistVals+oHistErr,oHistVals,
                      out=np.empty(np.array(oVals.shape)).fill(np.nan),
                      where=oHistVals!= 0,)
        ratio_toterr_down = np.divide(oHistVals-oHistErr,oHistVals,
                      out=np.empty(np.array(oVals.shape)).fill(np.nan),
                      where=oHistVals!= 0,)
        ratio_err = np.abs(np.divide(oHistErr,oHistVals,
                      out=np.empty(np.array(oVals.shape)).fill(np.nan),
                      where=oHistVals!= 0,))
        rax.stairs(values=ratio_toterr_up, edges = medges, baseline= ratio_toterr_down,
                fill=True,
                **tot_error_opts,
            )
        # rax.stairs(values=ratio_statterr_up, edges = medges, baseline= ratio_statterr_down,                                                                 
        #         fill=True,                                                                                                                                  
        #         **stat_error_opts,                                                                                                                          
        #     )                                                                                                                                               
        ax.stairs(values=(oHistVals+oHistErr)/widths, edges = medges, baseline= (oHistVals-oHistErr)/widths,
                fill=True,
                **tot_error_opts,
            )
        # ax.stairs(values=(mcvals.values()+stat_unc_up)/widths, edges = edges, baseline= (mcvals.values()-stat_unc_down)/widths,                             
        #         fill=True,                                                                                                                                  
        #         **stat_error_opts,                         
        #     )                                                                                                                                               
        hep.histplot(oHistVals, medges, stack=False, histtype='errorbar', yerr = abs(oHistErr),
                 ax=ax, marker =["."], color = 'Black', linewidth=1, binwnorm=True,
                 label="Unfolded "+channel+" Data")
        hep.histplot(hist, stack=False, histtype='step',
                 ax=ax, linestyle ='--', color = 'Black', linewidth=1, binwnorm=True,
                     label="MC Truth "+channel)
        hep.histplot(ratio, medges, histtype='step', ax=rax, linestyle ="--", color = 'black', linewidth=1)
        hep.histplot(np.ones_like(ratio), medges, histtype='errorbar',ax=rax,marker=['.'], color = 'black', linewidth=1, yerr = ratio_err)
        leg = ax.legend(loc='best', fontsize=14, labelspacing=0.25)
        leg.set_visible(True)
        if norm:
            fig.savefig(os_path+"UnfoldOutputRatioPt{}_{}_normed.png".format(ptedges[j], ptedges[j+1]), bbox_inches="tight") 
        else:
            fig.savefig(os_path+"UnfoldOutputRatioPt{}_{}.png".format(ptedges[j], ptedges[j+1]), bbox_inches="tight") 
            
            
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
    response_matrix_u, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'][{'syst':"nominal"}].project("ptreco", "mreco", "ptgen", "mgen").to_numpy()
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    n_ptgen_bin = len(ptgen_edges)-1
    n_mgen_bin= len(mgen_edges)-1
    #### make TUnfold binning axes
    detectorBinning = ROOT.TUnfoldBinning("detector")
    recoBinning = detectorBinning.AddBinning("reco")
    recoBinning.AddAxis("m_{RECO}", n_mreco_bin, mreco_edges, False, False) # mreco
    recoBinning.AddAxis("pt_{RECO}", n_ptreco_bin, ptreco_edges, False, False) # ptreco
    generatorBinning = ROOT.TUnfoldBinning("generator")
    genBinning = generatorBinning.AddBinning("gen")
    genBinning.AddAxis("m_{GEN}", n_mgen_bin, mgen_edges, False, False) #mgen
    genBinning.AddAxis("pt_{GEN}", n_ptgen_bin, ptgen_edges, False, False) #ptgen 
    return detectorBinning, generatorBinning, mreco_edges, ptreco_edges, mgen_edges, ptgen_edges

def fillData(result, detectorBinning, mreco_edges, ptreco_edges, new=True):
    #### create data histograms
    if new:
        reco_str = "ptreco_mreco"
        gen_str = "ptgen_mgen"
    else:
        reco_str = "jet_pt_mass_reco"
        gen_str = "jet_pt_mass_reco"
    data_pt_m_u =  result[reco_str+"_u"][{'syst':"nominal"}].project('ptreco', 'mreco').values()
    data_pt_m_g = result[reco_str+"_g"][{'syst':"nominal"}].project('ptreco', 'mreco').values()
    recoBinning = detectorBinning.FindNode("reco")
    DataReco_u=recoBinning.CreateHistogram("histDataReco Ungroomed")  #data values in reco binnning --> input to unfolding
    DataReco_g=recoBinning.CreateHistogram("histDataReco Groomed")  #data values in reco binnning --> input to unfolding
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    for i in range(n_ptreco_bin):
        for j in range(n_mreco_bin):
    	    #### fill data hist
            recoBin=recoBinning.GetGlobalBinNumber(mreco_edges[j],ptreco_edges[i])
            data_weight_u=data_pt_m_u[i][j]
            DataReco_u.SetBinContent(recoBin, data_weight_u)
            DataReco_u.SetBinError(recoBin, data_weight_u**0.5)
            data_weight_g=data_pt_m_g[i][j]
            DataReco_g.SetBinContent(recoBin, data_weight_g)
            DataReco_g.SetBinError(recoBin, data_weight_g**0.5)
            #print("Data weight ", data_weight_u, " or ", ptreco_mreco_uu.GetBinContent(i, j) , " for matrix reco bin ", recoBin)
    return DataReco_u, DataReco_g

def getHists_other(result, syst, detectorBinning, generatorBinning):
    #### get hists from dictionary
    resp_matrix_u = result['response_matrix_u'][{'syst':syst}]
    resp_matrix_g = result['response_matrix_g'][{'syst':syst}]
    fakes = result['fakes'][{'syst':syst}]
    fakes_g = result['fakes_g'][{'syst':syst}]
    misses = result['misses'][{'syst':syst}]
    misses_g = result['mises_g'][{'syst':syst}]
    #### set up root tunfold histograms
    h = bins.detDist.CreateHistogram("h")
    hreco = bins.detDist.CreateHistogram("hreco")
    htrue = bins.genDist.CreateHistogram("htrue")
    
    hists = {}
    
    M_np = resp_matrix_u.project('ptgen','mgen','ptreco','mreco').values()
    M_np = M_np.reshape(M_np.shape[0]*M_np.shape[1], M_np.shape[2]*M_np.shape[3])
    M_np_error =resp_matrix_u.project('ptgen','mgen','ptreco','mreco').variances()
    M_np_error = M_np.reshape(M_np_error.shape[0]*M_np_error.shape[1], M_np_error.shape[2]*M_np_error.shape[3])**0.5
    
    miss_values = misses.project('ptgen', 'mgen').values().reshape(M_np.shape[0])
    fake_values = fakes.project('ptreco', 'mreco').values().reshape(M_np.shape[1])
    
    fake_hist = h.Clone('fakes')
    fake_hist.Reset()
    hist['MCGenRec_u']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Ungroomed "+syst)
    hist['MCGenRec_g']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Groomed "+syst)
def getHists(result, syst, detectorBinning, generatorBinning, new=True, doMiss = False):
    if new:
        reco_str = "ptreco_mreco"
        gen_str = "ptgen_mgen"
    else:
        reco_str = "jet_pt_mass_reco"
        gen_str = "jet_pt_mass_gen"
    hist = {}
    ptgen_mgen_u=result[gen_str+"_u"][{'syst':syst}].project('ptgen', 'mgen').values()
    ptgen_mgen_g=result[gen_str+"_g"][{'syst':syst}].project('ptgen', 'mgen').values()
    genErr_u=np.sqrt(result[gen_str+"_u"][{'syst':syst}].project('ptgen', 'mgen').variances())
    genErr_g=np.sqrt(result[gen_str+"_g"][{'syst':syst}].project('ptgen', 'mgen').variances())
    #### in future datasets input for fakes and misses depends on syst as well
    if syst in [ax for ax in result['fakes'].project("syst").axes[0]]:
        fakes_ptreco_mreco_u = result["fakes"][{'syst':syst}].project('ptreco', 'mreco').values()
        fakeErr = np.sqrt(result["fakes"][{'syst':syst}].project('ptreco', 'mreco').variances())
        fakes_ptreco_mreco_g = result["fakes_g"][{'syst':syst}].project('ptreco', 'mreco').values()
        fakeErr_g = np.sqrt(result["fakes_g"][{'syst':syst}].project('ptreco', 'mreco').variances())
    else:
        fakes_ptreco_mreco_u = np.zeros_like(result["ptreco_mreco_u"][{'syst':syst}].project('ptreco', 'mreco').values())
        fakeErr = np.zeros_like(np.sqrt(result["ptreco_mreco_u"][{'syst':syst}].project('ptreco', 'mreco').variances()))
        fakes_ptreco_mreco_g = np.zeros_like(result["ptreco_mreco_g"][{'syst':syst}].project('ptreco', 'mreco').values())
        fakeErr_g = np.zeros_like(np.sqrt(result["ptreco_mreco_g"][{'syst':syst}].project('ptreco', 'mreco').variances()))
    if syst in [ax for ax in result['misses'].project("syst").axes[0]]:
        misses_ptgen_mgen_u = result["misses"][{'syst':syst}].project('ptgen', 'mgen').values()
        missErr_u = np.sqrt(result["misses"][{'syst':syst}].project('ptgen', 'mgen').variances())
        misses_ptgen_mgen_g = result["misses_g"][{'syst':syst}].project('ptgen', 'mgen').values()
        missErr_g = np.sqrt(result["misses_g"][{'syst':syst}].project('ptgen', 'mgen').variances())
    else:
        misses_ptgen_mgen_u = np.zeros_like(result["ptgen_mgen_u"][{'syst':syst}].project('ptgen', 'mgen').values())
        missErr_u = np.zeros_like(np.sqrt(result["ptgen_mgen_u"][{'syst':syst}].project('ptgen', 'mgen').variances()))
        misses_ptgen_mgen_g = np.zeros_like(result["ptgen_mgen_g"][{'syst':syst}].project('ptgen', 'mgen').values())
        missErr_g = np.zeros_like(np.sqrt(result["ptgen_mgen_g"][{'syst':syst}].project('ptgen', 'mgen').variances()))
    ptreco_mreco_u  = result[reco_str+"_u"][{'syst':syst}].project('ptreco', 'mreco').values()
    ptreco_mreco_g = result[reco_str+"_g"][{'syst':syst}].project('ptreco', 'mreco').values()
    recoErr_u=np.sqrt(result[reco_str+"_u"][{'syst':syst}].project('ptreco', 'mreco').variances())
    recoErr_g=np.sqrt(result[reco_str+"_g"][{'syst':syst}].project('ptreco', 'mreco').variances())
    response_matrix_u, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").to_numpy()
    response_matrix_g = result['response_matrix_g'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").values()
    respErr_u = np.sqrt(result['response_matrix_u'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").variances())
    respErr_g = np.sqrt(result['response_matrix_g'][{'syst':syst}].project("ptreco", "mreco", "ptgen", "mgen").variances())
    n_ptreco_bin = len(ptreco_edges)-1
    n_mreco_bin= len(mreco_edges)-1
    n_ptgen_bin = len(ptgen_edges)-1
    n_mgen_bin= len(mgen_edges)-1
    hist['MCGenRec_u']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Ungroomed "+syst)
    hist['MCGenRec_g']=ROOT.TUnfoldBinning.CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Groomed "+syst)
    #### get gen, reco, and fake binning
    recoBinning = detectorBinning.FindNode("reco")
    genBinning = generatorBinning.FindNode("gen")
    # fakeBinning = generatorBinning.FindNode("fakesBin")
    #### create MC hists for comparison
    hist['MCReco_u']=recoBinning.CreateHistogram("histMCReco Ungroomed "+syst) #gen values in reco binning --> htruef in Sal's example
    hist['MCTruth_u']=genBinning.CreateHistogram("histMCTruth Ungroomed "+syst)  #gen values in gen binning    
    hist['MCReco_g']=recoBinning.CreateHistogram("histMCReco Groomed "+syst) #gen values in reco binning --> htruef in Sal's example
    hist['MCTruth_g']=genBinning.CreateHistogram("histMCTruth Groomed "+syst)  #gen values in gen binning
    hist['Misses_u']=genBinning.CreateHistogram("Misses Ungroomed "+syst)  #gen values in gen binning
    hist['Misses_g']=genBinning.CreateHistogram("Misses Groomed "+syst)  #gen values in gen binning
    hist['Fakes_u']=genBinning.CreateHistogram("Fakes Ungroomed "+syst)  #gen values in gen binning
    hist['Fakes_g']=genBinning.CreateHistogram("Fakes Groomed "+syst)  #gen values in gen binning
    # hist['MCTruth_RecoBinned_u']=recoBinning.CreateHistogram("histMCTruth Reco Binned, Ungroomed") #gen values in reco binning --> htruef in Sal's example 
    # hist['MCTruth_RecoBinned_g']=recoBinning.CreateHistogram("histMCTruth Reco Binned, Groomed") #gen values in reco binning --> htruef in Sal's example  
    #### Loop through reco and gen bins of MC input and fill hist of migrations
    #### reco loop: i is ptreco, j is mreco
    for i in range(n_ptreco_bin):
        for j in range(n_mreco_bin):
            glob_recobin=(i)*(n_mreco_bin)+j
            recoBin=recoBinning.GetGlobalBinNumber(mreco_edges[j],ptreco_edges[i])
            #### only fill fakes in fake genBin
            fake_weight = fakes_ptreco_mreco_u[i][j]
            fake_weight_g = fakes_ptreco_mreco_g[i][j]
            #print("Fake weight ", fake_weight," for i == ",i, " and j == ",j)
            # fakeBin=fakeBinning.GetStartBin()
            # hist['MCGenRec_u'].SetBinContent(fakeBin,recoBin,fake_weight)
            # hist['MCGenRec_u'].SetBinError(fakeBin,recoBin,fakesErr[i][j])
            #### fill MC reco hist for comparison of inputs
            reco_weight_u=ptreco_mreco_u[i][j]
            hist['MCReco_u'].SetBinContent(recoBin, reco_weight_u)
            hist['MCReco_u'].SetBinError(recoBin, recoErr_u[i][j])
            hist['Fakes_u'].SetBinContent(recoBin, fake_weight)
            hist['Fakes_u'].SetBinError(recoBin, fakeErr[i][j])
            #print("Reco weight ungroomed ", reco_weight_u ," for matrix reco bin " , recoBin)
            reco_weight_g=ptreco_mreco_g[i][j]
            hist['MCReco_g'].SetBinContent(recoBin, reco_weight_g)
            hist['MCReco_g'].SetBinError(recoBin, recoErr_g[i][j])
            hist['Fakes_g'].SetBinContent(recoBin, fake_weight_g)
            hist['Fakes_g'].SetBinError(recoBin, fakeErr_g[i][j])
            #print("Reco weight groomed " , reco_weight_g , " for matrix reco bin " , recoBin)            
	    #### gen loop: k is ptgen, l is mgen
            for k in range(n_ptgen_bin):
                for l in range(n_mgen_bin):
                    glob_genbin=(k)*(n_mgen_bin)+l
                    genBin=genBinning.GetGlobalBinNumber(mgen_edges[l],ptgen_edges[k])
     	            #### fill MC truth for closure test
	            #### ONLY FILL ONCE INSIDE i,j LOOP
                    if(i==1 and j==1):
                        #### fill Gen weights
                        # print("Bin l = ", l, " has mgen edge ", mgen_edges[l], " and bin k = ", k, " has ptgen edge ", ptgen_edges[k])
                        truth_weight_u = ptgen_mgen_u[k][l]
                        hist['MCTruth_u'].SetBinContent(genBin, truth_weight_u)
                        hist['MCTruth_u'].SetBinError(genBin, genErr_u[k][l])
                        #print("Truth weight ", truth_weight_u, " for matrix gen bin ", genBin)
                        truth_weight_g = ptgen_mgen_g[k][l]
                        hist['MCTruth_g'].SetBinContent(genBin, truth_weight_g)
                        hist['MCTruth_g'].SetBinError(genBin, genErr_g[k][l])
                        miss_weight_u = misses_ptgen_mgen_u[k][l]
                        hist['Misses_u'].SetBinContent(genBin, miss_weight_u)
                        hist['Misses_u'].SetBinError(genBin, missErr_u[k][l])
                        #print("Truth weight ", truth_weight_u, " for matrix gen bin ", genBin)
                        miss_weight_g = misses_ptgen_mgen_g[k][l]
                        hist['Misses_g'].SetBinContent(genBin, miss_weight_g)
                        hist['Misses_g'].SetBinError(genBin, missErr_g[k][l])
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
                    resp_weight_u = response_matrix_u[i][j][k][l]
                    #resp_weight_u = response_matrix_u.flatten()[glob_bin]
                    # print("Response weight for index i ",  i, " j ", j, " k ", k, " l ", l, " in unflattened matrix ", response_matrix_u[i][j][k][l])
                    # print("and weight for global index ", glob_bin, " in flattened matrix ", response_matrix_u.flatten()[glob_bin])
                    hist['MCGenRec_u'].SetBinContent(genBin,recoBin,resp_weight_u)
                    hist['MCGenRec_u'].SetBinError(genBin,recoBin,respErr_u[i][j][k][l])
                    #### fill groomed resp. matrices
                    resp_weight_g = response_matrix_g[i][j][k][l]
                    hist['MCGenRec_g'].SetBinContent(genBin,recoBin,resp_weight_g)
                    hist['MCGenRec_g'].SetBinError(genBin,recoBin,respErr_g[i][j][k][l])
    # print("Response matrix")
    # hist['MCGenRec_u'].Print("base")
    hist['fakes_ptreco_mreco_u'] = fakes_ptreco_mreco_u
    hist['fakes_ptreco_mreco_g'] = fakes_ptreco_mreco_g
    hist['misses_ptgen_mgen_u'] = misses_ptgen_mgen_u
    hist['misses_ptgen_mgen_g'] = misses_ptgen_mgen_g
    hist['ptreco_mreco_u'] = ptreco_mreco_u
    hist['ptreco_mreco_g'] = ptreco_mreco_g
    hist['ptgen_mgen_u'] = ptgen_mgen_u
    hist['ptgen_mgen_g'] = ptgen_mgen_g
    hist['response_matrix_u'] = response_matrix_u
    hist['response_matrix_g'] = response_matrix_g
    del fakes_ptreco_mreco_u,fakes_ptreco_mreco_g, misses_ptgen_mgen_u,misses_ptgen_mgen_g, ptreco_mreco_u, ptreco_mreco_g, ptgen_mgen_u, ptgen_mgen_g, response_matrix_u, response_matrix_g
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
