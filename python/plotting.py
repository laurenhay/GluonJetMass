# Plotting functions

import awkward as ak
import numpy as np
import pickle
import hist
import coffea
from python.plugins import *

# from hist import intervals
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl

def plotSyst(result, histname, axVar, label, logy=True, IOV = '', channel='', os_path=""):
    from cycler import cycler
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c', '#9F79EE']
    fill_opts = {
            'edgecolor': (0,0,0,0.3),
            'alpha': 0.8}
    error_opts = {
            'label': 'Stat. Unc.',
            'hatch': '///',
            'facecolor': 'none',
            'edgecolor': (0,0,0,.5),
            'linewidth': 0
        }
    data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 10.,
            'color': 'k',
            'elinewidth': 1,
        }
    fig_tot, (ax_tot, rax_tot) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
    if "pt" in axVar:
        rax_tot.set_xlim(0, 2000)
    ax_tot.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    rax_tot.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    availAxes = [ax.name for ax in result[histname].axes]
    availSysts = [ax for ax in result[histname].project("syst").axes[0]]
    print("Available axes: ", availAxes)
    print("Available systs ", availSysts)
    # fig.subplots_adjust(hspace=.1)
    mc = []
    rax_lim_dict = {}
    edges = [bin[0] for bin in result[histname].project(axVar).axes[0]] + [result[histname].project(axVar).axes[0][-1][1]]
    xlim = edges[-1]
    print("Plotting all systematics available")
    systematics = [syst for syst in availSysts if syst !="nominal"]
    for i, syst in enumerate(systematics):
        lines = ["-", "-", "-"]
        cols = ['green', 'red', 'black']
        #### Set up ratio plot
        if (syst in availSysts) and ("nominal" in availSysts) and (syst[-2:]=="Up"):
            fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
            ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
            ax.set_ylabel(r'Events', loc = 'top')
            if logy: 
                ax.set_yscale('log')
            syst = syst[:-2]
            ratio_D = np.ones_like(result[histname][{"syst":syst+"Up",}].project(axVar).values())
            ratio_U = np.ones_like(result[histname][{"syst":syst+"Up",}].project(axVar).values())
            #### Get mc hist
            # print("Up values: ", result[histname][{'syst':syst+"Up"}].project(axVar).values())
            if (syst+"Down") in availSysts:
                ratio_D = np.divide(result[histname][{'syst':syst+"Down"}].project(axVar).values(),result[histname][{"syst":"nominal"}].project(axVar).values(),
                                    out=np.empty(np.array(result[histname][{"syst":"nominal"}].project(axVar).values()).shape).fill(np.nan),
                                    where=result[histname][{"syst":"nominal"}].project(axVar).values()!= 0,)
                hep.histplot(ratio_D, edges, stack=True, histtype='step', ax=rax, density=False, linestyle ="-", color = 'red', linewidth=1)
                # print("Down values: ", result[histname][{'syst':syst+"Down"}].project(axVar).values())
                hep.histplot(ratio_D, edges, stack=True, histtype='step', ax=rax_tot, density=False, linestyle ="-", linewidth=1)
                hep.histplot(result[histname][{'syst':syst+"Down"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'red', linewidth=1,label=syst+"Down")
                hep.histplot(result[histname][{'syst':syst+"Down"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', linewidth=1,label=syst+"Down")
            #### Fill ratio plot
            ax.set_xlabel(None)
            # print("Nom values: ", result[histname][{'syst':"nominal"}].project(axVar).values())
            ratio_U = np.divide(
                result[histname][{'syst':syst+"Up"}].project(axVar).values(),
                result[histname][{"syst":"nominal"}].project(axVar).values(),
                out=np.empty(np.array(result[histname][{"syst":"nominal"}].project(axVar).values()).shape).fill(np.nan),
                where=result[histname][{"syst":"nominal"}].project(axVar).values()!= 0,)
            hep.histplot(result[histname][{'syst':syst+"Up"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'green', linewidth=1,label=syst+"Up")
            hep.histplot(result[histname][{'syst':syst+"Up"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', linewidth=1,label=syst+"Up")
            ax.autoscale(axis='x', tight=True)
            leg = ax.legend(loc='best', labelspacing=0.25)
            leg.set_visible(True)
            hep.histplot(ratio_U, edges, stack=False, histtype='step', ax=rax, linestyle ="-", color = 'green', linewidth=1)
            hep.histplot(ratio_U, edges, stack=False, histtype='step', ax=rax_tot, linestyle ="-", linewidth=1)
            #### plotting nominal values
            hep.histplot(np.ones_like(ratio_U), edges, stack=False, histtype='step',ax=rax, linestyle ="--", color = 'black', linewidth=1)
            hep.histplot(result[histname][{'syst':"nominal"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'black', linewidth=1,label="nominal")
            hep.histplot(result[histname][{'syst':"nominal"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', color = 'black', linewidth=1,label="nominal")
            rax.set_ylabel(r'$\frac{Up/Down}{Nominal}$', loc = 'center')
            rax.set_ylim(0.9,1.1)
            rax_tot.set_ylabel(r'$\frac{Up/Down}{Nominal}$', loc = 'center')
            # rax_tot.set_ylim(0.9,1.1)
            if ("eta" in axVar) | ("phi" in axVar):
                rax.set_xlim(-xlim, xlim)
            elif "pt" in axVar:
                rax.set_xlim(0, 2000)
            else:
                rax.set_xlim(0, xlim)
            cms = plt.text(0.04, 0.87, 'CMS $\it{Simulation}$',
                          fontsize=22,
                          fontfamily='sans',
                          fontweight='bold',
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          transform=ax.transAxes
                         )
            lumi = plt.text(1., 1., label,
                        fontsize=16,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes
                       )
            fig.savefig(os_path+"/UL{}{}_{}_{}_{}.png".format(IOV,channel, histname, axVar, syst), bbox_inches='tight') 
        elif syst in availSysts and "nominal" not in availSysts:
            fig, ax, = plt.subplots(nrows=1,ncols=1,figsize=(7,7))
            mc = [result[histname][{'syst':syst}].project(axVar),  result[histname][{'syst':syst}].project(axVar)]
            hep.histplot(mc, stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle =lines, color = cols, linewidth=1,label=syst)
            fig.savefig(os_path+"{}/UL{}_{}_{}_{}.png".format(IOV,IOV,channel, histname, axVar, syst), bbox_inches='tight')
        elif (syst in availSysts) and ("nominal" in availSysts) and (syst[-4:]=="Down"):
            print("Down -- plot both variations for up")
        else:
            print("Systematic not in desired hist")
        ax_tot.set_ylabel(r'Events', loc = 'top')
        # leg_tot = ax_tot.legend(loc='best', labelspacing=0.25)
        # add some labels
        fig_tot.savefig(os_path+"UL{}{}_{}_{}_allSyst.png".format(IOV,channel, histname, axVar), bbox_inches='tight')   
from hist.intervals import ratio_uncertainty
def plotDataMCwErrors(result_mc, result_data, hist_mc, hist_data, axVar, IOV, channel = "", norm = False, rax_lim=None, os_path=""):
    stat_unc_up = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
    stat_unc_down = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
    syst_unc_up, syst_unc_down = getTotSyst(result_mc, hist_mc, axis=axVar)
    print("Syst unc up vals: ", syst_unc_up)
    print("Syst unc down vals: ", syst_unc_down)
    #### following opts may be unnecessary
    tot_error_opts = {
            'label': 'Stat. + Syst. Unc.',
            'facecolor': 'orange',

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
    datahist = result_data[hist_data][{'syst':'nominal'}]
    mchist = result_mc[hist_mc][{'syst':'nominal'}]
    availAxes = [ax.name for ax in result_mc[hist_mc].axes]
    edges = [bin[0] for bin in result_mc[hist_mc].project(axVar).axes[0]] + [result_mc[hist_mc].project(axVar).axes[0][-1][1]]
    widths = result_mc[hist_mc].project(axVar).axes[0].widths
    print("widths", widths)
    xlim = edges[-1]
    fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
    ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
    ax.set_ylabel(r'$\frac{Events}{Bin Size} (GeV^{-1})$', loc = 'top')
    if "pt" in axVar or "m"==axVar[0]: 
        ax.set_yscale('log')
    if "_g" in hist_mc and "m"==axVar[0]:
        ax.set_xlabel(r'$m_{SD, RECO} (GeV)$' )
    else:
        ax.set_xlabel(r'$m_{RECO} (GeV)$' )
    ratio = np.ones_like(result_mc[hist_mc].project(axVar).values())
    #### Fill ratio plot
    rax.set_xlabel(None)
    if norm:
        mcvals = mchist.project(axVar)*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        datavals = datahist.project(axVar)
        stat_unc_up = stat_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        stat_unc_down = stat_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        syst_unc_up = syst_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        syst_unc_down = syst_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
    else:
        mcvals = mchist.project(axVar)
        datavals = datahist.project(axVar)
    ratio = np.divide(mcvals.values(),datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    #### Add MC error bars
    print("Total unc up: ", mcvals.values()+stat_unc_up+syst_unc_up)

    ax.stairs(values=(mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5)/widths, edges = edges, baseline= (mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5)/widths,
                fill=True,
                **tot_error_opts,
            )
    ax.stairs(values=(mcvals.values()+stat_unc_up)/widths, edges = edges, baseline= (mcvals.values()-stat_unc_down)/widths,
                fill=True,
                **stat_error_opts,
            )
    hep.histplot(datavals, stack=False, histtype='errorbar',
                 ax=ax, marker =["."], color = 'Black', linewidth=1, binwnorm=True,
                 label=channel + " Data")
    hep.histplot(mcvals, stack=False, histtype='step',
                 ax=ax, linestyle ='-', color = 'Black', linewidth=1, binwnorm=True,
                 label=channel + " MC")
    ax.autoscale(axis='x', tight=True)
    #### Want to stack uncertainties
        # print("Values in bins: ", mchist.project(axVar).values(), " errors of bins ", mchist.project(axVar).variances())
    if rax_lim != None:
        rax.set_ylim(rax_lim[0], rax_lim[1])
    leg = ax.legend(loc='best', labelspacing=0.25)
    leg.set_visible(True)
    #### Get ratio err values and plot
    ratio_totterr_up = np.divide((mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5),datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    ratio_totterr_down = np.divide(mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5,datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    ratio_statterr_up = np.divide(mcvals.values()+stat_unc_up,datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    ratio_statterr_down = np.divide(mcvals.values()-stat_unc_up,datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    rax.stairs(values=ratio_totterr_up, edges = edges, baseline= ratio_totterr_down,
                fill=True,
                **tot_error_opts,
            )
    rax.stairs(values=ratio_statterr_up, edges = edges, baseline= ratio_statterr_down,
                fill=True,
                **stat_error_opts,
            )
    hep.histplot(ratio, edges, histtype='step', ax=rax, linestyle ="-", color = 'black', linewidth=1)
    hep.histplot(np.ones_like(ratio), edges, histtype='step',ax=rax,linestyle ="--", color = 'black', linewidth=1)
    
    rax.set_ylabel(r'$\frac{MC}{Data}$', loc = 'center')
    if ("eta" in axVar) | ("phi" in axVar):
        rax.set_xlim(-xlim, xlim)
    elif "pt" in axVar:
        rax.set_xlim(0, 2000)
        # rax.set_ylim(0.0,1.0)
    else:
        rax.set_xlim(0, xlim)
    hep.cms.label("Preliminary", data = True, loc=0, ax=ax, fontsize=18);
    plt.show()
    if norm:
        fig.savefig(os_path+"ULwErrs{}{}_{}_{}_normed.png".format(IOV,channel, hist_mc, axVar), bbox_inches="tight") 
    else:
        fig.savefig(os_path+"ULwErrs{}{}_{}_{}.png".format(IOV,channel, hist_mc, axVar), bbox_inches="tight") 
        
def plotDataMC(result_mc, result_data, hist_mc, hist_data, axVar, result_herwig = None, IOV="", channel = "", rax_lim = [0.,2.0], norm = False):
    if result_herwig!=None:
        herwig=True
    else:
        herwig=False
    fill_opts = {
            'edgecolor': (0,0,0,0.3),
            'alpha': 0.8}
    error_opts = {
            'label': 'Stat. Unc.',
            'hatch': '///',
            'facecolor': 'none',
            'edgecolor': (0,0,0,.5),
            'linewidth': 0
        }
    data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 10.,
            'color': 'k',
            'elinewidth': 1,
        }
    datahist = result_data[hist_data][{'syst':'nominal'}]
    mchist = result_mc[hist_mc][{'syst':'nominal'}]
    if result_herwig!=None:
        herwighist = result_herwig[hist_mc][{'syst':'nominal'}]
    # if "m"==axVar[0]:
    #     datahist = result_data[hist_data].rebin(2)
    #     edges_data = [bin[0] for bin in result_data[hist_data].project(axVar).axes[0]] + [result_mc[hist_data].project(axVar).axes[0][-1][1]]
    #     print("Data mass bins after merging pt bins: ", edges_data)
    availAxes = [ax.name for ax in result_mc[hist_mc].axes]
    edges = [bin[0] for bin in result_mc[hist_mc].project(axVar).axes[0]] + [result_mc[hist_mc].project(axVar).axes[0][-1][1]]
    print("Edges dervied from mc hist: ", edges)
    xlim = edges[-1]
    fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
    ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
    ax.set_ylabel(r'$\frac{Events}{Bin Size} (GeV^{-1})$', loc = 'top')
    if "pt" in axVar or "m"==axVar[0]: 
        ax.set_yscale('log')
    if "_g" in hist_mc and "m"==axVar[0]:
        ax.set_xlabel(r'$m_{SD, RECO} (GeV)$' )
    ratio = np.ones_like(result_mc[hist_mc].project(axVar).values())
    #### Fill ratio plot
    rax.set_xlabel(None)
    if norm:
        mcvals = mchist.project(axVar)*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        datavals = datahist.project(axVar)
        # stat_unc_up = stat_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # stat_unc_down = stat_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # syst_unc_up = syst_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # syst_unc_down = syst_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        if herwig:
            herwigvals = herwighist.project(axVar)*datahist.project(axVar).integrate(axVar).value/herwighist.project(axVar).integrate(axVar).value
            
    else:
        mcvals = mchist.project(axVar)
        datavals = datahist.project(axVar)
        if herwig:
            herwigvals = herwighist.project(axVar)
    ratio = np.divide(mcvals.values(),datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=mcvals.values()!= 0,)
    if herwig:
        ratio_h = np.divide(herwigvals.values(),datavals.values(),
                      out=np.empty(np.array(herwigvals.values()).shape).fill(np.nan),
                      where=herwigvals.values()!= 0,)
    print("Integral of hist ", datahist.project(axVar).integrate(axVar).value)
    # normedHistVals = np.divide(datahist.project(axVar).values(),/datahist.project(axVar).integrate(axVar),
    #                   out=np.empty(np.array(mchist.project(axVar).values()).shape).fill(np.nan),
    #                   where=mchist.project(axVar).values()!= 0,)
    hep.histplot(datavals, stack=False, histtype='errorbar',
                 binwnorm=1, ax=ax, marker =["."], color = 'Black', linewidth=1,
                 label=channel + " Data")
    # print("Values in bins: ", datahist.project(axVar), " errors of bins ", datahist.project(axVar).variances(), )
    hep.histplot(mcvals, stack=False, histtype='step',
                 binwnorm=1, ax=ax, linestyle ='-', color = 'Blue', linewidth=1,
                 label=channel + " MC")
    if herwig:
        hep.histplot(herwigvals, stack=False, histtype='step',
                 binwnorm=1, ax=ax, linestyle ='-', color = 'Red', linewidth=1,
                 label=channel + " Herwig")
    # print("Values in bins: ", mchist.project(axVar).values(), " errors of bins ", mchist.project(axVar).variances())
    ax.autoscale(axis='x', tight=True)
    if rax_lim != None:
        rax.set_ylim(rax_lim[0], rax_lim[1])
    leg = ax.legend(loc='best', labelspacing=0.25)
    leg.set_visible(True)
    hep.histplot(ratio, edges, stack=False, histtype='step', ax=rax, density=False, linestyle ="-", color = 'blue', linewidth=1)
    if herwig:
        hep.histplot(ratio_h, edges, stack=False, histtype='step', ax=rax, density=False, linestyle ="-", color = 'red', linewidth=1)
    hep.histplot(np.ones_like(ratio), edges, stack=False, histtype='step',ax=rax, density=False, linestyle ="--", color = 'black', linewidth=1)
    rax.set_ylabel(r'$\frac{MC}{Data}$', loc = 'center')
    if ("eta" in axVar) | ("phi" in axVar):
        rax.set_xlim(-xlim, xlim)
    elif "pt" in axVar:
        rax.set_xlim(0, 2000)
        # rax.set_ylim(0.0,1.0)
    else:
        rax.set_xlim(0, xlim)
    cms = plt.text(0.25, 0.88, 'CMS $\it{Preliminary}$',
                  fontsize=22,
                  fontfamily='sans',
                  fontweight='bold',
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=ax.transAxes
                 )
    lumi = plt.text(1., 1., IOV,
                fontsize=16,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
               )
    if norm:
        fig.savefig("plots/comparison/{}/UL{}{}_{}_{}_normed.png".format(channel,IOV,channel, hist_mc, axVar)) 
    else:
        fig.savefig("plots/comparison/{}/UL{}{}_{}_{}.png".format(channel,IOV,channel, hist_mc, axVar)) 
    