import numpy as np
import awkward as ak
import correctionlib
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import copy
import os

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#based heavily on https://github.com/b2g-nano/TTbarAllHadUproot/blob/optimize/python/corrections.py and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py?ref_type=heads and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py?ref_type=heads

def ApplyVetoMap(IOV, jets, mapname='jetvetomap'):
    if IOV=="2016APV":
        IOV="2016"
    fname = "correctionFiles/jetvetomap/jetvetomaps_UL"+IOV+".json.gz"
    hname = {
        "2016"   : "Summer19UL16_V1",
        "2017"   : "Summer19UL17_V1",
        "2018"   : "Summer19UL18_V1"
    }
    print("Len of jets before veto", len(jets))
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    jetphi = np.where(jets.phi<3.141592, jets.phi, 3.141592)
    jetphi = np.where(jetphi>-3.141592, jetphi, -3.141592)
    vetoedjets = np.array(evaluator[hname[IOV]].evaluate(mapname, np.array(jets.eta), jetphi), dtype=bool)
    print("vetoed jets", vetoedjets)
    print("Sum of vetoed jets ", ak.sum(vetoedjets), " len of veto jets ", len(vetoedjets))
    print("Len of jets AFTER veto", len(jets[~vetoedjets]))
    return ~vetoedjets
    
def applyjmsSF(IOV, FatJet,  var = ''):
    jmsSF = {

        "2016APV":{"sf": 1.00, "sfup": 1.0794, "sfdown": 0.9906}, 

        "2016"   :{"sf": 1.00, "sfup": 1.0794, "sfdown": 0.9906}, 

        "2017"   :{"sf": 0.982, "sfup": 0.986, "sfdown": 0.978},

        "2018"   :{"sf": 0.999, "sfup": 1.001, "sfdown": 0.997}} 
    
    out = jmsSF[IOV]["sf"+var]
    

    FatJet = ak.with_field(FatJet, FatJet.mass * out, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * out, 'msoftdrop')
    return FatJet

def applyjmrSF(IOV, FatJet, var = ''):
    jmrSF = {

       #"2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016"   :{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 

        "2017"   :{"sf": 1.09, "sfup": 1.14, "sfdown": 1.04},

        "2018"   :{"sf": 1.108, "sfup": 1.142, "sfdown": 1.074}}     
    
    jmrvalnom = jmrSF[IOV]["sf"+var]
    
    recomass = FatJet.mass
    genmass = FatJet.matched_gen.mass


    # counts = ak.num(recomass)
    # recomass = ak.flatten(recomass)
    # genmass = ak.flatten(genmass)
    
    
    deltamass = (recomass-genmass)*(jmrvalnom-1.0)
    condition = ((recomass+deltamass)/recomass) > 0
    jmrnom = ak.where( recomass <= 0.0, 0 , ak.where( condition , (recomass+deltamass)/recomass, 0 ))


    FatJet = ak.with_field(FatJet, FatJet.mass * jmrnom, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * jmrnom, 'msoftdrop')

    return FatJet

def GetL1PreFiringWeight(events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L50
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    # Need to check if weights (up/dn) produced are comparable or greater than JEC weights --> if yes apply, if no take as a systematic uncertainty
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
    return L1PrefiringWeights

def HEMCleaning(IOV, JetCollection):
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (IOV == "2018"):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)
    JetCollection = ak.with_field(JetCollection, JetCollection.pt*isHEM, "pt" )
    return JetCollection
    
def HEMVeto(FatJets, runs):

    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    
    runid = (runs >= 319077)
    print(runid)
    # print("Fat jet phi ", FatJets.phi)
    # print("Fat jet phi length ", len(FatJets.phi))
    # print("Fat jet eta ", FatJets.eta)
    # print("Fat jet eta length ", len(FatJets.eta))
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -1.3) & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -2.5) & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    vetoHEMFatJets = ak.any((detector_region1 & jet_selection & runid) ^ (detector_region2 & jet_selection & runid), axis=1)
    print("Number of hem vetoed jets: ", ak.sum(vetoHEMFatJets))
    vetoHEM = ~(vetoHEMFatJets)
    
    return vetoHEM

def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
        
    fname = "correctionFiles/puWeights/{0}_UL/puWeights.json.gz".format(IOV)
    # print("PU SF filename: ", fname)
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "up")
    puDown = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "down")
    puNom = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "nominal")

    return puNom, puUp, puDown

def GetCorrectedSDMass(corr_jets, events, era, IOV, isData=False, uncertainties=None, useSubjets=True):
    print("Nevents with negative subjet id ", ak.sum(events.FatJet.subJetIdx1 < 0), " out of ", len(events))
    if useSubjets:
        print("Jets ", corr_jets.pt)
        print("SUbjet ids ", (corr_jets.subJetIdx1))
        print("SUbjets ", (events.SubJet.pt))
        print("SUbjet fields ", (events.SubJet.fields))
        SubJets=events.SubJet
        SubGenJetAK8 = events.SubGenJetAK8
        SubGenJetAK8['p4']= ak.with_name(events.SubGenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        SubJets["p4"] = ak.with_name(events.SubJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        corr_subjets = GetJetCorrections(SubJets, events, era, IOV, isData=isData, uncertainties = uncertainties, mode='AK4')
        del SubJets, SubGenJetAK8
    else:
        FatJets = events.FatJet
        GenJetAK8 = events.GenJetAK8
        GenJetAK8['p4']= ak.with_name(events.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
        FatJets["mass"] = FatJets.msoftdrop
        corr_subjets = GetJetCorrections(FatJets, events, era, IOV, isData=isData, uncertainties = uncertainties )
        corr_subjets =corr_subjets[corr_jets.subJetIdx1 > -1 ] 
        del FatJets, GenJetAK8
    corr_jets =corr_jets[(corr_jets.subJetIdx1 > -1)]
    fields = []
    fields.extend(field for field in corr_jets.fields if ("JES_" in field))
    fields.append("JER")
    if useSubjets:
        corr_jets["msoftdrop"] = (corr_subjets[corr_jets.subJetIdx1]+corr_subjets[corr_jets.subJetIdx2]).mass
    else:
        corr_jets["msoftdrop"] = corr_subjets.mass
    for field in fields:
        if useSubjets:
            corr_jets[field]["up"]["msoftdrop"] = (corr_subjets[field]["up"][corr_jets.subJetIdx1]+corr_subjets[field]["up"][corr_jets.subJetIdx2]).mass
            corr_jets[field]["down"]["msoftdrop"] = (corr_subjets[field]["down"][corr_jets.subJetIdx1]+corr_subjets[field]["down"][corr_jets.subJetIdx2]).mass
        else:
            corr_jets[field]["up"]["msoftdrop"] = corr_subjets[field]["up"].mass
            corr_jets[field]["down"]["msoftdrop"] = corr_subjets[field]["down"].mass
    del corr_subjets
    print("AK8 sdmass before corr ", events.FatJet.msoftdrop, " and after ", corr_jets.msoftdrop)
    return corr_jets
def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties = None, mode="AK8" ):
    AK_str = 'AK8PFPuppi'
    if mode=="AK4":
        AK_str = 'AK4PFPuppi'
    if uncertainties == None:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample", "RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    else:
        uncertainty_sources = uncertainties
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "Run2018A": "Summer19UL18_RunA_V5_DATA",
            "Run2018B": "Summer19UL18_RunB_V5_DATA",
            "Run2018C": "Summer19UL18_RunC_V5_DATA",
            "Run2018D": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "Run2017B": "Summer19UL17_RunB_V5_DATA",
            "Run2017C": "Summer19UL17_RunC_V5_DATA",
            "Run2017D": "Summer19UL17_RunD_V5_DATA",
            "Run2017E": "Summer19UL17_RunE_V5_DATA",
            "Run2017F": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV3_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
            "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        print(f"Error: Unknown year \"{IOV}\".")


    #print("extracting corrections from files for " + jec_tag)
    ext = extractor()
    if not isData:
    #For MC
        ext.add_weight_sets([
            '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_{1}.jec.txt'.format(jec_tag, AK_str),
            '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_{1}.jec.txt'.format(jec_tag, AK_str),
            '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_{1}.jec.txt'.format(jec_tag, AK_str),
            '* * '+'correctionFiles/JEC/{0}/{0}_UncertaintySources_{1}.junc.txt'.format(jec_tag, AK_str),
            '* * '+'correctionFiles/JEC/{0}/{0}_Uncertainty_{1}.junc.txt'.format(jec_tag, AK_str),
        ])
        #### Do AK8PUPPI jer files exist??
        if jer_tag:
            # print("JER tag: ", jer_tag)
            # print("File "+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)))
            # print("File "+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)))
            ext.add_weight_sets([
            '* * '+'correctionFiles/JER/{0}/{0}_PtResolution_{1}.jr.txt'.format(jer_tag, AK_str),
            '* * '+'correctionFiles/JER/{0}/{0}_SF_{1}.jersf.txt'.format(jer_tag, AK_str)])
            # print("JER SF added")
    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_{1}.jec.txt'.format(tag, AK_str),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_{1}.jec.txt'.format(tag, AK_str),
                '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_{1}.jec.txt'.format(tag, AK_str),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2L3Residual_{1}.jec.txt'.format(tag, AK_str),
                ])
                tags_done += [tag]
    ext.finalize()

    evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, AK_str),
            '{0}_L2Relative_{1}'.format(jec_tag, AK_str),
            '{0}_L3Absolute_{1}'.format(jec_tag, AK_str)]
        #### if jes in arguments add total uncertainty values for comparison and easy plotting
        if 'jes' in uncertainty_sources:
            jec_names.extend(['{0}_Uncertainty_{1}'.format(jec_tag, AK_str)])
            uncertainty_sources.remove('jes')
        jec_names.extend(['{0}_UncertaintySources_{1}_{2}'.format(jec_tag, AK_str, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, AK_str),
                              '{0}_SF_{1}'.format(jer_tag, AK_str)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, AK_str),
                '{0}_L3Absolute_{1}'.format(tag, AK_str),
                '{0}_L2Relative_{1}'.format(tag, AK_str),
                '{0}_L2L3Residual_{1}'.format(tag, AK_str),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)

    
    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    print("Rho value for jets ", events.fixedGridRhoFastjetAll)
    FatJets["pt"]= ak.values_astype(ak.fill_none(FatJets.pt, 0), np.float32)
    if mode=="AK4":
        SubGenJetAK8 = events.SubGenJetAK8
        SubGenJetAK8['p4']= ak.with_name(events.SubGenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.p4.nearest(SubGenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
        FatJets["pt"]= ak.values_astype(ak.fill_none(FatJets.pt, 0), np.float32)
        FatJets['area'] = ak.broadcast_arrays(0.503, FatJets.pt)[0]
    name_map = jec_stack.blank_name_map
    print("N events missing pt entry ", ak.sum(ak.num(FatJets.pt)<1))
    print("N events w/ pt entry ", ak.sum(ak.num(FatJets.pt)>0))
    print("Fatjet pt ", FatJets.pt)
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['Rho'] = 'rho'

    events_cache = events.caches[0]

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    print("pt and mass before correction ", FatJets['pt_raw'], ", ", FatJets['mass_raw'], " and after correction ", corrected_jets["pt"], ", ", corrected_jets["mass"])
    return corrected_jets
def GetLHEWeight(events):
    from parton import mkPDF
    #### make sure to `pip3 install --user parton` in singularity shell
    # os.environ['LHAPDF_DATA_PATH'] = '/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.1-7149a/x86_64-centos7-gcc8-opt/share/LHAPDF/'
    paths = ['/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/']
    # pdfset = PDFSet('PDF4LHC21_mc_pdfas', pdfdir=path[0])
    # print("PDF set obj ", pdfset)
    # print("PDF set info ", pdfset.info)
    # pdfmembers = []
    print("Length of events ", len(events))
    with suppress_stdout_stderr():
        pdf = mkPDF('PDF4LHC21_mc', pdfdir=paths[0])
    # print("PDFSet info ", pdf.pdfset.info)
    # print("PDF grids ", pdf.pdfgrids)
    
    #### get pdf values from events
    q2 = events.Generator.scalePDF
    x1 = events.Generator.x1
    x2 = events.Generator.x2
    id1 = events.Generator.id1
    id2 = events.Generator.id2
    xfxq1 = pdf.xfxQ(id1, x1, q2, grid=False)
    xfxq2 = pdf.xfxQ(id2, x2, q2, grid=False)
    # print("x*f(x) = ", xfxq1, xfxq2)
    # print("Length of xfxq1 ", len(xfxq1))
    # print("type of pdf weight ", type(xfxq1))
    pdfNom = xfxq1*xfxq2
    pdfUp = pdf.xfxQ(id1, x1, 2*q2, grid=False)*pdf.xfxQ(id1, x1, 2*q2, grid=False)
    pdfDown = pdf.xfxQ(id1, x1, 0.5*q2, grid=False)*pdf.xfxQ(id1, x1, 0.5*q2, grid=False)
    #### Dont know what alphas do
        #     muRUp = self.lha.evalAlphas(2*q)**4
        # muRDown = self.lha.evalAlphas(0.5*q)**4
    return pdfNom, pdfUp, pdfDown

def GetQ2Weights(events):
# https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py

    q2Nom = np.ones(len(events))
    q2Up = np.ones(len(events))
    q2Down = np.ones(len(events))
    if ("LHEScaleWeight" in events.fields):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1)==9):
            nom = events.LHEScaleWeight[:,4]
            scales = events.LHEScaleWeight[:,[0,1,3,5,7,8]]
            q2Up = ak.max(scales,axis=1)/nom
            q2Down = ak.min(scales,axis=1)/nom 
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1)==8):
            scales = events.LHEScaleWeight[:,[0,1,3,4,6,7]]
            q2Up = ak.max(scales,axis=1)
            q2Down = ak.min(scales,axis=1)

    return q2Nom, q2Up, q2Down

def GetPDFWeights(events):
    
    # hessian pdf weights https://arxiv.org/pdf/1510.03865v1.pdf
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/corrections.py#L60
    
    pdf_nom = np.ones(len(events))

    if "LHEPdfWeight" in events.fields:
        
        pdfUnc = ak.std(events.LHEPdfWeight,axis=1)/ak.mean(events.LHEPdfWeight,axis=1)
        pdfUnc = ak.fill_none(pdfUnc, 0.00)
        
        pdf_up = pdf_nom + pdfUnc
        pdf_down = pdf_nom - pdfUnc
        
        
#         arg = events.LHEPdfWeight[:, 1:-2] - np.ones((len(events), 100))
#         summed = ak.sum(np.square(arg), axis=1)
#         pdf_unc = np.sqrt((1. / 99.) * summed)
        
#         pdf_nom = np.ones(len(events))
#         pdf_up = pdf_nom + pdf_unc
#         pdf_down = np.ones(len(events))

    else:
        
        pdf_up = np.ones(len(events))
        pdf_down = np.ones(len(events))
        

    return pdf_nom, pdf_up, pdf_down