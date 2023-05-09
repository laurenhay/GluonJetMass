#### based on the trigger emulation method spelled out here https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2015/154
import awkward as ak
import numpy as np
from coffea import processor, util
import hist
import pandas

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("year")
parser.add_argument("trigger")
parser.add_argument("testing")
parser.add_argument("inputFiles")
parser.add_argument("coffea")
parser.add_argument("dask")

class triggerProcessor(processor.ProcessorABC):
    def __init__(self, year = None, trigger = "", data = False):
        self.year = year
        self.trigger = trigger
        self.data = data
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        HLT_cat = hist.axis.StrCategory([], growth=True, name="HLT_cat",label="")
        if data:
            pt_bin = hist.axis.Regular(100, 0, 2400.,name="pt", label="Jet pT (GeV)")
        else: pt_bin = hist.axis.Regular(100, 0, 3200.,name="pt", label="Jet pT (GeV)")
        self._histos = {
            'hist_trigEff': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_trigEff_ptCut': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_trigRef': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt': hist.Hist(dataset_cat, pt_bin, storage="weight", name="Events"),
            'cutflow':      processor.defaultdict_accumulator(int),
            }
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        #### choose events that have at least one fatjet
        events = events[ak.num(events.FatJet) >= 1]
        print("Events metadata: ", events.metadata)
        
        #### Choose trigger objects that belong to jets (id == 1), after this selection remove events 
        #### that don't have corresponding jet trigger objects
        trigObj = events.TrigObj
        print("Trigger object id:", trigObj.id)
        trigObj = trigObj[trigObj.id == 1]
        events = events[ak.num(trigObj, axis = -1) != 0]
        trigObj = trigObj[ak.num(trigObj, axis = -1) != 0]
        year = self.year
        trigger = self.trigger
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        #### get HLT objects from events
        if year == 2016:
            # HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500']
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == 2017:
            # HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == 2018:
            # HLT_paths = ['AK8PFJet15', 'AK8PFJet25', 'AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        else:
            print("Must provide a year to run trigger studies")
        
        #### Loop over HLT paths and calculate efficiencies for each
        efficiencies = {}
        for i in np.arange(len(HLT_paths)):
            path = HLT_paths[i]
            print("index: ", i, " HLT path: ", path)
            if i > 0:
                print("Number of events w/ trigger ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                print("HLT ref path: ", HLT_paths[i-1], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i-1]]))
                #### choose jets belonging to ref trigger (i-1) and passing pt cut of trigger of trigger of interest (i)
                efficiencies[path] = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (trigObj.pt[:,0] > trigThresh[i])]
                #### trig eff
                out['hist_trigEff_ptCut'].fill(dataset = datastring + str(year), HLT_cat = path, pt = efficiencies[path])
                #### gives ratio of trigger paths (do not add to 1)
                out['hist_trigEff'].fill(dataset = datastring + str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (events.HLT[HLT_paths[i]])])
                out['hist_trigRef'].fill(dataset = datastring +  str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
                out['hist_pt'].fill(dataset = datastring + str(year), pt = events.FatJet[:,0].pt)

        print("Final efficiencies:", efficiencies)
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
#### APPLY PRESCALES NOT WORKING YET
#### applyPrescales should only be run on data
class applyPrescales(processor.ProcessorABC):
    def __init__(self, trigger, year, turnOnPts, data = True, byRun = True):
        self.data = data
        self.trigger = trigger
        self.year = year
        self.turnOnPt = turnOnPts
        self.byRun = byRun
        if data:
            pt_bin = hist.axis.Regular(1000, 0, 2400.,name="pt", label="Jet pT (GeV)")
        else: pt_bin = hist.axis.Regular(1000, 0, 3200.,name="pt", label="Jet pT (GeV)")
#         pt_bin = hist.axis.Variable(pt_bins,name="pt", label="Jet pT (GeV)", underflow = True, overflow = True)
#         pt_bins = turnOnPts[turnOnPts > 0]
#         pt_bin = hist.axis.Variable(pt_bins,name="pt", label="Jet pT (GeV)", underflow = True, overflow = True)
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        HLT_cat = hist.axis.StrCategory([], growth=True, name="HLT_cat",label="")
        self._histos = {
            'hist_pt': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'hist_pt_byHLTpath': hist.Hist(dataset_cat, HLT_cat, pt_bin, storage="weight", name="Events"),
            'cutflow':      processor.defaultdict_accumulator(int),
            }
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        trigger = self.trigger
        turnOnPt = self.turnOnPt
        byRun = self.byRun
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        if self.year == 2016:
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            if byRun == False:
                prescales = [136006.59, 50007.75, 13163.18, 1501.12, 349.82, 61.17, 20.49, 6.99, 1.00, 1.00]
            else:
                goldenJSON = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
        elif self.year == 2017:
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]  
            if byRun ==False:
                prescales = [86061.17,  36420.75,  9621.74, 1040.40, 189.54, 74.73, 29.49, 9.85, 3.97, 1.00, 1.00]
            else:
                goldenJSON = "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
        elif self.year == 2018:
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            if byRun == False:
                prescales = [318346231.66, 318346231.66, 248642.75, 74330.16, 11616.52, 1231.88, 286.14, 125.78, 32.66, 15.83,                     7.96, 1.00, 1.00]
            else:
                goldenJSON = "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
        ####require at least one jet in each event
        HLT_paths = [trigger + str(i) for i in trigThresh]
        events = events[ak.num(events.FatJet) >= 1]
        print("Event metadata: ", events.metadata)
        print("Event fields: ", events.fields)
        ####sort 
        if byRun == False:
            for i in np.arange(len(HLT_paths))[::-1]:
                path = HLT_paths[i]
    #             print("Index i: ", i, " for path: ", path)
                if path in events.HLT.fields:
                    pt0 = events.FatJet[:,0].pt
                    out['hist_pt'].fill(dataset = datastring, HLT_cat = path, pt = pt0[events.HLT[path]])
                    if i == (len(HLT_paths) - 1):
                        print('last index')
                        pt_cut = (pt0 >= turnOnPt[i]) & events.HLT[path]
    #                     print("# events passing pt cut ",  turnOnPt[i], ": ", len(pt0[pt_cut]))
                        out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut])
                    else:
                        pt_cut = (pt0 >= turnOnPt[i]) & (pt0 < turnOnPt[i+1]) & events.HLT[path]
    #                     print("# events passing pt cut ",  turnOnPt[i], ": ", len(pt0[pt_cut]))
                        out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut])
        else:
            cert_jsonData = pandas.read_json(goldenJSON, orient = 'index')
            #print("Cert ", year, " data: \n", cert_jsonData)                                                                     
            #allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style                 csv'
            ps_csvData = pandas.read_csv("allRunsAK8HLT_skimmed.csv")
            for i in np.arange(len(HLT_paths))[::-1]:
                path = HLT_paths[i]
                print("Index i: ", i, " for path: ", path)
                if path in events.HLT.fields:
                    pt0 = events.FatJet[:,0].pt
                    runs = events.run
                    fields = events.fields
                    print("Lumi block:", events[events.run == runs[0]].luminosityBlock, "for event ", runs[0])
                    print("Event run:", runs[0])
                    weights = np.ones_like(runs)
                    ### here we will use correctionlib to assign weights
                    print("any run:", np.any(ps_csvData['# run'].to_numpy()))
                    weights = np.where(weights == np.any(ps_csvData['# run'].to_numpy()), ps_csvData[ps_csvData['hltpath/prescval'].str.contains(path)][ps_csvData['# run'] == weights, 0])
                    ps_runs = ps_csvData[ps_csvData['hltpath/prescval'].str.contains(path)]['# run']
                    print("Event runs: ", len(runs))
                    print("prescale runs:", len(ps_runs))
                    prescales = ps_csvData['totprescval'][ps_csvData['hltpath/prescval'].str.contains(path)]
                    print("prescales:", prescales)
                    out['hist_pt'].fill(dataset = datastring, HLT_cat = path, pt = pt0[events.HLT[path] & runs])
                    if i == (len(HLT_paths) - 1):
                        print('last index')
                        pt_cut = (pt0 >= turnOnPt[i]) & events.HLT[path]
    #                     print("# events passing pt cut ",  turnOnPt[i], ": ", len(pt0[pt_cut]))
                        out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut], weight = weights)
                    else:
                        pt_cut = (pt0 >= turnOnPt[i]) & (pt0 < turnOnPt[i+1]) & events.HLT[path]
    #                     print("# events passing pt cut ",  turnOnPt[i], ": ", len(pt0[pt_cut]))
                        out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut], weight = weights)
        return out
    def postprocess(self, accumulator):
        return accumulator
        
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    in_year = 2016
    data_bool = False
    processor = triggerProcessor(year = in_year, trigger = 'AK8PFJet', data = data_bool)
    datastring = "JetHT" if processor.data == True else "QCDsim"
    filename = "datasets_UL_NANOAOD.json" if processor.data == True else "fileset_QCD.json"
    result = runCoffeaJob(processor, jsonFile = filename, casa = True, dask = True, testing = False, year = processor.year, data =     processor.data)
    util.save(result, 'coffeaOutput/triggerAssignment_{}_{}_{}_NewHist.coffea'.format(datastring, processor.year,                     processor.trigger))
    processor = applyPrescales(year = in_year, trigger = 'AK8PFJet', data = data_bool)
    dataname = "JetHT" if processor.data == True else "QCDsim"
    filename = "datasets_UL_NANOAOD.json" if processor.data == True else "fileset_QCD.json"
    result = runCoffeaJob(processor, jsonFile = filename, casa = True, dask = False, testing = True, year = processor.year, data =     processor.data)
    util.save(result, 'coffeaOutput/applyPrescales_{}_{}_{}_test_NewHist.coffea'.format(datastring, processor.year,                   processor.trigger))
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()