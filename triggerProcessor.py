#### based on the trigger emulation method spelled out here https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2015/154
### Define to accumulate histograms
import awkward as ak
import numpy as np
from coffea import processor, hist, util

class triggerProcessor(processor.ProcessorABC):
    def __init__(self, year = None, trigger = "", data = False):
        self.year = year
        self.trigger = trigger
        self.data = data
        dataset_cat = hist.Cat("dataset", "Dataset")         
        HLT_cat = hist.Cat("HLT_cat", "")
        if data:
            pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 2400)
        else: pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 3200)
        self._histos = processor.dict_accumulator({
            'hist_trigEff': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_trigEff_ptCut': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_trigRef': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_pt': hist.Hist("Events", dataset_cat, pt_bin),
            'cutflow':      processor.defaultdict_accumulator(int),
            })
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self.accumulator.identity()
        #### choose events that have at least one fatjet
        events = events[ak.num(events.FatJet) >= 1]
        print("Events metadata: ", events.metadata)
        
        #### Choose trigger objects that belong to jets (id == 1), after this selection remove events 
        #### that don't have corresponding jet trigger objects
        trigObj = events.TrigObj
        print(trigObj.id)
        trigObj = trigObj[trigObj.id == 1]
        # deltaR  = events.FatJet[:, 0].delta_r(trigObj)
        # trigObj = trigObj[deltaR < 0.15]
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
            print("index: ", i, " path: ", path)
            if i > 0:
                print("Number of events w/ trigger ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                print("HLT path: ", path, " HLT values: ", events.HLT[HLT_paths[i]], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                print("HLT path: ", HLT_paths[i-1], " HLT values: ", events.HLT[HLT_paths[i-1]], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i-1]]))
                
                efficiencies[path] = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (trigObj.pt[:,0] > trigThresh[i])]
                
                out['hist_trigEff_ptCut'].fill(dataset = datastring + str(year), HLT_cat = path, pt = efficiencies[path])
                out['hist_trigEff'].fill(dataset = datastring + str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (events.HLT[HLT_paths[i]])])
                out['hist_trigRef'].fill(dataset = datastring +  str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
                out['hist_pt'].fill(dataset = datastring + str(year), pt = events.FatJet[:,0].pt)
                
       
                ####if using new hist format
                #hist_trigEff.fill(HLT_cat = path, jet_pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i]])])
                #hist_trigRef.fill(HLT_cat = path, jet_pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
        print("Final efficiencies:", efficiencies)
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
#### APPLY PRESCALES NOT WORKING YET
#### applyPrescales should only be run on data
class applyPrescales(processor.ProcessorABC):
    def __init__(self, trigger, year, data = True):
        self.data = data
        self.trigger = trigger
        self.year = year
        dataset_cat = hist.Cat("dataset", "Dataset")         
        HLT_cat = hist.Cat("HLT_cat", "")
        pt_bin = hist.Bin("pt", "Jet pT (GeV)", 500, 0, 2400)
        self._histos = processor.dict_accumulator({
            'hist_pt': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_pt_byHLTpath': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'cutflow':      processor.defaultdict_accumulator(int),
            })
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self.accumulator.identity()
        trigger = self.trigger
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        if self.year == 2016:
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            HLT_paths = [trigger + str(i) for i in trigThresh]
            turnOnPt = [0., 128., 196., 262., 296., 364., 433., 524., 583., 642.]
            prescales = [136006.59, 50007.75, 13163.18, 1501.12, 349.82, 61.17, 20.49, 6.99, 1.00, 1.00]
        elif self.year == 2017:
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
            turnOnPt = [0., 89.,  160., 254., 309., 381., 454., 546., 608., 669., 731]
            prescales = [86061.17,  36420.75,  9621.74, 1040.40, 189.54, 74.73, 29.49, 9.85, 3.97, 1.00, 1.00]
        elif self.year == 2018:
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
            prescales = [318346231.66, 318346231.66, 248642.75, 74330.16, 11616.52, 1231.88, 286.14, 125.78, 32.66, 15.83, 7.96,                1.00, 1.00]
            turnOnPt = [0., 0., 0., 0., 164., 252., 305., 379., 451., 544., 609., 668., 727.]
        ####require at least one jet in each event
        events = events[ak.num(events.FatJet) >= 1]
        ####sort 
        for i in np.arange(len(HLT_paths))[::-1]:
            path = HLT_paths[i]
            print("Index i: ", i, " for path: ", path)
            if path in events.HLT.fields:
                pt0 = events.FatJet[:,0].pt
                out['hist_pt'].fill(dataset = datastring, HLT_cat = path, pt = pt0[events.HLT[path]])
                if i == (len(HLT_paths) - 1):
                    print('last index')
                    pt_cut = (pt0 >= turnOnPt[i]) & events.HLT[path]
                    out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut])
                else:
                    pt_cut = (pt0 >= turnOnPt[i]) & (pt0 < turnOnPt[i+1]) & events.HLT[path]
                    out['hist_pt_byHLTpath'].fill(dataset = datastring, HLT_cat = path, pt = pt0[pt_cut])
        return out
    def postprocess(self, accumulator):
        return accumulator

print("trying to print year of processor")
print(applyPrescales(year = 2016, trigger = 'AK8PFJet', data = True).year)
        
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    #result = runCoffeaJob(triggerProcessor(year = 2016, trigger = 'AK8PFJet', data = True), jsonFile = "datasets_UL_NANOAOD.json", casa = True, dask = True, testing = False, year = 2016, data = True)
    result = runCoffeaJob(triggerProcessor(year = 2017, trigger = 'AK8PFJet', data = False), jsonFile = "fileset_QCD.json", casa = True, dask = True, testing = False, year = 2017)
    util.save(result, 'coffeaOutput/triggerAssignment_QCDsim_2017_test.coffea')
    result = runCoffeaJob(applyPrescales(year = 2017, trigger = 'AK8PFJet', data = True), jsonFile = "datasets_UL_NANOAOD.json", casa = True, dask = True, testing = False, year = 2017, data = True)
    util.save(result, 'coffeaOutput/applyPrescales_2017_all.coffea')
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()