### This file houses plugins
import pandas as pd
import time
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

#reads in files and adds redirector, can specify year, default is all years
def handleData(jsonFile, redirector, year = '', testing = True, data = False):
    qualifier = str(year)
    if data == True:
        inputs = 'JetHT_data'
    else: inputs = 'QCD_sim'
    df = pd.read_json(jsonFile) 
    dict = {}
    for key in df[inputs].keys():
        if qualifier in key:
            if testing:
                dict[key] = [redirector +  df[inputs][key][0]]
            else:
                dict[key] = [redirector + df[inputs][key][i] for i in range(len(df[inputs][key]))]
    return dict

#initiate dask client and run coffea job
from dask.distributed import Client

def runCoffeaJob(processor_inst, jsonFile, lpc = False, casa = False, testing = False, year = None, data = False):
    #default is to run locally
    tstart = time.time()
    executor = processor.futures_executor
    redirector = 'root://cmsxrootd.fnal.gov/'
    casa_redirector = 'root://xcache/'
    exe_args = {"schema": NanoAODSchema, 'skipbadfiles': True,}
    samples = handleData(jsonFile, redirector, year = year, testing = testing, data = data)
    client = None
    if casa:
        print("Running on coffea casa")
        from coffea_casa import CoffeaCasaCluster
        from dask.distributed.diagnostics.plugin import UploadDirectory #do i need this?
        client = Client("tls://lauren-2emeryl-2ehay-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
        samples = handleData(jsonFile, casa_redirector, year = year, testing = testing, data = data)
        exe_args = {
            "client": client,
            'skipbadfiles':True,
            "schema": NanoAODSchema,
            "align_clusters": True,
        }
        executor = processor.dask_executor
    elif lpc:
        print("Running on LPC Condor")
        from lpcjobqueue import LPCCondorCluster
        cluster = LPCCondorCluster(shared_temp_directory="/tmp")
        #### minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        exe_args = {
            "client": client,
            'skipbadfiles':True,
            "savemetrics": True,
            "schema": NanoAODSchema,
            "align_clusters": True,
        }
#         print("Waiting for at least one worker...")
        client.wait_for_workers(1)
        executor = processor.dask_executor
    else:
        print("Running locally")
    result = processor.run_uproot_job(samples,
                                      "Events",
                                      processor_instance = processor_inst,
                                      executor = executor,
                                      executor_args = exe_args,
                                      chunksize = 1000,
                                      maxchunks = 1,
                                     )
    elapsed = time.time() - tstart
    print(result)
    print("Time taken to run over samples ", elapsed)
    return result