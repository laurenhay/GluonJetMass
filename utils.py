from coffea.lumi_tools import LumiMask


def getLumiMask(year):

    files = { '2016APV': "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2016': "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2017': "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
              '2018': "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
            }

    mask = LumiMask(files[year])

    return mask