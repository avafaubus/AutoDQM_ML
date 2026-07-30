# intermediate step: hardcoded bad data and good data, labels it
import ROOT
from ROOT import TFile
import os
import uproot
import numpy as np
import pandas
import awkward
from collections import defaultdict
import re


EOS_path = "/eos/cms/store/data/Run2025F/Muon0/DQMIO/"

bad_files = {
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/701791AF-63CB-403F-BD19-69C65527A78A.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/75D652DA-FD32-4977-B8F4-F8931A041EA4.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/AD2648B7-92A5-4310-8643-921E3F05EA32.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/D1AA7C5E-09AC-49A2-96D2-0498BD7240C3.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/71D0327D-2774-4C0F-8E60-E2EC7231796A.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/86EC6762-2534-4DC4-809A-0C57E2522477.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/397/456/00000/CC2F5E9F-D500-4771-8C3B-29B3CFE8FC35.root",
}

good_files = {
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/128F2944-E4FF-4BC2-9836-FB3C03CF0AA7.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/6ADDC953-3D28-4865-8802-9095520CADC9.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/804C98A6-C379-407D-B42C-9D7933E79C46.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/985F9DC6-770B-4D04-9528-4E2C02E98D91.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/D5F533AC-19EE-40FA-B011-DA62264B22F3.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/23A04F0F-965C-4A4F-A1D9-851AEFCC058B.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/73E34F35-9DD9-48DE-95AE-0B6968015C7E.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/865CA6F5-9C54-42DB-94C6-F388D0F4EBED.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/A25FE6A2-2C55-45B2-8D88-D1AC0D478DFE.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/DB2AE756-9997-49E0-A224-2EAC5224C15A.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/5F12BFB4-14BD-4878-A0F0-415514FACDF1.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/77F85F9B-5DF8-45B6-9724-C643E3A991ED.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/8AC1142A-55F1-4CB4-B1E5-7032101B8EEB.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/AC474F6A-E98D-49D2-8A5A-BF22376D9F76.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/E92BD1FA-86E2-4E01-933E-6ED48B798D92.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/663A5D01-CE1D-43F0-9D5D-BE9EF4E136AF.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/7BC6DA10-9818-4871-AE6C-0D1732CCBA20.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/91489DA7-0870-4925-86B0-228202A89A08.root",
    "/eos/cms/store/data/Run2025F/Muon0/DQMIO/PromptReco-v1/000/396/988/00000/B8AD2216-8CD4-4BD6-BDE0-707FF403C06C.root",
}

wanted_hists = {
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalm2",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalm3",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalm4",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalp1",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalp2",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalp3",
    "CSC/CSCOfflineMonitor/recHits/hRHGlobalp4",
}

def LSdict(file):
    df = ROOT.RDataFrame("Indices", file)

    data = df.AsNumpy(columns=[
        "Run.Run",
        "Lumi.Lumi",
        "Type.Type",
        "FirstIndex",
        "LastIndex",
        ])

    run = data["Run.Run"]
    lumi = data["Lumi.Lumi"]
    typ = data["Type.Type"]
    first = data["FirstIndex"]
    last = data["LastIndex"]

    indices = {}

    for r, l, t, f, la in zip(run, lumi, typ, first, last):
        if int(t)!=6:
            continue
        key = (int(r),int(l))
        indices[key] = (int(f), int(la))
    return indices

def get_Hists(tree, indices, run, lumi):


    key = (int(run), int(lumi))

    if key not in indices:
        raise ValueError(f"No indices for run {run}, lumi {lumi}")

    first, last = indices[key]
    idx = first
    found = 0
    hist_data = {}

    while idx <= last and found < len(wanted_hists):
        tree.GetEntry(idx)
        fullname = str(tree.FullName)

        if fullname in wanted_hists:
            hist = tree.Value

            nx = hist.GetNbinsX()
            ny = hist.GetNbinsY()

            arr = np.empty((nx, ny), dtype=np.float32)
            for xbin in range(nx):
                for ybin in range(ny):
                    arr[xbin, ybin] = hist.GetBinContent(xbin + 1, ybin + 1)
            hist_data[fullname] = arr
            print("Found:", fullname)
            found +=1
        idx +=1
        if found == len(wanted_hists):
            break
    return hist_data

def process_all_files(output_dir):

    data = None
    for file in list(bad_files)+list(good_files):
        try:

            f = ROOT.TFile.Open(file)
            tree = f.Get("TH2Fs")
            indices = LSdict(file)

            for (run, lumi) in indices.keys():
                hist_data = get_Hists(tree, indices, run, lumi)

                if len(hist_data) != len(wanted_hists):
                    continue

                if data is None:
                    data = {}
                    for key in hist_data.keys():
                        data[key] = []

                    data["run_number"] = []
                    data["lumi"] = []
                    data["year"] = []
                    data["label"] = []

                for name, histogram in hist_data.items():
                    data[name].append(histogram)

                data["run_number"].append(run)
                data["lumi"].append(lumi)
                data["year"].append(2025)

                if file in good_files:
                    data["label"].append(0)

                elif file in bad_files:
                    data["label"].append(1)

        except Exception as e:
            print(e)
            continue
        finally:
            f.Close()
        

    if data is not None:
        output_file = os.path.join(output_dir, "Muon0_2025_labelled.parquet")
        array = awkward.Array(data)
        awkward.to_parquet(array, output_file, compression=None)
        print(f"Wrote {output_file}")











