import os
from os import listdir
from os.path import join, dirname, realpath


from pymagicc.io import MAGICCData
from openscm.highlevel import OpenSCMDataFrame


here = dirname(realpath(__file__))

INPUT_PATH = join(here, "..", "data", "rcps")

models = {
    "RCP26": "IMAGE",
    "RCP45": "MiniCAM",
    "RCP60": "AIM",
    "RCP85": "MESSAGE",
    "historical": "unspecified"
}

for file in listdir(INPUT_PATH):
    if file.endswith(".DAT"):
        inname = join(INPUT_PATH, file)
        df = MAGICCData(inname).data.copy()

        scenario = file.split("_")[0]
        scenario = "historical" if scenario == "20THCENTURY" else scenario

        df["scenario"] = scenario
        df["model"] = models[scenario]

        outname = join(
            INPUT_PATH,
            file.lower().replace(".dat", ".csv")
        )
        print("Saving {} as {}".format(inname, outname))

        OpenSCMDataFrame(df).to_csv(outname)
