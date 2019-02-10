import os
from os import listdir
from os.path import join, dirname, realpath


from pymagicc.io import MAGICCData
from openscm.highlevel import ScmDataFrame


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
        df = MAGICCData(inname).to_iamdataframe().data

        scenario = file.split("_")[0]
        scenario = "historical" if scenario == "20THCENTURY" else scenario

        df["scenario"] = scenario
        df["model"] = models[scenario]
        df.drop("todo", axis="columns", inplace=True)

        outname = join(
            here,
            "..",
            "openscm",
            "scenarios",
            file.lower().replace(".dat", ".csv")
        )
        print("Saving {} as {}".format(inname, outname))

        ScmDataFrame(df).to_csv(outname)
