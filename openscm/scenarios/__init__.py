from os.path import join, dirname, realpath


from ..highlevel import ScmDataFrame


here = dirname(realpath(__file__))

rcps = ScmDataFrame(join(here, "rcp26_emissions.csv"))
for rcp in ["rcp45", "rcp60", "rcp85"]:
	rcps.append(join(here, "{}_emissions.csv".format(rcp)))