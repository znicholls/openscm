from os.path import join, dirname, realpath


from ..highlevel import ScmDataFrame


here = dirname(realpath(__file__))

rcps = ScmDataFrame(join(here, "rcp26_emissions.csv"))
"""ScmDataFrame: RCP emissions data
"""

# appending, whether from file or from DataFrame instance 
# is super slow and I don't get why
rcp45 = ScmDataFrame(join(here, "rcp45_emissions.csv"))
rcps.append(rcp45)
for rcp in ["rcp60", "rcp85"]:
	rcps.append(join(here, "{}_emissions.csv".format(rcp)))