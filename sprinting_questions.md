- where do we store readers specific to MAGICC? Do we make pymagicc a dependency of OpenSCM?
- should OpenSCM try and push everything onto models or should models request what they want? How do you handle case where e.g. emissions inputs have regional detail but model only runs on global inputs or emissions inputs have C6F14 but model doesn't use that to run?

- high level stores dataframe with multiple scenarios
- it pushes the scenarios, one at a time, into low level
- hence scenario filtering etc. is handled in high level, parameter set literally contains parameter set for that run and that run only
