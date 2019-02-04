from openscm.scenarios import rcps

def test_rcp26():
    value = rcps.filter(
        variable="Emissions|CH4",
        region="World",
        year=1994,
        scenario="RCP26"
    )["value"]

    assert value == 12

def test_rcp45():
    value = rcps.filter(
        variable="Emissions|BC",
        region="World",
        year=1765,
        scenario="RCP45"
    )["value"]
    
    assert value == 12

def test_rcp60():
    value = rcps.filter(
        variable="Emissions|CO2|MAGICC Fossil and Industrial",
        region="World",
        year=2034,
        scenario="RCP60"
    )["value"]
    
    assert value == 12

def test_rcp85():
    value = rcps.filter(
        variable="Emissions|N2O",
        region="World",
        year=2500,
        scenario="RCP85"
    )["value"]
    
    assert value == 12


