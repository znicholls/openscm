from base import _AdapterTester

from openscm.adapters.ph99 import PH99

from conftest import assert_pint_equal

class TestPH99Adapter(_AdapterTester):
    tadapter = PH99

    def test_initialize(self, test_adapter):
        assert test_adapter.model is None
        super().test_initialize(test_adapter)
        assert test_adapter.model is not None

    def test_set_config(self, test_adapter, test_config_paraset):
        super().test_set_config(test_adapter, test_config_paraset)

        tc1 = 3.8
        test_config_paraset.get_writable_scalar_view("c1", ("World",), "ppb").set(
            tc1 * 1000
        )

        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)

        assert_pint_equal(test_adapter.model.c1, tc1 * unit_registry("ppm"))

    def test_run(self, test_adapter, test_config_paraset, test_drivers_core):
        super().test_run(test_adapter, test_config_paraset, test_drivers_core)

        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)
        test_adapter.set_drivers(test_drivers_core)
        res = test_adapter.run()

        def get_comparison_time_for_year(yr):
            return convert_datetime_to_openscm_time(datetime.datetime(yr, 1, 1))

        assert_core(
            10.1457206,
            get_comparison_time_for_year(2017),
            res,
            ("Emissions", "CO2"),
            "World",
            "GtC / yr",
            res.start_time,
            ONE_YEAR_IN_S_INTEGER,
        )

        assert_core(
            1.632585,
            get_comparison_time_for_year(2100),
            res,
            ("Surface Temperature"),
            "World",
            "K",
            res.start_time,
            ONE_YEAR_IN_S_INTEGER,
        )
