from datetime import datetime


def convert_datetime_to_openscm_time(dt_in):
    openscm_reference_time = datetime(1970, 1, 1, 0, 0, 0)
    return (dt_in - openscm_reference_time).total_seconds()
