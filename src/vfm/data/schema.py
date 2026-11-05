"""Column contract between the raw Volve workbook and the rest of the code base.

Units follow the source: pressures in bar, temperatures in degC, choke opening in percent
and volumes in standard cubic metres (Sm3).
"""

from __future__ import annotations

SHEET_DAILY = "Daily Production Data"

RAW_TO_CANONICAL: dict[str, str] = {
    "DATEPRD": "date",
    "NPD_WELL_BORE_NAME": "well",
    "FLOW_KIND": "flow_kind",
    "WELL_TYPE": "well_type",
    "ON_STREAM_HRS": "on_stream_hrs",
    "AVG_DOWNHOLE_PRESSURE": "p_downhole",
    "AVG_DOWNHOLE_TEMPERATURE": "t_downhole",
    "AVG_DP_TUBING": "dp_tubing",
    "AVG_ANNULUS_PRESS": "p_annulus",
    "AVG_CHOKE_SIZE_P": "choke_pct",
    "AVG_WHP_P": "p_wellhead",
    "AVG_WHT_P": "t_wellhead",
    "DP_CHOKE_SIZE": "dp_choke",
    "BORE_OIL_VOL": "oil_vol",
    "BORE_GAS_VOL": "gas_vol",
    "BORE_WAT_VOL": "water_vol",
}

#: Gauges for which a reading of exactly zero is physically impossible while a well
#: flows; the source writes 0 for "no data", so these are treated as missing. Even a
#: fully open choke keeps a differential pressure above ~1.5 bar in this data set.
GAUGES_ZERO_IS_MISSING: tuple[str, ...] = (
    "p_downhole",
    "t_downhole",
    "p_wellhead",
    "t_wellhead",
    "dp_choke",
    "choke_pct",
)

#: Sensor columns exposed to the models (``dp_tubing`` is re-derived during cleaning).
SENSOR_COLUMNS: tuple[str, ...] = (
    "p_downhole",
    "t_downhole",
    "dp_tubing",
    "p_annulus",
    "choke_pct",
    "p_wellhead",
    "t_wellhead",
    "dp_choke",
)

VOLUME_COLUMNS: dict[str, str] = {
    "oil_vol": "oil_rate",
    "gas_vol": "gas_rate",
    "water_vol": "water_rate",
}

REQUIRED_RAW_COLUMNS: frozenset[str] = frozenset(RAW_TO_CANONICAL)
