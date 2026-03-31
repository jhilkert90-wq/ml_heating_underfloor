#!/usr/bin/env python3
import pandas as pd
from influxdb import InfluxDBClient


def filter_pv_like_ha(series, timeout_seconds=900):
    """Replicate the Home Assistant PV filter template:
      - current > 0  → use current value
      - time since raw value last changed > timeout_seconds → 0
      - otherwise → hold previous filtered value
    """
    raw = series.values.astype(float)
    timestamps = series.index
    filtered = raw.copy()

    last_change_time = timestamps[0]
    prev_raw = raw[0]

    for i in range(len(raw)):
        if raw[i] != prev_raw:
            last_change_time = timestamps[i]
            prev_raw = raw[i]

        if raw[i] > 0:
            filtered[i] = raw[i]
        elif (timestamps[i] - last_change_time).total_seconds() > timeout_seconds:
            filtered[i] = 0.0
        else:
            filtered[i] = filtered[i - 1] if i > 0 else 0.0

    return pd.Series(filtered, index=series.index, name=series.name)

# -----------------------------------
# CONFIG
# -----------------------------------
INFLUX_V1_HOST = "localhost"
INFLUX_V1_PORT = 8086
INFLUX_V1_DB = "db"  # deine Influx v1 Datenbank
EXPORT_FILE = "ml_heating_import.csv"
ZIEL_TEMP = 22.7
START_TIME = "2025-10-01T00:00:00Z"

# -----------------------------------
# CONNECT TO INFLUXDB V1
# -----------------------------------
client = InfluxDBClient(
    host=INFLUX_V1_HOST,
    port=INFLUX_V1_PORT,
    database=INFLUX_V1_DB
)

print("ðŸ“¥ Hole Daten aus InfluxDB v1...")

# Sensordaten: VLT, AT
query_sensors = f"""
SELECT mean("VLT") AS VLT, mean("RLT") AS RLT, mean("Pel") AS Pel, mean("AT") AS AT, mean("BT53") AS BT53, mean("Sonnenschutz_aktiv") AS Sonnenschutz_aktiv, mean("Pth_WW") AS Pth_WW, mean("Heizung_Soll") AS Heizung_Soll
FROM "Sensordaten"
WHERE time >= '{START_TIME}'
GROUP BY time(5m) fill(previous)
"""

# Raumtemperaturen: RT_Flur_OG, RT_Flur_EG
query_rt = f"""
SELECT mean("RT_Flur_OG") AS RT_Flur_OG, mean("RT_Flur_EG") AS RT_Flur_EG, mean("RT_Bad_OG") AS RT_Bad_OG, mean("RT_WZ") AS RT_WZ
FROM "Raumtemperaturen"
WHERE time >= '{START_TIME}'
GROUP BY time(5m) fill(previous)
"""


# Kurvensteuerung: current_RT
query_curves = f"""
SELECT mean("current_RT") AS current_RT
FROM "Kurvensteuerung"
WHERE time >= '{START_TIME}'
GROUP BY time(5m) fill(previous)
"""

# PV Erzeugung: PV_Generate
query_pv = f"""
SELECT mean("PV_Generate") AS PV_Generate
FROM "PV-Daten"
WHERE time >= '{START_TIME}'
GROUP BY time(5m) fill(previous)
"""


result_sensors = client.query(query_sensors)
result_curves = client.query(query_curves)
result_pv = client.query(query_pv)
result_rt = client.query(query_rt)


df_sensors = pd.DataFrame(list(result_sensors.get_points()))
df_curves = pd.DataFrame(list(result_curves.get_points()))
df_pv = pd.DataFrame(list(result_pv.get_points()))
df_rt = pd.DataFrame(list(result_rt.get_points()))

# -----------------------------------
# MERGE DATAFRAMES
# -----------------------------------
df_sensors['time'] = pd.to_datetime(df_sensors['time'])
df_curves['time'] = pd.to_datetime(df_curves['time'])
df_pv['time'] = pd.to_datetime(df_pv['time'])
df_rt['time'] = pd.to_datetime(df_rt['time'])


df = pd.merge(df_sensors, df_curves, on="time", how="outer")
df = pd.merge(df, df_pv, on="time", how="outer")
df = pd.merge(df, df_rt, on="time", how="outer")

df = df.sort_values("time")
df = df.set_index("time")
print(f"ðŸ“Š Gesamtzeilen nach Merge: {len(df)}")


df["rt_mittelwert"] = df[
    ["RT_Flur_EG", "RT_Flur_OG", "RT_WZ", "RT_Bad_OG"]
].median(axis=1)
# -----------------------------------
# UMBENENNEN UND ZIELTEMP HINZUFÜGEN
# -----------------------------------
df = df.rename(columns={
    "VLT": "nibe_bt2_supply_temp_s1",
    "RLT": "nibe_eb100_ep14_bt3_return_temp",
    "AT": "nibe_bt1_outdoor_temperature",
    "Pel": "nibe_el_leistung",
    "current_RT": "current_rt",
    "RT_Flur_OG": "rt_flur_og",
    "PV_Generate": "pv_leistung_gefiltert",
    "Sonnenschutz_aktiv": "sonnenschutz_aktiv",
    "Heizung_Soll": "ml_vorlauftemperatur"
})
df["pv_leistung_gefiltert"] = filter_pv_like_ha(df["pv_leistung_gefiltert"])
df["soll_rt"] = ZIEL_TEMP
df["tv_an"] = ((df.index.hour >= 6) & (df.index.hour < 22)).astype(int)
df["hp_current_flow_rate"] = 12.2
df["kamin_an"] = ((df["BT53"] > 55) & df["BT53"].notna()).astype(int)
df["warmwassermodus"] = (df["Pth_WW"] > 0).astype(int)


# Spaltenreihenfolge fÃ¼r ML Heating
df = df[[
    "nibe_bt2_supply_temp_s1",
    "nibe_bt1_outdoor_temperature",
    "nibe_eb100_ep14_bt3_return_temp",
    "hp_current_flow_rate",
    "nibe_el_leistung",
    "current_rt",
    "soll_rt",
    "rt_flur_og",
    "rt_mittelwert",
    "pv_leistung_gefiltert",
    "tv_an",
    "kamin_an",
    "warmwassermodus",
    "sonnenschutz_aktiv",
    "ml_vorlauftemperatur"
]]

# -----------------------------------
# CSV FÃœR INFLUXDB V2 EXPORT (ANNOTATED CSV)
# -----------------------------------
EXPORT_FILE = "ml_heating_import.csv"
print("ðŸ“¤ Erzeuge InfluxDB v2 kompatible CSV...")

with open(EXPORT_FILE, "w") as f:
    # group (10 Spalten)
    f.write("#group,false,false,true,true,false,false,true,true,true\n")

    # datatype (10 Spalten)
    f.write("#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,"
            "dateTime:RFC3339,double,string,string,string\n")

    # default (10 Spalten)
    f.write("#default,mean,,,,,,,,\n")

    # Header (10 Spalten)
    f.write(",result,table,_start,_stop,_time,_value,_field,_measurement,entity_id\n")

    start_time = df.index.min().tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    stop_time = df.index.max().tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')

    for idx, (ts, row) in enumerate(df.iterrows()):
        ts_utc = ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        for field, value in row.items():
            f.write(
                f",,{idx},{start_time},{stop_time},{ts_utc},"
                f"{float(value):.2f},value,sensor,{field}\n"
            )


print(f"âœ… Fertig: {EXPORT_FILE} erzeugt â€“ bereit fÃ¼r InfluxDB v2 Import")
