import pandas as pd

import numpy as np

 

# -----------------------------

# CONFIGURATION

# -----------------------------

CSV_FILE = "G1.csv"

start = pd.Timestamp("2026-08-14 00:00:00")

end   = pd.Timestamp("2026-11-11   00:00:00")  # Set this to the date AFTER the last date you want to see

 

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.set_option("display.width", 250)

pd.set_option("display.colheader_justify", "center")

 

 

# -----------------------------

# FUNCTIONS

# -----------------------------

def glucose_to_a1c(glucose_value):

    return (glucose_value + 46.7) / 28.7

 

 

def count_spikes(values, threshold):

    in_spike = False

    spikes = 0

    for v in values:

        if v >= threshold and not in_spike:

            spikes += 1

            in_spike = True

        elif v < threshold and in_spike:

            in_spike = False

    return spikes

 

 

def overnight_avg(values, timestamps):

    hours = timestamps.dt.hour

    return values[(hours >= 0) & (hours < 6)].mean()

 

 

def daytime_avg(values, timestamps):

    hours = timestamps.dt.hour

    return values[(hours >= 6) & (hours < 22)].mean()

 

 

def compute_mage(values):

    """

    Compute MAGE (Mean Amplitude of Glycemic Excursions)

    using the standard 1 SD threshold method.

    """

    if len(values) < 2:

        return np.nan

 

    sd = np.std(values)

    excursions = []

    direction = None

    start = values.iloc[0]

 

    for v in values:

        if direction is None:

            direction = "up" if v > start else "down"

 

        if direction == "up":

            if v - start >= sd:

                excursions.append(v - start)

                start = v

                direction = None

        else:

            if start - v >= sd:

                excursions.append(start - v)

                start = v

                direction = None

 

    if len(excursions) == 0:

        return 0

    return np.mean(excursions)

 

 

# -----------------------------

# LOAD & CLEAN DATA

# -----------------------------

df = pd.read_csv(CSV_FILE, header=None, names=["timestamp", "value"])

df["value"] = df["value"].replace("Out of range", 200)

 

df["timestamp"] = pd.to_datetime(

    df["timestamp"],

    format="%Y-%m-%dT%H:%M%z",

    errors="coerce"

).dt.tz_convert(None)

 

df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

 

df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(subset=["value"])

 

df = df.sort_values("timestamp")

 

df["date"] = df["timestamp"].dt.floor("D")

 

 

# -----------------------------

# BASIC STATISTICS

# -----------------------------

total_readings = len(df)

avg_glucose = df["value"].mean()

median_glucose = df["value"].median()

min_glucose = df["value"].min()

max_glucose = df["value"].max()

std_glucose = df["value"].std()

 

overall_cv = (std_glucose / avg_glucose) * 100

overall_mage = compute_mage(df["value"])

 

 

# -----------------------------

# TIME-IN-RANGE (overall)

# -----------------------------

below_140 = (df["value"] < 140).mean() * 100

below_180 = (df["value"] < 180).mean() * 100

above_180 = (df["value"] >= 180).mean() * 100

above_200 = (df["value"] >= 200).mean() * 100

 

spikes_140 = count_spikes(df["value"], 140)

spikes_160 = count_spikes(df["value"], 160)

spikes_180 = count_spikes(df["value"], 180)

spikes_200 = count_spikes(df["value"], 200)

 

 

# -----------------------------

# DAILY STATISTICS

# -----------------------------

daily_stats = df.groupby("date")["value"].agg(["mean", "max", "min", "median"])

daily_stats = daily_stats.reset_index()

 

grouped = df.groupby("date")

 

daily_stats["spikes_140"] = grouped["value"].apply(lambda x: count_spikes(x, 140)).values

daily_stats["spikes_160"] = grouped["value"].apply(lambda x: count_spikes(x, 160)).values

daily_stats["spikes_180"] = grouped["value"].apply(lambda x: count_spikes(x, 180)).values

daily_stats["spikes_200"] = grouped["value"].apply(lambda x: count_spikes(x, 200)).values

 

daily_stats["pct<140"] = grouped["value"].apply(lambda x: (x < 140).mean() * 100).values

daily_stats["pct<160"] = grouped["value"].apply(lambda x: (x < 160).mean() * 100).values

daily_stats["pct<180"] = grouped["value"].apply(lambda x: (x < 180).mean() * 100).values

daily_stats["pct≥180"] = grouped["value"].apply(lambda x: (x >= 180).mean() * 100).values

daily_stats["pct≥200"] = grouped["value"].apply(lambda x: (x >= 200).mean() * 100).values

 

daily_stats["overnight_avg"] = grouped.apply(

    lambda g: overnight_avg(g["value"], g["timestamp"])

).values

 

daily_stats["daytime_avg"] = grouped.apply(

    lambda g: daytime_avg(g["value"], g["timestamp"])

).values

 

# Daily SD, CV, MAGE

daily_stats["sd"] = grouped["value"].std().values

daily_stats["cv"] = (daily_stats["sd"] / daily_stats["mean"]) * 100

daily_stats["mage"] = grouped["value"].apply(lambda x: compute_mage(x)).values

 

 

# -----------------------------

# OVERNIGHT VS DAYTIME (overall)

# -----------------------------

df["hour"] = df["timestamp"].dt.hour

 

overnight = df[(df["hour"] >= 0) & (df["hour"] < 6)]

daytime = df[(df["hour"] >= 6) & (df["hour"] < 22)]

 

overnight_avg_overall = overnight["value"].mean()

daytime_avg_overall = daytime["value"].mean()

 

 

# -----------------------------

# ROLLING A1C (14-day and 30-day)

# -----------------------------

daily_stats["A1C"] = daily_stats["mean"].apply(glucose_to_a1c).round(2)

 

df_daily = df.copy()

df_daily["date"] = df_daily["timestamp"].dt.floor("D")

 

def rolling_a1c(window_days):

    results = []

    for d in daily_stats["date"]:

        start = d - pd.Timedelta(days=window_days)

        window_values = df_daily[(df_daily["date"] > start) & (df_daily["date"] <= d)]["value"]

        if len(window_values) == 0:

            results.append(None)

        else:

            mean_glucose = window_values.mean()

            results.append(glucose_to_a1c(mean_glucose))

    return results

 

daily_stats["A1C_14d"] = rolling_a1c(14)

daily_stats["A1C_30d"] = rolling_a1c(30)

 

daily_stats["A1C_14d"] = daily_stats["A1C_14d"].round(2)

daily_stats["A1C_30d"] = daily_stats["A1C_30d"].round(2)

 

 

# -----------------------------

# FORMATTING

# -----------------------------

glucose_cols = ["mean", "max", "min", "median", "overnight_avg", "daytime_avg"]

daily_stats[glucose_cols] = (

    daily_stats[glucose_cols]

    .round(0)

    .fillna(0)

    .astype(int)

)

 

pct_cols = ["pct<140", "pct<160", "pct<180", "pct≥180", "pct≥200"]

daily_stats[pct_cols] = daily_stats[pct_cols].round(1).fillna(0)

 

daily_stats["sd"] = daily_stats["sd"].round(1)

daily_stats["cv"] = daily_stats["cv"].round(1)

daily_stats["mage"] = daily_stats["mage"].round(1)

 

 

# -----------------------------

# OUTPUT

# -----------------------------

print("\n==============================")

print("      GLUCOSE SUMMARY")

print("==============================\n")

 

print(f"Total readings: {total_readings}")

print(f"Average glucose: {avg_glucose:.0f} mg/dL")

print(f"Median glucose: {median_glucose:.0f} mg/dL")

print(f"Minimum glucose: {min_glucose:.0f} mg/dL")

print(f"Maximum glucose: {max_glucose:.0f} mg/dL")

print(f"Standard deviation: {std_glucose:.0f}")

print(f"Coefficient of Variation (CV): {overall_cv:.1f}%")

print(f"MAGE (overall): {overall_mage:.1f} mg/dL")

 

print("\n--- Time in Range (Overall) ---")

print(f"Percent <140 mg/dL: {below_140:.1f}%")

print(f"Percent <180 mg/dL: {below_180:.1f}%")

print(f"Percent ≥180 mg/dL: {above_180:.1f}%")

print(f"Percent ≥200 mg/dL: {above_200:.1f}%")

 

print("\n--- Spike Counts (Overall) ---")

print(f"Spikes ≥140 mg/dL: {spikes_140}")

print(f"Spikes ≥160 mg/dL: {spikes_160}")

print(f"Spikes ≥180 mg/dL: {spikes_180}")

print(f"Spikes ≥200 mg/dL: {spikes_200}")

 

print("\n--- Overnight vs Daytime ---")

print(f"Overnight average (00:00–06:00): {overnight_avg_overall:.1f} mg/dL")

print(f"Daytime average (06:00–22:00): {daytime_avg_overall:.1f} mg/dL")

 

print("\n--- Daily Statistics ---")

print(daily_stats)

 

print("\nAnalysis complete.\n")