#!/usr/bin/env python
# coding: utf-8

# Chicago Crime Data Analysis
# Data Source: Kaggle - Crimes in Chicago

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("Libraries loaded successfully")

DATA_DIR = "data/"
csv_files = glob.glob(DATA_DIR + "Chicago_Crimes_*.csv")
print(f"Found {len(csv_files)} data files")
for f in csv_files:
    print(f"  - {f}")
try:
    chunks = []
    for f in csv_files:
        df_chunk = pd.read_csv(f, nrows=20000, low_memory=False)
        chunks.append(df_chunk)
        print(f"  Loaded {f}: {len(df_chunk)} rows")
    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal samples: {len(df)} rows")

except FileNotFoundError:
    print("Data files not found! Please check data/ directory")

df.info()

df.describe(include="all")

# Data Preprocessing

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = df["Date"].dt.hour
df["Weekday"] = df["Date"].dt.dayofweek
def get_season(m):
    if m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    elif m in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"
df["Season"] = df["Month"].apply(get_season)
print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
print("Missing values:")
print(df.isnull().sum())

# 4. Analysis and Visualization

# 4.1 Crime Types Distribution

top_crimes = df["Primary Type"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(top_crimes)), top_crimes.values, color="#1F3A5F")
ax.set_yticks(range(len(top_crimes)))
ax.set_yticklabels(top_crimes.index)
ax.set_xlabel("Number of Crimes")
ax.set_title("Crime Types - Top 10", fontsize=14, fontweight="bold")
for bar, val in zip(bars, top_crimes.values):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=9)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig("output/crime_types.png", dpi=150, bbox_inches="tight")
plt.show()

# 4.2 Time Patterns

hourly = df["Hour"].value_counts().sort_index()
monthly = df.groupby("Month").size()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hourly.index, hourly.values, marker="o", linewidth=2, color="#1F3A5F")
axes[0].set_xticks(range(0, 24, 2))
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Crime Count")
axes[0].set_title("Crime Count by Hour", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[1].bar(monthly.index, monthly.values, color="#C0392B", alpha=0.8)
axes[1].set_xticks(range(1, 13))
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Crime Count")
axes[1].set_title("Crime Count by Month", fontsize=12, fontweight="bold")
for i, v in zip(monthly.index, monthly.values):
    axes[1].text(i, v + 50, f"{v:,}", ha="center", fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig("output/time_patterns.png", dpi=150, bbox_inches="tight")
plt.show()

# 4.3 Arrest Rate Analysis

arrest_rate = df.groupby("Primary Type")["Arrest"].mean().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#27AE60" if v > 0.3 else "#E74C3C" if v < 0.1 else "#F39C12" for v in arrest_rate.values]
bars = ax.barh(range(len(arrest_rate)), arrest_rate.values * 100, color=colors)
ax.set_yticks(range(len(arrest_rate)))
ax.set_yticklabels(arrest_rate.index)
ax.set_xlabel("Arrest Rate (%)")
ax.set_title("Arrest Rate by Crime Type (Top 15)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, arrest_rate.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1%}", va="center", fontsize=9)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig("output/arrest_rate.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Overall arrest rate: {df['Arrest'].mean():.1%}")

# 4.4 Area Distribution

top_areas = df["Community Area"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(top_areas)), top_areas.values, color="#8E44AD", alpha=0.8)
ax.set_yticks(range(len(top_areas)))
ax.set_yticklabels([f"Area {int(i)}" for i in top_areas.index])
ax.set_xlabel("Crime Count")
ax.set_title("Top 15 Community Areas by Crime Count", fontsize=14, fontweight="bold")
for bar, val in zip(bars, top_areas.values):
    ax.text(val + 10, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=9)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig("output/area_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# 4.5 Weekday Pattern

weekday_map = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
weekday_counts = df["Weekday"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(weekday_map, weekday_counts.values, color="#D35400", alpha=0.85)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Crime Count")
ax.set_title("Crime Count by Day of Week", fontsize=14, fontweight="bold")
for bar, val in zip(bars, weekday_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 20, f"{val:,}", ha="center", fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("output/weekday_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

# 4.6 Seasonal Pattern

season_order = ["Spring", "Summer", "Autumn", "Winter"]
season_counts = df["Season"].value_counts()
season_counts = season_counts.reindex([s for s in season_order if s in season_counts.index])
fig, ax = plt.subplots(figsize=(8, 5))
colors_list = ["#27AE60", "#E74C3C", "#E67E22", "#3498DB"]
bars = ax.bar(season_counts.index, season_counts.values, color=colors_list[:len(season_counts)], alpha=0.85)
ax.set_xlabel("Season")
ax.set_ylabel("Crime Count")
ax.set_title("Crime Count by Season", fontsize=14, fontweight="bold")
for bar, val in zip(bars, season_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 20, f"{val:,}", ha="center", fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig("output/season_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. Conclusion
# Key Findings
# 1. Theft: Most common crime type
# 2. Nighttime: Crime peaks in evening/night hours
# 3. Arrest Rate: Varies significantly by crime type
# 4. Seasonality: Certain seasons show higher crime rates
# 5. Hotspots: Crime concentrated in specific community areas
# Limitations
# - Sample data only (subset of full dataset)
# - Missing coordinates for geo-spatial analysis
# - No demographic or socioeconomic data included
# - Further statistical modeling needed for causation
