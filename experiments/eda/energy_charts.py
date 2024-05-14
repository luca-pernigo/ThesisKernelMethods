import datetime
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns


country="CH"
country_name="Switzerland"
year=2021

df=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/{year}/clean/{country.lower()}.csv")

dates=pd.to_datetime(df.Time, format='%Y-%m-%d %H:%M:%S')


# load plot
plt.figure(figsize=(15,5))
plt.plot(dates, df["Load"], color="black")

plt.xlabel("Time")
plt.ylabel("Load(MW)")
# plt.title(f"Energy load {country_name} ({year})")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_historical_load_{year}.png")
plt.show()

# load vs temperature
plt.plot(df["Temperature"], df["Load"], "o", alpha=0.4, markersize=3)

plt.ylabel("Load (MW)")
plt.xlabel("Temperature")
# plt.title(f"Load versus temperature {country_name} ({year})")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_load_vs_temperature_{year}.png")
plt.show()

# load vs wind_speed
plt.plot(df["Wind_speed"], df["Load"], "o", alpha=0.4, markersize=3)

plt.ylabel("Load (MW)")
plt.xlabel("Wind speed")
# plt.title(f"Load versus wind speed {country_name} ({year})")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_load_vs_wind_speed_{year}.png")
plt.show()


# box plot load vs day of the week
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(12, 9))
xvalues = ["Workday", "Holiday"]
palette = ['r', 'b']


sns.boxplot(data=df, x='Is_holiday', y='Load', palette=palette,
            width=0.7, dodge=False, ax=ax)

ax.xaxis.set_ticklabels(xvalues)

ax.set_xlabel('Day type')
ax.set_ylabel("Load(MW)", fontsize=14)
# plt.title(f"Load by day type {country_name} ({year})")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_is_holiday_boxplot_{year}.png")
plt.show()


# box plot load vs holiday
# group by
# plot all box plots
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(12, 9))
xvalues = ["Monday", "Tuesday", "Wednesday", "Thurday", "Friday", "Saturday", "Sunday"]
palette = ['plum', 'g', 'orange', 'b', 'r','grey','yellow']


sns.boxplot(data=df, x='Day_of_week', y='Load', palette=palette,
            width=0.7, dodge=False, ax=ax)

ax.xaxis.set_ticklabels(xvalues)

ax.set_xlabel('Day of week')
ax.set_ylabel("Load(MW)", fontsize=14)
# plt.title("Load by day of the week Switzerland (2021)")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_day_of_week_boxplot_{year}.png")
plt.show()

# heat map
plt.figure(figsize=(10,5))
hot= df[["Load","Temperature", "Wind_speed", "Day_of_week", "Is_holiday"]].corr()
sns.heatmap(hot,cmap="coolwarm",annot=True, alpha=0.9)
# plt.title("Load versus independent variables heatmap")
plt.savefig(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/images/{country}_heatmap_{year}.png")
plt.show()