from earthquake_df_load import fetch_and_load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import folium

output="images"
os.makedirs(output,exist_ok=True)

# INSPECT DATA
#-----------------------------------
def inspect_data(df):
    print("\n--- HEAD ---")
    print(df.head(10))

    print("\n--- TAIL ---")
    print(df.tail(10))

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE ---")
    print(df.describe())

    print("\n--- ISNULL ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATED ---")
    print(df.duplicated().sum())

    print("\n--- COLUMNS ---")
    print(df.columns)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- CORRELATION TABLE ---")
    print(df.corr(numeric_only=True)) 
    """
    --- CORRELATION TABLE ---
             Lat.degrees  Lon.degrees   Depthkm      Mag.  Year     Month       Day
Lat.degrees     1.000000    -0.107134 -0.423601 -0.445240   NaN  0.016759 -0.021858
Lon.degrees    -0.107134     1.000000 -0.077835  0.056439   NaN -0.012639  0.009791
Depthkm        -0.423601    -0.077835  1.000000  0.328083   NaN  0.007079 -0.004680
Mag.           -0.445240     0.056439  0.328083  1.000000   NaN -0.030410  0.034590
Year                 NaN          NaN       NaN       NaN   NaN       NaN       NaN
Month           0.016759    -0.012639  0.007079 -0.030410   NaN  1.000000 -0.991515
Day            -0.021858     0.009791 -0.004680  0.034590   NaN -0.991515  1.000000
"""  
# CLEAR DATA
#-----------------------------------
def clear_data(df):
    df.drop_duplicates(inplace=True)
    df.dropna(subset=df.columns,inplace=True)

    df["Date & TimeUTC"] = df["Date & TimeUTC"].str.split("\n").str[0]
    df["Date & TimeUTC"] = pd.to_datetime(df["Date & TimeUTC"])

    df["Year"] = df["Date & TimeUTC"].dt.year
    df["Date"] = df["Date & TimeUTC"].dt.date
    df["Month"] = df["Date & TimeUTC"].dt.month
    df["Day"] = df["Date & TimeUTC"].dt.day
    df["Time"] = df["Date & TimeUTC"].dt.time

    df["Year"]=df["Year"].astype(float)

    return df

# ANALYSIS OF DAILY EARTHQUAKE
#-----------------------------------
def analysis_of_daily_earthquakes(df):
    plt.figure(figsize=(10,6))
    eartquakes_daily=df["Date"].value_counts().sort_index(ascending=True)
    sb.barplot(x=eartquakes_daily.index,y=eartquakes_daily.values)
    plt.title("Daily Number of Recent Earthquakes")
    plt.xlabel("Date")
    plt.ylabel("Number of Eartquakes")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{output}/analysis_of_daily_earthquakes.png")
    plt.show()

# ANALYSIS OF TOP 50 REGIONS IN EARTHQUAKE
#-----------------------------------
def analysis_of_top50_regions(df):
    plt.figure(figsize=(10,10))
    eartquakes_vs_countries=df["Region"].value_counts().sort_values(ascending=False).head(50)
    sb.barplot(y=eartquakes_vs_countries.index,x=eartquakes_vs_countries.values)
    plt.title("Top 50 Regions Where Recent Earthquakes Have Occurred Most Frequently")
    plt.ylabel("Regions")
    plt.xlabel("Number of Eartquakes")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{output}/analysis_of_top50_regions.png")
    plt.show()

# EARTHQUAKE MAP
#-----------------------------------
def earthquakeMap(df):
    central_lat=df["Lat.degrees"].mean()
    central_lon=df["Lon.degrees"].mean()

    map = folium.Map(location=[central_lat, central_lon], zoom_start=2, tiles='OpenStreetMap')

    for index, row in df.iterrows():
        folium.Marker(
            location=[row['Lat.degrees'], row['Lon.degrees']],
            popup=row['Region'],
            tooltip="Click for detail",
            icon=folium.Icon(color='red')
        ).add_to(map)
    
    map.save(f"{output}/map.html")
    
    print("Saved Map as html file")

# ANALYSIS OF MAGNITUDE EARTHQUAKE
#-----------------------------------
def analysis_of_magnitude_of_earthquakes(df):
    plt.figure(figsize=(20,8))
    sb.scatterplot(data=df,x="Date & TimeUTC",y="Mag.")
    plt.title("Magnitudes of Earthquakes Occurring Over Time")
    plt.xlabel("Date")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{output}/analysis_of_magnitude_of_earthquakes.png")
    plt.show()

# ANALYSIS OF TOP 10 EARTHQUAKE
#-----------------------------------
def analysis_of_top_10_earthquake(df):
    plt.figure(figsize=(10,15))
    top_magnitudes = df.sort_values(by="Mag.", ascending=False).head(10)
    sb.barplot(data=top_magnitudes,x="Region",y="Mag.")
    plt.xticks(rotation=15,ha="right",size=8)
    plt.ylim(5.4,6.3)
    plt.title("The 10 Largest Recent Earthquakes and Their Locations")
    plt.xlabel("Region")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output}/analysis_of_top_10_earthquake.png")
    plt.show()

# MAIN PIPELINE
#-----------------------------------
def main():
    fetch_and_load_data()
    df=pd.read_csv("dataset/dataset.csv")
    inspect_data(df)
    df=clear_data(df)
    analysis_of_daily_earthquakes(df)
    analysis_of_top50_regions(df)
    earthquakeMap(df)
    analysis_of_top_10_earthquake(df)

    df.to_csv("dataset/dataset_edited.csv")

# RUN
#-----------------------------------
if __name__=="__main__":
    main()