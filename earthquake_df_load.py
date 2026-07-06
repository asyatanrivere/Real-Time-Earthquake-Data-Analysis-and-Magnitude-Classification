from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import random
import os


def fetch_and_load_data():

    dataset="dataset"

    os.makedirs(dataset,exist_ok=True)
    
    url="https://www.emsc-csem.org/"

    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0",
        "Referer":"https://www.google.com/",
        "Accept-Language":"tr,en;q=0.9,en-GB;q=0.8,en-US;q=0.7"

    }
    time.sleep(random.uniform(1,5))
    
    page=requests.get(url,headers=headers)
    soup=BeautifulSoup(page.text,"html.parser")

    table=soup.find_all("table")[0]
    titles=table.find_all("th")
    table_titles=[item.text.strip() for item in titles]
    del table_titles[0]
    del table_titles[0] # ['Date & TimeUTC', 'Lat.degrees', 'Lon.degrees', 'Depthkm', 'Mag.', 'Region']

    df=pd.DataFrame(columns=table_titles)

    driver=webdriver.Chrome()

    driver.get(url)
    driver.minimize_window()
    sleep(3)
    
    # FETCHING DATAS
    #-----------------------------------

    eartquake_date=driver.find_elements(By.CLASS_NAME,"tbdat")
    eartquake_date_data = [element.text for element in eartquake_date]

    eartquake_lat=driver.find_elements(By.CLASS_NAME,"tblat")
    eartquake_lat_data=[element.text for element in eartquake_lat]

    eartquake_lon=driver.find_elements(By.CLASS_NAME,"tblon")
    eartquake_lon_data=[element.text for element in eartquake_lon]

    eartquake_depth=driver.find_elements(By.CLASS_NAME,"tbdep")
    eartquake_depth_data=[element.text for element in eartquake_depth]

    eartquake_mag=driver.find_elements(By.CLASS_NAME,"tbmag")
    eartquake_mag_data=[element.text for element in eartquake_mag]

    eartquake_region=driver.find_elements(By.CLASS_NAME,"tbreg")
    eartquake_region_data=[element.text for element in eartquake_region]

    print(len(eartquake_date_data))
    print(len(eartquake_lat_data))
    print(len(eartquake_lon_data))
    print(len(eartquake_depth_data))
    print(len(eartquake_mag_data))
    print(len(eartquake_region_data))

    for i in range(1,3001):
        lenght=len(df)
        df.loc[lenght]=[eartquake_date_data[i],eartquake_lat_data[i],eartquake_lon_data[i],eartquake_depth_data[i],eartquake_mag_data[i],eartquake_region_data[i]]

    driver.quit()

    df.to_csv(f"{dataset}/dataset.csv",index=False)
