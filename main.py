import os
import requests
import zipfile
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from shapely.geometry import mapping


data_path = "data"
if not os.path.exists(data_path):
   os.makedirs(data_path)

#
# TODO consolidate quotes
#

# population density data
url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_MT_GLOBE_R2019A/GHS_POP_E2015_GLOBE_R2019A_4326_30ss/V1-0/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.zip"
response = requests.get(url)
open("data/population_density.zip", "wb").write(response.content)

with zipfile.ZipFile(data_path + "/population_density.zip", "r") as zip_file:
    zip_file.extractall(data_path)
    
tif_file = rasterio.open(data_path + "/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif")

# country shapes
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0"}  # fake a user agent
url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
response = requests.get(url, headers=headers)
open(data_path + "/country_shapes.zip", "wb").write(response.content)

with zipfile.ZipFile(data_path + "/country_shapes.zip", "r") as zip_file:
    zip_file.extractall(data_path)
    
shape_df = gpd.read_file(data_path + "/ne_10m_admin_0_countries.shp")

# wealth data
# retrieved from https://en.wikipedia.org/wiki/List_of_countries_by_wealth_per_adult on 2023-01-30
# data based on https://www.credit-suisse.com/media/assets/corporate/docs/about-us/research/publications/global-wealth-databook-2022.pdf
wealth_df = pd.read_csv("mean_wealth_per_country_2021.csv")

old_names = ["Bahamas", "DR Congo", "Congo", "Czech Republic", "Hong Kong",
             "São Tomé and Príncipe", "Serbia", "Tanzania", "United States"]
new_names = ["The Bahamas", "Democratic Republic of the Congo", "Republic of the Congo",
             "Czechia", "Hong Kong S.A.R.", "São Tomé and Principe", "Republic of Serbia",
             "United Republic of Tanzania", "United States of America"]

for i in range(len(old_names)):
    wealth_df.loc[wealth_df["country"] == old_names[i], "country"] = new_names[i]
    
    
def save_plot_map_array(_map_array: np.ndarray, image_name: str) -> None:
    map_values = np.concatenate(_map_array[0], axis=0)
    map_values = map_values[map_values > 0]
    bounds = [-200, -1] + list(np.percentile(map_values, [25, 43, 57, 69, 79, 87, 93, 97, 99]))

    our_cmap = cm.get_cmap("hot_r", 10)
    newcolors = our_cmap(np.linspace(0, 1, 10))
    background_colour = np.array([0, 0, 0, 0])
    newcolors = np.vstack((background_colour, newcolors))
    our_cmap = ListedColormap(newcolors)
    norm = colors.BoundaryNorm(bounds, our_cmap.N)

    #
    # TODO: choose background colour
    # light blue with red hue: #D5E9E8
    #
    fig, ax = plt.subplots(facecolor="#1e0f5a")
    ax.imshow(_map_array[0], norm=norm, cmap=our_cmap)
    ax.axis("off")
    plt.savefig(image_name + ".png", dpi=1000, bbox_inches="tight", pad_inches=0)


map_array_population = tif_file.read()
save_plot_map_array(map_array_population, "population_density")

map_array_wealth = map_array_population.copy()
map_array_wealth[0][map_array_wealth[0] > 0.0] = 0.0  # create "blank canvas" for wealth density

for country in wealth_df["country"]:
    shape = [mapping(shape_df["geometry"][shape_df["ADMIN"] == country].iloc[0])]
    country_array, out_transform = mask(tif_file, shape, nodata=0)

    # simple estimate of wealth/area in USD/km^2
    country_array *= wealth_df.loc[wealth_df["country"] == country, "mean_wealth"]
    map_array_wealth += country_array
    print(country)

save_plot_map_array(map_array_wealth, "wealth_density")

#
# TODO: transition animation etc.
#



