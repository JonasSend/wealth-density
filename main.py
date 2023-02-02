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
from matplotlib import cm, animation
from matplotlib.colors import ListedColormap
from shapely.geometry import mapping
from PIL import Image, ImageDraw, ImageFont


### download data, unzip, and load into workspace ###

data_path = "data"
if not os.path.exists(data_path):
   os.makedirs(data_path)


url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_MT_GLOBE_R2019A/GHS_POP_E2015_GLOBE_R2019A_4326_30ss/V1-0/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.zip"
response = requests.get(url)
open("data/population_density.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/population_density.zip", "r") as zip_file:
    zip_file.extractall(data_path)
tif_file = rasterio.open(data_path + "/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif")  # population density raster


headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0"}  # fake a user agent
url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
response = requests.get(url, headers=headers)
open(data_path + "/country_shapes.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/country_shapes.zip", "r") as zip_file:
    zip_file.extractall(data_path)
shape_df = gpd.read_file(data_path + "/ne_10m_admin_0_countries.shp")


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
    

### save raster data as images ###

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

    fig, ax = plt.subplots(facecolor="#D5E9E8")
    ax.imshow(_map_array[0], norm=norm, cmap=our_cmap)
    ax.axis("off")
    plt.savefig(image_name + ".png", dpi=1000, bbox_inches="tight", pad_inches=0)


map_array = tif_file.read()
save_plot_map_array(map_array, "population_density")


# for wealth data, multiply population data by wealth per capita if available
# this gives a simple estimate of wealth/area in USD/km^2
# no data is treated as no wealth

map_array[0][map_array[0] > 0.0] = 0.0  # create "blank canvas" for wealth density

for country in wealth_df["country"]:
    shape = [mapping(shape_df["geometry"][shape_df["ADMIN"] == country].iloc[0])]
    country_array, out_transform = mask(tif_file, shape, nodata=0)

    country_array *= wealth_df.loc[wealth_df["country"] == country, "mean_wealth"].iloc[0]
    map_array += country_array
    print(country)

save_plot_map_array(map_array, "wealth_density")


### load both images and generate GIF that transitions between them ###

font = ImageFont.truetype("consola.ttf", 300)

population_image = Image.open('population_density.png')
image_draw = ImageDraw.Draw(population_image)
image_draw.text((1600, 1700), "population", font=font, fill=(0, 0, 0))
population_image_array = np.array(population_image.convert('RGB'))

wealth_image = Image.open('wealth_density.png')
image_draw = ImageDraw.Draw(wealth_image)
image_draw.text((1850, 1700), "wealth", font=font, fill=(0, 0, 0))
wealth_image_array = np.array(wealth_image.convert('RGB'))

image_shape = population_image_array.shape

# transform into flat arrays to facilitate easier processing
population_image_flat_array = population_image_array.reshape((-1,3))
wealth_image_flat_array = wealth_image_array.reshape((-1,3))

fig = plt.figure(dpi=600, frameon=False)
ax = plt.axes()
ax.axis("off")
empty_image = ax.imshow(np.zeros(image_shape))
fps = 25
frames = fps * 5


def animate(i):
    weight_wealth = calculate_weight(i)
    image = ax.imshow(np.zeros(image_shape))
    image.set_array(fade(population_image_flat_array, wealth_image_flat_array, weight_wealth).reshape(image_shape))
    return [image]


def fade(flat_image_1, flat_image_2, weight_image_2):
    faded_image = (flat_image_1 * (1 - weight_image_2)) + (flat_image_2 * weight_image_2)
    return faded_image.astype(int)


def calculate_weight(i):
    short_segment = round(frames * 1.5 / 5)
    long_segment = round(frames / 5)
    if i <= long_segment:
        return 0
    elif i <= short_segment + long_segment:
        return (i - long_segment) / short_segment
    elif i <= long_segment * 2 + short_segment:
        return 1
    else:
        return 1 - (i - (long_segment * 2 + short_segment)) / short_segment


anim = animation.FuncAnimation(fig, animate, frames=frames)
anim.save('animaton.gif', fps=fps)
