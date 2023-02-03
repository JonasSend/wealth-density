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
from matplotlib.image import AxesImage
from shapely.geometry import mapping
from PIL import Image, ImageDraw, ImageFont

# download data, unzip, and load into workspace #

data_path = "data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_MT_GLOBE_R2019A/GHS_POP_E2015_GLOBE_R2019A_4326_30ss/V1-0/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.zip"
response = requests.get(url)
open("data/population_density.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/population_density.zip", "r") as zip_file:
    zip_file.extractall(data_path)
population_density_raster = rasterio.open(data_path + "/GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif")

fake_user_agent = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0"}
url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
response = requests.get(url, headers=fake_user_agent)
open(data_path + "/country_shapes.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/country_shapes.zip", "r") as zip_file:
    zip_file.extractall(data_path)
shape_df = gpd.read_file(data_path + "/ne_10m_admin_0_countries.shp")

# retrieved from https://en.wikipedia.org/wiki/List_of_countries_by_wealth_per_adult on 2023-01-30
# data based on https://www.credit-suisse.com/media/assets/corporate/docs/about-us/research/publications/global-wealth-databook-2022.pdf
wealth_df = pd.read_csv("mean_wealth_per_country_2021.csv")
# change country names to match data
old_names = ["Bahamas", "DR Congo", "Congo", "Czech Republic", "Hong Kong",
             "São Tomé and Príncipe", "Serbia", "Tanzania", "United States"]
new_names = ["The Bahamas", "Democratic Republic of the Congo", "Republic of the Congo",
             "Czechia", "Hong Kong S.A.R.", "São Tomé and Principe", "Republic of Serbia",
             "United Republic of Tanzania", "United States of America"]
for i in range(len(old_names)):
    wealth_df.loc[wealth_df["country"] == old_names[i], "country"] = new_names[i]


# save raster data as images #

def save_as_map(_map_array: np.ndarray, image_name: str) -> None:
    map_values = np.concatenate(_map_array[0], axis=0)
    map_values = map_values[map_values > 0]
    # boundaries arbitrary - chosen for nice visual apearance
    bounds = [-200, -1] + list(np.percentile(map_values, [25, 43, 57, 69, 79, 87, 93, 97, 99]))

    our_cmap = cm.get_cmap("afmhot", 10)
    newcolors = our_cmap(np.linspace(0, 1, 10))
    background_colour = np.array([0, 0, 0, 0])
    newcolors = np.vstack((background_colour, newcolors))
    our_cmap = ListedColormap(newcolors)
    norm = colors.BoundaryNorm(bounds, our_cmap.N)

    _fig, _ax = plt.subplots(facecolor="#000000")
    _ax.imshow(_map_array[0], norm=norm, cmap=our_cmap)
    _ax.axis("off")
    plt.savefig(image_name + ".png", dpi=1000, bbox_inches="tight", pad_inches=0)


map_array = population_density_raster.read()
save_as_map(map_array, "population_density")

# for wealth data, multiply population data by wealth per capita
# this gives a simple estimate of wealth/area in USD/km^2
# replace no data with median

# create "blank canvas" for wealth density
map_array[0][map_array[0] > 0.0] = 0.0
median_wealth = wealth_df["mean_wealth"].median()

for country in shape_df["ADMIN"]:
    shape = [mapping(shape_df["geometry"][shape_df["ADMIN"] == country].iloc[0])]
    country_array, out_transform = mask(population_density_raster, shape, nodata=0)

    country_wealth = wealth_df.loc[wealth_df["country"] == country, "mean_wealth"]
    if len(country_wealth) == 0:
        country_wealth = median_wealth
    else:
        country_wealth = country_wealth.iloc[0]
    country_array *= country_wealth
    map_array += country_array
    print(country)

save_as_map(map_array, "wealth_density")


# load both images and generate loop GIF that transitions between them #

font = ImageFont.truetype("segoeui.ttf", 150)
# font colour: median colour of the plot's colour palette
font_colour = (255, 156, 28)
text_position = (250, 1900)

# it makes sense to manually adjust a few colour "anomalies" in the population density map before loading
# a bit hacky: create the image with and without text to let text fade in and out

population_image = Image.open('population_density.png')
population_image_with_text = population_image.copy()
image_draw = ImageDraw.Draw(population_image_with_text)
image_draw.text(text_position, "Population", font=font, fill=font_colour)
population_image_array = np.array(population_image.convert('RGB'))
population_image_with_text_array = np.array(population_image_with_text.convert('RGB'))

wealth_image = Image.open('wealth_density.png')
wealth_image_with_text = wealth_image.copy()
image_draw = ImageDraw.Draw(wealth_image_with_text)
image_draw.text(text_position, "Wealth", font=font, fill=font_colour)
wealth_image_array = np.array(wealth_image.convert('RGB'))
wealth_image_with_text_array = np.array(wealth_image_with_text.convert('RGB'))

image_shape = population_image_array.shape

# transform into flat arrays to facilitate easier processing
population_image_flat_array = population_image_array.reshape((-1, 3))
population_image_with_text_flat_array = population_image_with_text_array.reshape((-1, 3))
wealth_image_flat_array = wealth_image_array.reshape((-1, 3))
wealth_image_with_text_flat_array = wealth_image_with_text_array.reshape((-1, 3))

# set up images for animation
fig = plt.figure(dpi=600, frameon=False)
ax = plt.axes()
ax.axis("off")
empty_image = ax.imshow(np.zeros(image_shape))
fps = 5
# show each image for 1.5 seconds and use 1 second for transition between the two
frames = fps * 5
short_segment = round(frames * 1.5 / 5)
long_segment = round(frames / 5)


def animate(_i: int) -> AxesImage:
    """use empty image and fill with content for each frame
    hacky if-else clause to determine which images are used for fade-ins and -outs
    """
    if _i <= long_segment:
        empty_image.set_array(population_image_with_text_flat_array.reshape(image_shape))
    elif _i <= round(short_segment / 2) + long_segment:
        empty_image.set_array(
            fade(population_image_with_text_flat_array, 1 - (2 * (_i - long_segment) / short_segment)).reshape(
                image_shape))
    elif _i <= short_segment + long_segment:
        empty_image.set_array(
            fade(wealth_image_with_text_flat_array, (2 * (_i - long_segment) / short_segment) - 1).reshape(image_shape))
    elif _i <= long_segment * 2 + short_segment:
        empty_image.set_array(wealth_image_with_text_flat_array.reshape(image_shape))
    elif _i <= round(short_segment / 2) + long_segment * 2 + short_segment:
        empty_image.set_array(fade(wealth_image_with_text_flat_array,
                                   1 - (2 * (_i - (long_segment * 2 + short_segment)) / short_segment)).reshape(
            image_shape))
    else:
        empty_image.set_array(fade(population_image_with_text_flat_array,
                                   (2 * (_i - (long_segment * 2 + short_segment)) / short_segment) - 1).reshape(
            image_shape))
    return [empty_image]


def fade(image_flat_array: np.ndarray, weight_image: float) -> np.ndarray:
    """mix image with text with a 50/50 mix of population and wealth images without text"""
    weight_other_images = (1 - weight_image)
    faded_image = (image_flat_array * weight_image) + (
                (population_image_flat_array + wealth_image_flat_array) * weight_other_images)
    return faded_image.astype(int)


population_wealth_animation = animation.FuncAnimation(fig, animate, frames=frames)
population_wealth_animation.save('population_vs_wealth_density.gif', fps=fps)
