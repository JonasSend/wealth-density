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
from matplotlib import colormaps, animation
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage
from shapely.geometry import mapping
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngImageFile


""" download and process data """

data_path = "data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

url = "http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"
response = requests.get(url)
open("data/population_density.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/population_density.zip", "r") as zip_file:
    zip_file.extractall(data_path)
population_density_raster = rasterio.open(data_path + "/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")

url = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel"
response = requests.get(url)
open(data_path + "/population.xls", "wb").write(response.content)
population_df = pd.read_excel(data_path + "/population.xls", sheet_name=0, skiprows=3, engine='xlrd')
population_df = population_df[["Country Code", "2022"]]
population_df = population_df.rename(columns={"Country Code": "country_code", "2022": "population"})

# total welath by country in million USD
# retrieved from https://en.wikipedia.org/wiki/List_of_countries_by_wealth_per_adult on 2023-08-12
# data based on https://www.credit-suisse.com/media/assets/corporate/docs/about-us/research/publications/global-wealth-databook-2022.pdf
wealth_df = pd.read_csv("wealth_2021.csv")

# get wealth per capita in USD
wealth_df = wealth_df.merge(population_df, on="country_code")
wealth_df["wealth_per_capita"] = wealth_df["wealth"] / wealth_df["population"] * 1000000

fake_user_agent = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0"}
url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
response = requests.get(url, headers=fake_user_agent)
open(data_path + "/country_shapes.zip", "wb").write(response.content)
with zipfile.ZipFile(data_path + "/country_shapes.zip", "r") as zip_file:
    zip_file.extractall(data_path)
shape_df = gpd.read_file(data_path + "/ne_10m_admin_0_countries.shp")


""" save raster data as images """


def save_as_map(_map_array: np.ndarray, image_name: str) -> None:
    map_values = np.concatenate(_map_array[0], axis=0)
    map_values = map_values[map_values > 0]
    # boundaries arbitrary - chosen for nice visual apearance
    bounds = [-200, -1] + list(np.percentile(map_values, [25, 43, 57, 69, 79, 87, 93, 97, 99]))

    newcolors = colormaps["afmhot"](np.linspace(0, 1, 10))
    background_colour = np.array([0, 0, 0, 0])
    newcolors = np.vstack((background_colour, newcolors))
    colour_map = ListedColormap(newcolors)
    norm = colors.BoundaryNorm(bounds, colour_map.N)

    _fig, _ax = plt.subplots(facecolor="#000000")
    _ax.imshow(_map_array[0], norm=norm, cmap=colour_map)
    _ax.axis("off")
    plt.savefig(image_name + ".png", dpi=1000, bbox_inches="tight", pad_inches=0)


map_array = population_density_raster.read()
save_as_map(map_array, "population_density")

# for wealth data, multiply population data by wealth per capita
# this gives a simple estimate of wealth/area in USD/km^2
# replace no data with median

# create "blank canvas" for wealth density
map_array[0][map_array[0] > 0.0] = 0.0
median_wealth = wealth_df["wealth_per_capita"].median()

for code in shape_df["ADM0_A3"]:
    shape = [mapping(shape_df["geometry"][shape_df["ADM0_A3"] == code].iloc[0])]
    country_array, out_transform = mask(population_density_raster, shape, nodata=0)

    country_wealth = wealth_df.loc[wealth_df["country_code"] == code, "wealth_per_capita"]
    if len(country_wealth) == 0:
        country_wealth = median_wealth
    else:
        country_wealth = country_wealth.iloc[0]
    country_array *= country_wealth
    map_array += country_array
    print(code)

save_as_map(map_array, "wealth_density")


""" load both images and generate loop GIF that transitions between them """

font = ImageFont.truetype("Tahoma.ttf", 150)
font_signature = ImageFont.truetype("Tahoma.ttf", 50)
# font colour: median colour of the plot's colour palette
font_colour = (255, 156, 28)
text_position = (250, 1900)
text_position_signature = (4200, 2006) 


def add_signature(_image_draw: ImageDraw.ImageDraw) -> None:
    _image_draw.text(text_position_signature, "Jonas Send", font=font_signature, fill=font_colour)
    
    
def convert_to_rgb_array(_image: PngImageFile) -> np.ndarray:
    return np.array(_image.convert('RGB'))


# create the images with and without text to let text fade in and out
population_image = Image.open('population_density.png')
image_draw = ImageDraw.Draw(population_image)
add_signature(image_draw)
population_image_array = convert_to_rgb_array(population_image)

population_image_with_text = population_image.copy()
image_draw = ImageDraw.Draw(population_image_with_text)
image_draw.text(text_position, "Population", font=font, fill=font_colour)
add_signature(image_draw)
population_image_with_text_array = convert_to_rgb_array(population_image_with_text)

wealth_image = Image.open('wealth_density.png')
image_draw = ImageDraw.Draw(wealth_image)
add_signature(image_draw)
wealth_image_array = convert_to_rgb_array(wealth_image)

wealth_image_with_text = wealth_image.copy()
image_draw = ImageDraw.Draw(wealth_image_with_text)
image_draw.text(text_position, "Wealth", font=font, fill=font_colour)
add_signature(image_draw)
wealth_image_with_text_array = convert_to_rgb_array(wealth_image_with_text)

image_shape = population_image_array.shape

# transform into flat arrays to facilitate easier processing
population_image_flat_array = population_image_array.reshape((-1, 3))
population_image_with_text_flat_array = population_image_with_text_array.reshape((-1, 3))
wealth_image_flat_array = wealth_image_array.reshape((-1, 3))
wealth_image_with_text_flat_array = wealth_image_with_text_array.reshape((-1, 3))

# set up images for animation
fig = plt.figure(dpi=500, figsize=[5.052151238591917, 2.5], frameon=False) # hacky: adjusted to image size
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # no white borders
ax = plt.axes()
ax.axis("off")
image = ax.imshow(np.zeros(image_shape))
fps = 20
# show each image for 2 seconds and use 1 second for transition between the two
frames = fps * 6
frames_long_segment = fps * 2


def animate(_i: int) -> AxesImage:
    """fill image with content for each frame
    hacky if-else clause to determine which images are used for fade-ins and -outs
    """
    if _i <= frames_long_segment:
        set_image(population_image_with_text_flat_array)
    elif _i <= round(fps / 2) + frames_long_segment:
        set_image(fade(population_image_with_text_flat_array, 1 - (2 * (_i - frames_long_segment) / fps)))
    elif _i <= fps + frames_long_segment:
        set_image(fade(wealth_image_with_text_flat_array, (2 * (_i - frames_long_segment) / fps) - 1))
    elif _i <= frames_long_segment * 2 + fps:
        set_image(wealth_image_with_text_flat_array)
    elif _i <= round(fps / 2) + frames_long_segment * 2 + fps:
        set_image(fade(wealth_image_with_text_flat_array, 1 - (2 * (_i - (frames_long_segment * 2 + fps)) / fps)))
    else:
        set_image(fade(population_image_with_text_flat_array, (2 * (_i - (frames_long_segment * 2 + fps)) / fps) - 1))
    return [image]


def set_image(image_flat_array: np.ndarray) -> np.ndarray:
    image.set_array(image_flat_array.reshape(image_shape))


def fade(image_flat_array: np.ndarray, weight_image: float) -> np.ndarray:
    """mix image with text with a 50/50 mix of population and wealth images without text"""
    weight_other_images = (1 - weight_image) / 2
    faded_image = ((image_flat_array * weight_image) + (population_image_flat_array * weight_other_images)
                   + (wealth_image_flat_array* weight_other_images))
    return faded_image.astype(int)


population_wealth_animation = animation.FuncAnimation(fig, animate, frames=frames)
population_wealth_animation.save('population_vs_wealth_density.gif', fps=fps)
