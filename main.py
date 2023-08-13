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
from typing import List


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

colours = colormaps["afmhot"](np.linspace(0, 1, 10))
background_colour = np.array([0, 0, 0, 0])
colours = np.vstack((background_colour, colours))


def save_as_map(_map_array: np.ndarray, image_name: str) -> None:
    map_values = np.concatenate(_map_array[0], axis=0)
    map_values = map_values[map_values > 0]
    # boundaries relatively arbitrary - chosen for nice visual apearance
    bounds = [-200, -1] + list(np.percentile(map_values, [25, 43, 57, 69, 79, 87, 93, 97, 99]))
    
    colour_map = ListedColormap(colours)
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

font = ImageFont.truetype("Avenir Next.ttc", 170, index=10)
font_signature = ImageFont.truetype("Avenir Next.ttc", 70, index=10)
font_colour_index = 8 # pick colour from the map's colour palette - determines length of transition
font_colour_signature = (255, 255, 255)
text_position = (2325, 1900)
text_position_signature = (4060, 2120) 


def add_signature(_image_draw: ImageDraw.ImageDraw) -> None:
    _image_draw.text(text_position_signature, "jonassend.com", font=font_signature, fill=font_colour_signature)
    
    
def convert_to_rgb_array(_image: PngImageFile) -> np.ndarray:
    return np.array(_image.convert('RGB'))


def mix(image1: np.ndarray, image2: np.ndarray, weight1: float) -> np.ndarray:
    mixed_image = image1 * weight1 + image2 * (1- weight1)
    return mixed_image.astype(int)


# create the images with different font colours
population_image = Image.open('population_density.png')
image_draw = ImageDraw.Draw(population_image)
add_signature(image_draw)

image_shape = convert_to_rgb_array(population_image).shape

wealth_image = Image.open('wealth_density.png')
image_draw = ImageDraw.Draw(wealth_image)
add_signature(image_draw)

population_image_array = convert_to_rgb_array(population_image).reshape((-1, 3))
wealth_image_array = convert_to_rgb_array(wealth_image).reshape((-1, 3))


def create_transition_images(image1: PngImageFile, image2: PngImageFile, text: str, ) -> List[np.ndarray]:
    _images = []
    
    for i in range(1, font_colour_index + 1):
        font_colour = tuple(int(x * 255) for x in colours[i][:-1])
        
        image1_with_text = image1.copy()
        image_draw = ImageDraw.Draw(image1_with_text)
        image_draw.text(text_position, text, font=font, fill=font_colour, anchor="mm")
        
        image2_with_text = image2.copy()
        image_draw = ImageDraw.Draw(image2_with_text)
        image_draw.text(text_position, text, font=font, fill=font_colour, anchor="mm")
        
        image1_with_text_array = convert_to_rgb_array(image1_with_text).reshape((-1, 3))
        image2_with_text_array = convert_to_rgb_array(image2_with_text).reshape((-1, 3))
        
        weight_image1 = .5 + (i - 1) * (.5 / (font_colour_index - 1))
        
        _images.append(mix(image1_with_text_array, image2_with_text_array, weight_image1))
        
        # hacky: save feature image
        if i == font_colour_index and text == "Wealth":
            image1_with_text.save("wealth_density_feature_image.png")
    
    return create_transition_sequence(_images)


def create_transition_sequence(_images: List[np.ndarray]) -> List[np.ndarray]:
    transition_sequence = []
    for i in range(len(_images)):
        if i != 0:    
            transition_sequence.append(_images[i])
        if i < len(_images) - 1:
            transition_sequence.append(mix(_images[i], _images[i+1], .5))
    return transition_sequence
    
    
population_transition_images = create_transition_images(population_image, wealth_image, "Population")
wealth_transition_images = create_transition_images(wealth_image, population_image, "Wealth")
    
# define image sequence
FPS = 20
frames_long_segment = int(FPS * 2.5)

images = []
images.extend([population_transition_images[-1]] * frames_long_segment)
images.extend(population_transition_images[::-1])
images.extend(wealth_transition_images)
images.extend([wealth_transition_images[-1]] * frames_long_segment)
images.extend(wealth_transition_images[::-1])
images.extend(population_transition_images)
    
# set up images for animation
fig = plt.figure(dpi=500, figsize=[5.052151238591917, 2.5], frameon=False) # hacky: adjusted to image size
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) # no white borders
ax = plt.axes()
ax.axis("off")
image = ax.imshow(np.zeros(image_shape))


def animate(_i: int) -> AxesImage:
    """fill image with content for each frame"""
    set_image(images[_i])
    return [image]


def set_image(image_flat_array: np.ndarray) -> np.ndarray:
    image.set_array(image_flat_array.reshape(image_shape))


population_wealth_animation = animation.FuncAnimation(fig, animate, frames=len(images))
population_wealth_animation.save('population_vs_wealth_density.gif', fps=FPS)
