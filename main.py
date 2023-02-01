import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from rasterio.mask import mask

# downloaded from https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_MT_GLOBE_R2019A/ on 2023-01-23
tif_file = rasterio.open('GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif')

# downloaded from https://www.naturalearthdata.com/downloads/ on 2023-01-23
shape_df = gpd.read_file('ne_10m_admin_0_countries.shp')

# retrived from https://en.wikipedia.org/wiki/List_of_countries_by_wealth_per_adult on 2023-01-30
# data based on https://www.credit-suisse.com/media/assets/corporate/docs/about-us/research/publications/global-wealth-databook-2022.pdf
wealth_df = pd.read_csv("mean_wealth_per_country_2021.csv")

old_names = ['Bahamas', 'DR Congo', 'Congo', 'Czech Republic', 'Hong Kong',
             'São Tomé and Príncipe', 'Serbia', 'Tanzania', 'United States']
new_names = ['The Bahamas', 'Democratic Republic of the Congo', 'Republic of the Congo',
             'Czechia', 'Hong Kong S.A.R.', 'São Tomé and Principe', 'Republic of Serbia',
             'United Republic of Tanzania', 'United States of America']
for i in range(len(old_names)):
    wealth_df.loc[wealth_df["country"] == old_names[i], "country"] = new_names[i]

# create "blank canvas"
map_array = tif_file.read()
map_array[0][map_array[0] > 0.0] = 0.0

for country in wealth_df["country"]:
    shape = [mapping(shape_df["geometry"][shape_df["ADMIN"] == country].iloc[0])]
    country_array, out_transform = mask(tif_file, shape, nodata=0)

    # simple estimate of wealth/area in USD/km^2
    country_array *= wealth_df.loc[wealth_df["country"] == country, "mean_wealth"]
    map_array += country_array
    print(country)


#
# TODO: revise and organise imports
#
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors

map_values = np.concatenate(map_array[0], axis=0)
map_values = map_values[map_values > 0]
bounds = [-200, -1] + list(np.percentile(map_values, [25, 43, 57, 69, 79, 87, 93, 97, 99]))

our_cmap = cm.get_cmap('hot_r', 10)
newcolors = our_cmap(np.linspace(0, 1, 10))
background_colour = np.array([0, 0, 0, 0])
newcolors = np.vstack((background_colour, newcolors))
our_cmap = ListedColormap(newcolors)
norm = colors.BoundaryNorm(bounds, our_cmap.N)

#
# TODO: choose background colour
#
fig, ax = plt.subplots(facecolor="#bcedff")
ax.imshow(map_array[0], norm=norm, cmap=our_cmap)
ax.axis('off')
plt.savefig('wealth_density.png', dpi=1000, bbox_inches="tight", pad_inches=0)



