import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np

geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Зона застройки"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[37.7173, 55.7558], [38.6179, 56.7565],
                     [39.6185, 55.7559], [39.6179, 55.7552]]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Лесная зона1", "restriction": "no_build"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[37.9500, 55.8060], [37.9605, 55.8055],
                     [37.9710, 55.9060], [38.9205, 55.8055]]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Лесная зона2", "restriction": "no_build"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[37.9190, 55.7570], [37.9195, 56.7575],
                     [37.9500, 55.7570], [37.9595, 55.7565]]
                ]
            }
        },
        # {
        #     "type": "Feature",
        #     "properties": {"name": "Лесная зона2", "restriction": "no_build"},
        #     "geometry": {
        #         "type": "Polygon",
        #         "coordinates": [
        #             [[37.6200, 55.7580], [37.6205, 55.7585],
        #              [37.6210, 55.7580], [37.6205, 55.7575]]
        #         ]
        #     }
        # }
    ]
}



gdf = gpd.GeoDataFrame.from_features(geojson_data)
# print(gdf.iloc[0].geometry)

polys1 = gdf.iloc[0].geometry
# polys2 = gdf.iloc[1].geometry
# polys3 = gdf.iloc[2].geometry



# polys1 = gpd.GeoSeries([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
#                               Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])
#
#
# polys2 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
#                               Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])


df1 = gpd.GeoDataFrame({'geometry': polys1, 'df1': [1, 2]})

# df2 = gpd.GeoDataFrame({'geometry': polys2, 'df2': [1, 2]})
#
# df3 = gpd.GeoDataFrame({'geometry': polys3, 'df3': [1, 2, 3]})

ax = df1.plot(color='red')

# df2.plot(ax=ax, color='green', alpha=0.5)
#
# df3.plot(ax=ax, color='green', alpha=0.5)

plt.savefig('color_prev.png')

from PIL import Image

img = Image.open('color_prev.png')

new_size = (img.width * 2, img.height * 2)
img = img.resize(new_size, Image.Resampling.LANCZOS)
img.save('resized_image.png')

pixels = img.load()
pixels[350, 350]  # цвет пикселя [1](https://www.CyberForum.ru/python-graphics/thread2634435.html)
print(pixels[320, 240])
width, height = img.size
print(f"Размеры изображения: {width}x{height} пикселей")
# plt.show()

# scatter = plt.scatter([38.5], [56])
# print(scatter)
# print(scatter.get_facecolors())
#
# # Ваш массив
# array = np.array(scatter.get_facecolors())
#
# # Получаем RGB значения (игнорируем последний элемент - это альфа-канал)
# rgb = array[0][:3]
#
#
# # Умножаем на 255 для получения целых чисел от 0 до 255
# rgb_int = (rgb * 255).astype(int)
#
# print(f"RGB значения: ({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})")
#
# # Для проверки - создадим точку с этим цветом
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(2, 2))
# plt.scatter([0], [0], c=array[0][:3], s=500)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.gca().axis('off')
# plt.savefig('color_preview.png')
#
