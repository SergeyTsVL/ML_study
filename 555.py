import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString
import json
from typing import List, Dict, Union


class TerritoryPlanner:
    def __init__(self, geojson_data, density_limit, min_distance):
        """
        Инициализация планировщика территории

        Args:
            geojson_data: GeoJSON данные с границами участка и ограничениями
            density_limit: Максимальная допустимая плотность застройки (0-1)
            min_distance: Минимальное расстояние между объектами в метрах
        """
        self.gdf = gpd.GeoDataFrame.from_features(geojson_data)
        self.density_limit = density_limit
        self.min_distance = min_distance
        self.buildable_area = None
        self.buildings = []

    def calculate_buildable_area(self):
        """Расчёт площади, доступной для застройки"""
        # print(1111111111111111)
        total_area = self.gdf.iloc[0].geometry.area
        print(22222222222222222, self.gdf.iloc[0].geometry)
        # print(self.gdf.describe())
        # print(self.gdf['name'])
        # print(self.gdf['geometry'])

        restricted_mask = self.gdf['properties'].apply(
            lambda x: 'no_build' in str(x) if isinstance(x, dict) else False
        )
        # restricted_mask = self.gdf['properties'].apply(
        #     lambda x: not x.get('restricted', False)
        # )
        # restricted_mask = self.gdf.iloc[1].geometry.area
        # print(restricted_mask)
        # print(333333333333333333)
        restricted_area = float(self.gdf[restricted_mask].geometry.area.sum())
        # print(444444444444444)
        # print(restricted_area)
        self.buildable_area = total_area - restricted_area
        # self.buildable_area = total_area - restricted_mask
        # print(5555555555555555)
        # print(self.gdf)
        # print(self.buildable_area)
        return self.buildable_area






    def generate_building(self, max_size):
        """Генерация случайного здания с учётом ограничений"""
        while True:
            # print('generate_building(self, max_size)')
            width = np.random.uniform(10, max_size)
            height = np.random.uniform(10, max_size)

            min_x = self.gdf.iloc[0].geometry.bounds[0]
            max_x = self.gdf.iloc[0].geometry.bounds[2]
            min_y = self.gdf.iloc[0].geometry.bounds[1]
            max_y = self.gdf.iloc[0].geometry.bounds[3]

            x = np.random.uniform(min_x, max_x - width)
            y = np.random.uniform(min_y, max_y - height)

            building = Polygon([
                (x, y), (x + width, y),
                (x + width, y + height), (x, y + height), (x, y)
            ])
            if building.geometry is None:
                # Исправляем геометрию, если она None
                print('building.geometry == None')
                # building.geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            # print(building)
            try:
                print((self.gdf.iloc[0].geometry.contains(building)))
                print(self._intersects_with_restricted(building))
                print(self._intersects_with_existing(building))
                print('++++++++++++++++++++++++')
                # print("Геометрия здания:", building.geometry)
                # print("Геометрия существующих зданий:", [b.geometry for b in self.buildings])
                # print("Область поиска:", self.buildings.total_bounds)
                if (self.gdf.iloc[0].geometry.contains(building) and
                        not self._intersects_with_restricted(building) and
                        not self._intersects_with_existing(building)):
                    return building
            except:
                print("Не выполнено условие пересечения объектов")
                break

    def _intersects_with_restricted(self, building):
        """Проверка пересечения с запрещёнными зонами"""
        restricted_mask = self.gdf['properties'].apply(
            lambda x: 'no_build' in str(x) if isinstance(x, dict) else False
        )
        return self.gdf[restricted_mask].geometry.intersects(building).any()

    def _intersects_with_existing(self, building):
        """Проверка пересечения с существующими зданиями"""
        return any(b.geometry.intersects(building) for b in self.buildings)

    def plan_territory(self, max_buildings):
        """Планирование застройки территории"""
        print("Начало планирования застройки...")

        # Проверка входных данных
        if not isinstance(max_buildings, int) or max_buildings <= 0:
            raise ValueError("max_buildings должно быть положительным целым числом")

        # Расчёт доступной площади
        print("Расчёт доступной площади...")
        self.calculate_buildable_area()

        # Проверка площади
        if self.buildable_area <= 0:
            raise ValueError(f"Недостаточно площади для застройки. Доступно: {self.buildable_area}")

        # Настройка максимального размера здания
        max_size = float(min(self.buildable_area / 1000, 100))
        print(f"Максимальный размер здания: {max_size} кв.м.")

        # Генерация зданий
        print("Генерация зданий...")
        while len(self.buildings) < max_buildings:
            # print(99999999999999999)
            # print(len(self.buildings), max_buildings)
            try:
                building = self.generate_building(max_size)
                # print(building)
                # print(building.area)
                # print(6666666666666)
                if building.area <= self.buildable_area * self.density_limit:
                    # print(building.area)
                    self.buildings.append({
                        'type': 'residential',
                        'geometry': building
                    })
                    # print(888888888888888)
                    self.buildable_area -= building.area
                    print(f"Построено зданий: {len(self.buildings)}/{max_buildings}")
                else:
                    print("Здание слишком большое для оставшейся площади")
                    break
            except Exception as e:
                print(f"Ошибка при генерации здания: {str(e)}")
                break

        # Создание GeoJSON результата
        print("Создание GeoJSON результата...")
        return self._create_output_geojson()

    def _create_output_geojson(self):
        """Создание выходного GeoJSON файла"""
        features = []
        for building in self.buildings:
            features.append({
                'type': 'Feature',
                'properties': building['type'],
                'geometry': json.loads(json.dumps(building['geometry'].__geo_interface__))
            })
        return {
            'type': 'FeatureCollection',
            'features': features
        }

    # def visualize(self, output_file='territory_plan.png'):
    #     """Визуализация плана застройки"""
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #
    #     self.gdf.plot(ax=ax, color='#D3D3D3', edgecolor='black')
    #
    #     restricted_mask = self.gdf['properties'].apply(
    #         lambda x: 'no_build' in str(x) if isinstance(x, dict) else False
    #     )
    #     self.gdf[restricted_mask].plot(ax=ax, color='red', alpha=0.3)
    #
    #     for building in self.buildings:
    #         geom = building['geometry']
    #         if geom.geom_type == 'Polygon':
    #             plt.fill(*geom.exterior.xy, color='blue', alpha=0.5)
    #             plt.plot(*geom.exterior.xy, color='black')
    #
    #     plt.title('План застройки территории')
    #     plt.axis('equal')
    #     plt.savefig(output_file)
    #
    #     plt.close()

    def visualize(self, output_file='territory_plan.png'):
        if 'properties' not in self.gdf.columns:
            self.gdf['properties'] = None
        # print(self.gdf)

        self.gdf['properties'] = self.gdf['properties'].apply(
            lambda x: {'restricted': False} if not isinstance(x, dict) else x
        )
        # print(self.gdf)
        restricted_mask = self.gdf['properties'].apply(
            lambda x: x.get('restricted', False)
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        self.gdf.plot(ax=ax, color='#D3D3D3', edgecolor='black')
        plt.savefig(output_file)
        plt.close()

    # В классе TerritoryPlanner:
    # def debug_restricted_areas(self):
    #     """Отладка ограничений"""
    #     print("Анализ ограничений:")
    #     for _, row in self.gdf.iterrows():
    #         props = row.get('properties', {})
    #         print(f"Объект: {props.get('name')}")
    #         print(f"Тип: {row.geometry.geom_type}")
    #         print(f"Ограничения: {props}")
    #         print("-" * 50)




# Пример использования
def validate_geojson_data(geojson_data):
    """Проверка корректности GeoJSON данных"""
    if not isinstance(geojson_data, dict):
        return False

    if geojson_data.get('type') != 'FeatureCollection':
        return False
    # print(geojson_data)
    if not isinstance(geojson_data.get('features'), list):
        return False

    return True








# Пример входных данных
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
            "properties": {"name": "Лесная зона", "restriction": "no_build"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[37.9500, 55.8060], [37.9605, 55.8055],
                     [37.9710, 55.9060], [38.9205, 55.8055]]
                ]
            }
        },
        # {
        #     "type": "Feature",
        #     "properties": {"name": "Лесная зона1", "restriction": "no_build"},
        #     "geometry": {
        #         "type": "Polygon",
        #         "coordinates": [
        #             [[37.6190, 55.7570], [37.6195, 55.7575],
        #              [37.6200, 55.7570], [37.6195, 55.7565]]
        #         ]
        #     }
        # },
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

    # Использование:
# planner = TerritoryPlanner(geojson_data, density_limit=0.5, min_distance=1)
# planner.debug_restricted_areas()

# planner = TerritoryPlanner(geojson_data, density_limit=0.5, min_distance=10)
# obj = TerritoryPlanner(
#     geojson_data=geojson_data,
#     density_limit=10,
#     min_distance=5
# )  # Создаем экземпляр класса
# obj.visualize()

# Проверка данных
if not validate_geojson_data(geojson_data):
    print("Ошибка: Некорректный формат GeoJSON данных")
else:
    # Создание планировщика
    planner = TerritoryPlanner(geojson_data, density_limit=0.5, min_distance=1)
    # print(planner.visualize())
    # planner.visualize()
    planner.visualize()

    try:
        # Планирование застройки
        print("Начало планирования застройки...")
        result = planner.plan_territory(max_buildings=10)
        print("Планирование завершено успешно")

        # Проверка результата
        if not isinstance(result, dict):
            print("Ошибка: Результат не является словарём")
            raise ValueError("Некорректный формат результата")

        # Сохранение результатов
        print("Сохранение GeoJSON файла...")
        with open('territory_plan.geojson', 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("GeoJSON файл сохранён успешно")

        # Визуализация
        print("Создание визуализации...")
        planner.visualize('territory_plan.png')
        print("Визуализация сохранена успешно")

    except Exception as e:
        print("\n=== Детальная информация об ошибке ===")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение об ошибке: {str(e)}")

        # Дополнительная информация о состоянии планировщика
        print("\n=== Состояние планировщика ===")
        print(f"Количество зданий: {len(planner.buildings)}")
        print(f"Доступная площадь: {planner.buildable_area}")

        # Проверка GeoJSON данных
        print("\n=== Проверка GeoJSON данных ===")
        print(f"Количество объектов: {len(planner.gdf)}")
        print("Типы объектов:")

        for _, row in planner.gdf.iterrows():
            print(f"- {row.geometry.geom_type}: {row.get('properties', {}).get('name', 'Без имени')}")



