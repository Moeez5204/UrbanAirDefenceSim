import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from shapely.geometry import shape

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300


def download_lahore_data():
    """
    get real buildings from OpenStreetMap for Lahore, Pakistan
    """
    print("--- Downloading Real Lahore Data from OpenStreetMap ---")

    # Lahore metropolitan area bounds (central area)
    north, south, east, west = 31.6000, 31.4800, 74.4500, 74.2800

    try:
        # Download buildings
        print("Downloading building data...")
        buildings_gdf = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})
        print(f"Downloaded {len(buildings_gdf)} buildings")
        return buildings_gdf, (north, south, east, west)

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Falling back to synthetic data...")
        return None, (north, south, east, west)


def process_lahore_buildings(buildings_gdf, bbox):
    """
    Filtered polygons with > 10m^2
    assigned each building to a zone
    calc building heights through zone and area
    """
    print("Processing Lahore building data...")

    north, south, east, west = bbox
    building_footprints = []

    # Define major Lahore zones for height assignment
    def get_lahore_zone(centroid):
        x, y = centroid.x, centroid.y

        zones = {
            'Old City': ((74.30, 31.55), (74.35, 31.59)),
            'Gulberg': ((74.33, 31.50), (74.36, 31.53)),
            'Defence': ((74.36, 31.48), (74.39, 31.51)),
            'Cantt': ((74.34, 31.53), (74.37, 31.56)),
            'DHA': ((74.38, 31.46), (74.42, 31.50)),
            'Model Town': ((74.32, 31.49), (74.35, 31.52)),
        }

        for zone_name, ((x1, y1), (x2, y2)) in zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return 'Other'

    def assign_height_from_zone(zone, area):
        """Assign realistic heights based on Lahore's urban zones."""
        zone_heights = {
            'Old City': (1, 3, 3.0),  # Low-rise, traditional
            'Gulberg': (5, 15, 3.5),  # Commercial high-rise
            'Defence': (4, 12, 3.4),  # Mixed commercial/residential
            'Cantt': (2, 8, 3.2),  # Military/administrative
            'DHA': (2, 6, 3.2),  # Planned residential
            'Model Town': (2, 5, 3.2),  # Planned residential
            'Other': (1, 4, 3.0)  # Default
        }

        min_stories, max_stories, meter_per_story = zone_heights.get(zone, (1, 4, 3.0))

        # Larger buildings tend to have more stories
        area_factor = min(1.0, area / 1000.0)
        story_bonus = int(area_factor * (max_stories - min_stories))

        stories = random.randint(min_stories, min_stories + story_bonus)
        height = stories * meter_per_story

        return height, stories

    processed_count = 0
    for idx, building in buildings_gdf.iterrows():
        try:
            if building.geometry.type == 'Polygon' and building.geometry.area > 10:
                centroid = building.geometry.centroid
                zone = get_lahore_zone(centroid)
                height, stories = assign_height_from_zone(zone, building.geometry.area)

                building_data = {
                    'footprint': building.geometry,
                    'height': height,
                    'stories': stories,
                    'center': (centroid.x, centroid.y),
                    'zone': zone,
                    'area': building.geometry.area
                }
                building_footprints.append(building_data)
                processed_count += 1

        except Exception as e:
            continue

    print(f"Successfully processed {len(building_footprints)} buildings")
    return building_footprints


def create_synthetic_lahore_fallback(bbox):
    """
    generate artifical lahore(not much real data avaliable)
    created grids, reandom buildings , assigned heights based on zones

    """
    print("Creating synthetic Lahore model...")
    north, south, east, west = bbox
    building_footprints = []

    # Create a grid for synthetic buildings
    grid_x = np.linspace(west, east, 25)
    grid_y = np.linspace(south, north, 25)

    def assign_lahore_height(center_x, center_y, area):
        """Assign heights based on synthetic Lahore zones."""
        # Define synthetic zones
        gulberg_center = (74.3430, 31.5189)
        defence_center = (74.3680, 31.4910)
        old_city_center = (74.3230, 31.5750)
        dha_center = (74.3850, 31.4700)

        dist_gulberg = np.sqrt((center_x - gulberg_center[0]) ** 2 + (center_y - gulberg_center[1]) ** 2)
        dist_defence = np.sqrt((center_x - defence_center[0]) ** 2 + (center_y - defence_center[1]) ** 2)
        dist_old_city = np.sqrt((center_x - old_city_center[0]) ** 2 + (center_y - old_city_center[1]) ** 2)
        dist_dha = np.sqrt((center_x - dha_center[0]) ** 2 + (center_y - dha_center[1]) ** 2)

        # Zone influence
        if dist_gulberg < 0.015:
            stories = random.randint(8, 15)
            zone = 'Gulberg'
        elif dist_defence < 0.012:
            stories = random.randint(6, 12)
            zone = 'Defence'
        elif dist_dha < 0.015:
            stories = random.randint(3, 8)
            zone = 'DHA'
        elif dist_old_city < 0.02:
            stories = random.randint(1, 3)
            zone = 'Old City'
        else:
            stories = random.randint(1, 5)
            zone = 'Other'

        height = stories * 3.2
        return height, stories, zone

    # Generate synthetic buildings
    for i in range(len(grid_x) - 1):
        for j in range(len(grid_y) - 1):
            if random.random() < 0.7:  # Building density
                # Create irregular lot
                lot_width = (grid_x[i + 1] - grid_x[i]) * random.uniform(0.8, 1.2)
                lot_height = (grid_y[j + 1] - grid_y[j]) * random.uniform(0.8, 1.2)

                lot = box(
                    grid_x[i] + random.uniform(0, 0.001),
                    grid_y[j] + random.uniform(0, 0.001),
                    grid_x[i] + lot_width,
                    grid_y[j] + lot_height
                )

                buffer_size = random.uniform(0.00005, 0.00015)
                building_poly = lot.buffer(-buffer_size, join_style=2)

                if isinstance(building_poly, Polygon) and building_poly.area > 1e-8:
                    center_x, center_y = building_poly.centroid.x, building_poly.centroid.y
                    height, stories, zone = assign_lahore_height(center_x, center_y, building_poly.area)

                    building_data = {
                        'footprint': building_poly,
                        'height': height,
                        'stories': stories,
                        'center': (center_x, center_y),
                        'zone': zone,
                        'area': building_poly.area
                    }
                    building_footprints.append(building_data)

    print(f"Generated {len(building_footprints)} synthetic buildings")
    return building_footprints


def export_building_data(building_footprints, filename='building_data_3.1.1.json'):
    """
    Export building data for use in Phase 3.1.2 (Alpha Complex Construction)
    """
    print(f"Exporting building data to {filename}...")

    # Convert to serializable format
    export_data = []
    for building in building_footprints:
        building_export = {
            'height': building['height'],
            'stories': building['stories'],
            'center': building['center'],
            'zone': building['zone'],
            'area': building['area'],
            'footprint': building['footprint'].__geo_interface__  # Convert shapely to GeoJSON
        }
        export_data.append(building_export)

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(export_data)} buildings to {filename}")
    return filename


def load_building_data(filename='building_data_3.1.1.json'):
    """
    Load building data from exported JSON file
    """
    print(f"Loading building data from {filename}...")

    with open(filename, 'r') as f:
        data = json.load(f)

    building_footprints = []
    for item in data:
        building = {
            'height': item['height'],
            'stories': item['stories'],
            'center': tuple(item['center']),
            'zone': item['zone'],
            'area': item['area'],
            'footprint': shape(item['footprint'])  # Convert back to shapely
        }
        building_footprints.append(building)

    print(f"Loaded {len(building_footprints)} buildings from {filename}")
    return building_footprints


def run_phase_3_1_1():
    """
    Run Phase 3.1.1: Synthetic City Model Reconstruction
    """
    print("=== Phase 3.1.1: Synthetic City Model Reconstruction ===")

    # Download real Lahore data
    buildings_gdf, bbox = download_lahore_data()

    if buildings_gdf is not None:
        # Process real data
        print("Using real OpenStreetMap data...")
        building_footprints = process_lahore_buildings(buildings_gdf, bbox)
    else:
        # Use synthetic fallback
        print("Using synthetic Lahore data...")
        building_footprints = create_synthetic_lahore_fallback(bbox)

    # Analyze the model
    analyze_building_data(building_footprints)

    # Export the data for 3.1.2
    export_filename = export_building_data(building_footprints)

    print(f"\nPhase 3.1.1 Complete!")
    print(f"Data exported to {export_filename} for Phase 3.1.2")

    return building_footprints, bbox


# Visualization and analysis functions (keep your existing ones)
def analyze_building_data(building_footprints):
    """Provide basic statistics about the generated model."""
    print("\n--- Lahore 3D Model Statistics ---")

    heights = [b['height'] for b in building_footprints]
    stories = [b['stories'] for b in building_footprints]
    zones = [b['zone'] for b in building_footprints]
    areas = [b['area'] for b in building_footprints]

    print(f"Total buildings: {len(building_footprints)}")
    print(f"Height statistics:")
    print(f"  Min: {min(heights):.1f}m, Max: {max(heights):.1f}m, Avg: {np.mean(heights):.1f}m")
    print(f"Area statistics:")
    print(f"  Min: {min(areas):.1f}m², Max: {max(areas):.1f}m², Avg: {np.mean(areas):.1f}m²")

    # Zone distribution
    print(f"\nZone distribution:")
    zone_counts = {}
    for zone in set(zones):
        zone_buildings = [b for b in building_footprints if b['zone'] == zone]
        zone_heights = [b['height'] for b in zone_buildings]
        zone_counts[zone] = len(zone_buildings)
        print(f"  {zone}: {len(zone_buildings)} buildings, Avg height: {np.mean(zone_heights):.1f}m")

    # Building categories
    low_rise = len([h for h in heights if h <= 12.8])
    mid_rise = len([h for h in heights if 12.8 < h <= 25.6])
    high_rise = len([h for h in heights if h > 25.6])

    print(f"\nBuilding categories:")
    print(f"  Low-rise (≤4 stories): {low_rise} buildings ({low_rise / len(heights) * 100:.1f}%)")
    print(f"  Mid-rise (4-8 stories): {mid_rise} buildings ({mid_rise / len(heights) * 100:.1f}%)")
    print(f"  High-rise (>8 stories): {high_rise} buildings ({high_rise / len(heights) * 100:.1f}%)")


if __name__ == "__main__":
    run_phase_3_1_1()