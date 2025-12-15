# city_generator.py - City layout and building generation

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Building3D:
    """3D building representation"""
    x: float
    y: float
    width: float
    depth: float
    height: float
    color: Tuple[int, int, int]
    zone: str
    is_special_asset: bool = False
    asset_name: str = ""
    asset_type: str = ""


@dataclass
class Sector:
    """Defense sector"""
    name: str
    center_x: float
    center_y: float
    radius: float
    color: Tuple[int, int, int]
    priority: float
    resources: int = 0


class CityGenerator:
    """Generates the city layout with larger, more spaced buildings and 3x density"""

    def __init__(self):
        # Increase city size to accommodate more buildings
        self.city_width = 3000
        self.city_height = 3000
        self.center = (self.city_width / 2, self.city_height / 2)

        # Colors
        self.building_colors = {
            'Old City': (190, 140, 90),
            'Gulberg': (110, 170, 220),
            'Defence': (170, 200, 140),
            'Cantt': (210, 170, 120),
            'DHA': (140, 180, 210),
            'Model Town': (200, 180, 160),
            'Other': (140, 140, 140)
        }

        # Define city zones with updated parameters for larger, more spaced buildings
        self.zones = {
            'Walled_City': {
                'center': (self.center[0] - 600, self.center[1] - 300),
                'radius': 450,
                'density': 0.85,
                'building_size_range': (60, 160),
                'height_range': (15, 45),
                'min_spacing': 40,
                'color': self.building_colors['Old City']
            },
            'Central_Lahore': {
                'center': self.center,
                'radius': 600,
                'density': 0.75,
                'building_size_range': (80, 200),
                'height_range': (30, 150),
                'min_spacing': 50,
                'color': self.building_colors['Defence']
            },
            'Gulberg': {
                'center': (self.center[0] + 450, self.center[1] + 150),
                'radius': 525,
                'density': 0.70,
                'building_size_range': (100, 240),
                'height_range': (40, 180),
                'min_spacing': 60,
                'color': self.building_colors['Gulberg']
            },
            'Cantonment': {
                'center': (self.center[0] - 300, self.center[1] + 600),
                'radius': 480,
                'density': 0.60,
                'building_size_range': (120, 200),
                'height_range': (20, 90),
                'min_spacing': 70,
                'color': self.building_colors['Cantt']
            },
            'Other': {
                'center': (self.center[0] + 750, self.center[1] - 450),
                'radius': 750,
                'density': 0.50,
                'building_size_range': (60, 160),
                'height_range': (10, 75),
                'min_spacing': 80,
                'color': self.building_colors['Other']
            }
        }

        # Track placed buildings for spacing checks
        self.placed_buildings = []

    def generate_city(self, special_assets):
        """Generate complete city with buildings and sectors"""
        print("Generating city layout with larger, more spaced buildings...")

        # Reset placed buildings
        self.placed_buildings = []

        # Create buildings with 3x density
        buildings = self.create_buildings(special_assets)

        # Create sectors
        sectors = self.create_sectors(special_assets)

        return {
            'width': self.city_width,
            'height': self.city_height,
            'center': self.center,
            'buildings': buildings,
            'sectors': sectors,
            'zones': self.zones
        }

    def create_buildings(self, special_assets):
        """Create buildings for each zone with 3x density and better spacing"""
        buildings = []

        for zone_name, zone_data in self.zones.items():
            print(f"  Generating buildings for {zone_name}...")
            center_x, center_y = zone_data['center']
            radius = zone_data['radius']
            density = zone_data['density']
            size_min, size_max = zone_data['building_size_range']
            height_min, height_max = zone_data['height_range']
            color = zone_data['color']
            min_spacing = zone_data['min_spacing']

            # 3x more buildings
            num_buildings = int(density * 240)
            buildings_created = 0
            attempts_per_building = 0

            for i in range(num_buildings):
                building = self.create_building_with_spacing(
                    zone_name, zone_data, special_assets,
                    center_x, center_y, radius,
                    size_min, size_max, height_min, height_max,
                    color, min_spacing
                )

                if building:
                    buildings.append(building)
                    self.placed_buildings.append((building.x, building.y, building.width, building.depth))
                    buildings_created += 1
                    attempts_per_building = 0
                else:
                    attempts_per_building += 1
                    if attempts_per_building > 50:
                        break

            print(f"    Created {buildings_created} buildings in {zone_name}")

        print(f"✓ Created {len(buildings)} buildings total (3x density with spacing)")
        return buildings

    def create_building_with_spacing(self, zone_name, zone_data, special_assets,
                                     center_x, center_y, radius,
                                     size_min, size_max, height_min, height_max,
                                     color, min_spacing):
        """Create a single building with proper spacing from others"""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            # Random position in zone with more spread
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius * 0.9)
            x = center_x + math.cos(angle) * distance
            y = center_y + math.sin(angle) * distance

            # Building dimensions - doubled size
            width = random.uniform(size_min, size_max)
            depth = random.uniform(size_min * 0.7, size_max * 0.7)
            height = random.uniform(height_min, height_max)

            # Add variety: some buildings are extra large
            if random.random() < 0.1:
                width *= 1.5
                depth *= 1.5
                height *= 1.2

            # Check if position is valid with spacing
            if not self.is_position_valid_with_spacing(x, y, width, depth, min_spacing):
                attempts += 1
                continue

            # Check for special asset
            is_special, asset_name, asset_type = self.check_special_asset(
                x, y, special_assets, height
            )

            # Create the building
            return Building3D(
                x=x, y=y, width=width, depth=depth,
                height=height,
                color=color,
                zone=zone_name,
                is_special_asset=is_special,
                asset_name=asset_name,
                asset_type=asset_type
            )

        return None

    def is_position_valid_with_spacing(self, x, y, width, depth, min_spacing):
        """Check if position is valid with proper spacing from other buildings"""
        # Check distance from other buildings
        for bx, by, bwidth, bdepth in self.placed_buildings:
            distance = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
            min_distance = (max(width, depth) + max(bwidth, bdepth)) / 2 + min_spacing

            if distance < min_distance:
                return False

        return True

    def check_special_asset(self, x, y, special_assets, default_height):
        """Check if position contains a special asset"""
        for asset in special_assets:
            asset_pos = asset.get('position', {'x': 0, 'y': 0})
            ax, ay = asset_pos.get('x', 0), asset_pos.get('y', 0)
            if abs(x - ax) < 100 and abs(y - ay) < 100:
                return True, asset['name'], asset.get('type', 'unknown')
        return False, "", ""

    def create_sectors(self, special_assets):
        """Create defense sectors"""
        sectors = []

        for zone_name, zone_data in self.zones.items():
            center_x, center_y = zone_data['center']
            radius = zone_data['radius']

            # Assign colors and priorities
            if zone_name == 'Walled_City':
                color = (255, 100, 100)
                priority = 0.8
            elif zone_name == 'Central_Lahore':
                color = (100, 150, 255)
                priority = 0.9
            elif zone_name == 'Gulberg':
                color = (100, 255, 150)
                priority = 0.7
            elif zone_name == 'Cantonment':
                color = (255, 200, 100)
                priority = 0.85
            else:
                color = (180, 100, 255)
                priority = 0.5

            # Calculate resources from special assets in this sector
            resources = sum(
                asset.get('resources_allocated', 0)
                for asset in special_assets
                if asset.get('sector') == zone_name
            )

            sectors.append(Sector(
                name=zone_name,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                color=color,
                priority=priority,
                resources=resources
            ))

        print(f"✓ Created {len(sectors)} sectors")
        return sectors