# lahore_model_builder.py
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import colorsys
import math


@dataclass
class Building3D:
    """3D building representation for PyGame"""
    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    height: float
    center: Tuple[float, float]
    zone: str
    color: Tuple[float, float, float]
    footprint: List[Tuple[float, float]]


@dataclass
class Canyon3D:
    """3D urban canyon representation"""
    centerline: List[Tuple[float, float, float]]
    width: float
    depth: float
    color: Tuple[float, float, float]
    name: str
    threat_level: str


@dataclass
class Threat3D:
    """3D threat representation"""
    position: Tuple[float, float, float]
    target_id: str
    threat_level: str
    color: Tuple[float, float, float]
    size: float
    velocity: Tuple[float, float, float]


@dataclass
class Asset3D:
    """3D defended asset representation"""
    position: Tuple[float, float, float]
    asset_id: str
    asset_name: str
    priority: float
    color: Tuple[float, float, float]
    size: float


@dataclass
class Lahore3DModel:
    """Complete 3D model of Lahore"""
    buildings: List[Building3D] = field(default_factory=list)
    canyons: List[Canyon3D] = field(default_factory=list)
    threats: List[Threat3D] = field(default_factory=list)
    assets: List[Asset3D] = field(default_factory=list)

    # Map boundaries
    min_x: float = 0
    max_x: float = 0
    min_y: float = 0
    max_y: float = 0
    min_z: float = 0
    max_z: float = 0

    # Statistics
    stats: Dict = field(default_factory=dict)


class LahoreModelBuilder:
    """Builds 3D models from JSON data - UPDATED FOR GLOBAL COORDINATE PROCESSING"""

    def __init__(self):
        self.model = Lahore3DModel()
        # Lahore bounds for normalization
        self.lon_min = 74.28
        self.lon_max = 74.45
        self.lat_min = 31.48
        self.lat_max = 31.60
        self.map_scale = 600  # Scale factor for visualization

        # Global canyon processing containers
        self.raw_canyon_data = []
        self.all_utm_points = []
        self.canyon_utm_bounds_calculated = False

    def normalize_coordinates(self, lon: float, lat: float) -> Tuple[float, float]:
        """Normalize lon/lat to [-1, 1] range and scale for visualization"""
        # Normalize to 0-1 range
        norm_x = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        norm_y = (lat - self.lat_min) / (self.lat_max - self.lat_min)

        # Center around 0 and scale
        x = (norm_x - 0.5) * 2 * self.map_scale
        y = (norm_y - 0.5) * 2 * self.map_scale

        return x, y

    def load_building_data(self, filename='building_data_3.1.1.json'):
        """Load and convert building data to 3D models"""
        print(f"Loading building data from {filename}...")

        try:
            with open(filename, 'r') as f:
                building_data = json.load(f)

            print(f"Loaded {len(building_data)} buildings")

            # Zone color mapping
            zone_colors = {
                'Old City': (0.6, 0.4, 0.2),  # Brown
                'Gulberg': (0.8, 0.2, 0.2),  # Red
                'Defence': (0.2, 0.2, 0.8),  # Blue
                'Cantt': (0.2, 0.8, 0.2),  # Green
                'DHA': (0.8, 0.8, 0.2),  # Yellow
                'Model Town': (0.8, 0.2, 0.8),  # Purple
                'Other': (0.5, 0.5, 0.5)  # Gray
            }

            building_count = 0
            # Limit buildings for performance but ensure good coverage
            max_buildings = 300
            step = max(1, len(building_data) // max_buildings)

            for i in range(0, len(building_data), step):
                if building_count >= max_buildings:
                    break

                building_3d = self._create_building_3d(building_data[i], zone_colors)
                if building_3d:
                    self.model.buildings.append(building_3d)
                    building_count += 1

            print(f"Created {building_count} 3D buildings")
            return True

        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading building data: {e}")
            return False

    def _create_building_3d(self, building_data: Dict, zone_colors: Dict) -> Building3D:
        """Convert building data to 3D representation with proper scaling"""
        try:
            # Extract building properties
            height = building_data['height']
            center = building_data['center']  # (lon, lat)
            zone = building_data['zone']
            area = building_data['area']

            # Convert to normalized coordinates
            lon, lat = center
            x, y = self.normalize_coordinates(lon, lat)
            z = 0

            # Height scaling - make buildings visible relative to map size
            # Buildings are typically 5-50m, scale for visibility
            height_scaled = height * 3  # Make buildings appear taller

            # Update bounds
            self.model.min_x = min(self.model.min_x, x)
            self.model.max_x = max(self.model.max_x, x)
            self.model.min_y = min(self.model.min_y, y)
            self.model.max_y = max(self.model.max_y, y)
            self.model.max_z = max(self.model.max_z, height_scaled)

            # Create building footprint - scale based on area
            # Make buildings larger for visibility
            base_size = math.sqrt(area) * 0.15  # Scale down footprint
            min_size = 5.0  # Minimum building size
            max_size = 25.0  # Maximum building size
            size = max(min_size, min(base_size, max_size))

            # Create 8 vertices for a box
            vertices = [
                (x - size, y - size, z),  # 0: bottom front left
                (x + size, y - size, z),  # 1: bottom front right
                (x + size, y + size, z),  # 2: bottom back right
                (x - size, y + size, z),  # 3: bottom back left
                (x - size, y - size, z + height_scaled),  # 4: top front left
                (x + size, y - size, z + height_scaled),  # 5: top front right
                (x + size, y + size, z + height_scaled),  # 6: top back right
                (x - size, y + size, z + height_scaled)  # 7: top back left
            ]

            # Create faces (6 sides of cube)
            faces = [
                [0, 1, 2, 3],  # Bottom
                [4, 5, 6, 7],  # Top
                [0, 1, 5, 4],  # Front
                [2, 3, 7, 6],  # Back
                [0, 3, 7, 4],  # Left
                [1, 2, 6, 5]  # Right
            ]

            # Create footprint for ground visualization
            footprint = [
                (x - size, y - size),
                (x + size, y - size),
                (x + size, y + size),
                (x - size, y + size)
            ]

            # Get zone color
            color = zone_colors.get(zone, (0.5, 0.5, 0.5))

            # Adjust color based on height (taller = darker/more intense)
            height_factor = min(1.0, height_scaled / 150)
            if zone == 'Gulberg' or zone == 'Defence':
                # Commercial areas - brighter
                color = tuple(min(1.0, c * (0.8 + height_factor * 0.4)) for c in color)
            else:
                # Other areas - slightly darker
                color = tuple(c * (0.7 + height_factor * 0.3) for c in color)

            return Building3D(
                vertices=vertices,
                faces=faces,
                height=height_scaled,
                center=(x, y),
                zone=zone,
                color=color,
                footprint=footprint
            )

        except Exception as e:
            print(f"Error creating building 3D: {e}")
            return None

    def load_strategic_data(self, filename='strategic_features_3.1.4.json'):
        """Load and convert strategic features to 3D models - FIXED to avoid duplication"""
        print(f"Loading strategic data from {filename}...")

        try:
            # Clear existing canyons before loading
            self.model.canyons = []

            with open(filename, 'r') as f:
                strategic_data = json.load(f)

            # Reset global containers
            self.raw_canyon_data = []
            self.all_utm_points = []
            self.canyon_utm_bounds_calculated = False

            # Load canyons
            if 'strategic_features' in strategic_data:
                features = strategic_data['strategic_features']

                # First pass: collect all canyon data
                canyon_count = 0
                for i, canyon in enumerate(features.get('canyons', [])):
                    canyon_count += 1
                    self._create_canyon_3d(canyon, i)  # This just collects data

                print(f"Collected data for {canyon_count} canyons")

                # Check if we already have canyons in the model
                if self.model.canyons:
                    print(f"Warning: Already have {len(self.model.canyons)} canyons. Skipping processing.")
                    return True

                # Process all canyons together
                self.process_all_canyons_together()

                print(f"Successfully created {len(self.model.canyons)} 3D canyons with proper spacing")

            return True

        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading strategic data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_canyon_3d(self, canyon_data: Dict, index: int) -> Canyon3D:
        """Collect canyon data for later global processing - DON'T create object here"""
        try:
            persistence = canyon_data.get('persistence', 100)
            threat_level = canyon_data.get('threat_level', 'medium')
            centerline = canyon_data.get('centerline', [])

            if not centerline:
                return None

            # Collect UTM points for global bounds calculation
            for point in centerline:
                if len(point) >= 2:
                    self.all_utm_points.append((point[0], point[1]))

            # Store raw canyon data for later processing
            self.raw_canyon_data.append({
                'index': index,
                'centerline': centerline,
                'persistence': persistence,
                'threat_level': threat_level
            })

            # Return None for now - we'll process all canyons together later
            return None

        except Exception as e:
            print(f"Error in canyon preprocessing: {e}")
            return None

    def load_threat_data(self, filename='lahore_3d_data.json'):
        """Load and convert threat data to 3D models"""
        print(f"Loading threat data from {filename}...")

        try:
            with open(filename, 'r') as f:
                threat_data = json.load(f)

            # Load threats
            threat_count = 0
            for threat in threat_data.get('threat_data', []):
                if threat_count >= 30:  # Limit for performance
                    break

                threat_3d = self._create_threat_3d(threat)
                if threat_3d:
                    self.model.threats.append(threat_3d)
                    threat_count += 1

            # Load assets
            asset_count = 0
            for asset in threat_data.get('defended_assets', []):
                if asset_count >= 15:  # Limit for performance
                    break

                asset_3d = self._create_asset_3d(asset)
                if asset_3d:
                    self.model.assets.append(asset_3d)
                    asset_count += 1

            print(f"Created {threat_count} threats and {asset_count} assets")
            return True

        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading threat data: {e}")
            return False

    def _create_threat_3d(self, threat_data: Dict) -> Threat3D:
        """Convert threat data to 3D representation with scaled coordinates"""
        try:
            position_data = threat_data.get('target_position', [74.35, 31.55, 100])

            # Generate random position within city bounds if not specified
            if len(position_data) < 3 or position_data[0] == 0:
                # Random position within Lahore bounds
                lon = np.random.uniform(self.lon_min + 0.02, self.lon_max - 0.02)
                lat = np.random.uniform(self.lat_min + 0.02, self.lat_max - 0.02)
                alt = np.random.uniform(80, 200)
            else:
                lon, lat, alt = position_data

            x, y = self.normalize_coordinates(lon, lat)
            z = alt * 0.3  # Scale altitude down for visualization

            # Ensure position is within reasonable bounds
            bounds = 550  # Should match your map_scale
            x = max(-bounds, min(x, bounds))
            y = max(-bounds, min(y, bounds))
            z = max(50, min(z, 300))

            threat_level = threat_data.get('threat_level', 'medium')

            # Color and size based on threat level
            if threat_level == 'CRITICAL':
                color = (1.0, 0.0, 0.0)  # Red
                size = 12.0
            elif threat_level == 'HIGH':
                color = (1.0, 0.5, 0.0)  # Orange
                size = 10.0
            elif threat_level == 'MEDIUM':
                color = (1.0, 1.0, 0.0)  # Yellow
                size = 8.0
            else:
                color = (0.5, 0.5, 0.5)  # Gray
                size = 6.0

            # Random velocity (slower for more controlled movement)
            velocity = (
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.05, 0.05)
            )

            return Threat3D(
                position=(x, y, z),
                target_id=threat_data.get('target_id', f'Threat_{np.random.randint(1000)}'),
                threat_level=threat_level,
                color=color,
                size=size,
                velocity=velocity
            )

        except Exception as e:
            print(f"Error creating threat 3D: {e}")
            return None

    def _create_asset_3d(self, asset_data: Dict) -> Asset3D:
        """Convert asset data to 3D representation with scaled coordinates"""
        try:
            position_data = asset_data.get('position', [74.35, 31.55, 0])
            if len(position_data) < 2:
                return None

            lon, lat = position_data[0], position_data[1]
            x, y = self.normalize_coordinates(lon, lat)
            z = position_data[2] if len(position_data) > 2 else 0

            priority = asset_data.get('priority', 0.5)

            # Color based on priority
            if priority > 0.9:
                color = (0.0, 1.0, 0.0)  # Green - high priority
            elif priority > 0.7:
                color = (0.0, 0.7, 0.7)  # Cyan - medium-high
            else:
                color = (0.0, 0.0, 1.0)  # Blue - medium

            # Size based on priority
            size = 8 + priority * 12

            return Asset3D(
                position=(x, y, z),
                asset_id=asset_data.get('asset_id', 'Unknown'),
                asset_name=asset_data.get('asset_name', 'Unknown'),
                priority=priority,
                color=color,
                size=size
            )

        except Exception as e:
            print(f"Error creating asset 3D: {e}")
            return None

    def create_synthetic_data(self):
        """Create synthetic data for visualization when JSON files are missing"""
        print("Creating synthetic data for visualization...")

        # Create synthetic buildings
        zone_colors = {
            'Old City': (0.6, 0.4, 0.2),
            'Gulberg': (0.8, 0.2, 0.2),
            'Defence': (0.2, 0.2, 0.8),
            'Cantt': (0.2, 0.8, 0.2),
            'DHA': (0.8, 0.8, 0.2),
            'Model Town': (0.8, 0.2, 0.8),
            'Other': (0.5, 0.5, 0.5)
        }

        zones = list(zone_colors.keys())

        for i in range(150):  # Create 150 synthetic buildings
            zone = np.random.choice(zones)

            # Random position within scaled bounds
            x = np.random.uniform(-self.map_scale * 0.8, self.map_scale * 0.8)
            y = np.random.uniform(-self.map_scale * 0.8, self.map_scale * 0.8)

            # Zone-based height
            if zone == 'Gulberg':
                height = np.random.uniform(30, 80)
            elif zone == 'Defence':
                height = np.random.uniform(20, 60)
            elif zone == 'Old City':
                height = np.random.uniform(5, 25)
            else:
                height = np.random.uniform(10, 40)

            height_scaled = height * 3

            size = np.random.uniform(8, 25)

            vertices = [
                (x - size, y - size, 0),
                (x + size, y - size, 0),
                (x + size, y + size, 0),
                (x - size, y + size, 0),
                (x - size, y - size, height_scaled),
                (x + size, y - size, height_scaled),
                (x + size, y + size, height_scaled),
                (x - size, y + size, height_scaled)
            ]

            faces = [
                [0, 1, 2, 3], [4, 5, 6, 7],
                [0, 1, 5, 4], [2, 3, 7, 6],
                [0, 3, 7, 4], [1, 2, 6, 5]
            ]

            building = Building3D(
                vertices=vertices,
                faces=faces,
                height=height_scaled,
                center=(x, y),
                zone=zone,
                color=zone_colors[zone],
                footprint=[(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]
            )

            self.model.buildings.append(building)

            # Update bounds
            self.model.min_x = min(self.model.min_x, x - size)
            self.model.max_x = max(self.model.max_x, x + size)
            self.model.min_y = min(self.model.min_y, y - size)
            self.model.max_y = max(self.model.max_y, y + size)
            self.model.max_z = max(self.model.max_z, height_scaled)

        print(f"Created {len(self.model.buildings)} synthetic buildings")

    def calculate_statistics(self):
        """Calculate model statistics"""
        self.model.stats = {
            'total_buildings': len(self.model.buildings),
            'total_canyons': len(self.model.canyons),
            'total_threats': len(self.model.threats),
            'total_assets': len(self.model.assets),
            'map_bounds': {
                'x_min': self.model.min_x,
                'x_max': self.model.max_x,
                'y_min': self.model.min_y,
                'y_max': self.model.max_y,
                'z_max': self.model.max_z
            },
            'avg_building_height': np.mean([b.height for b in self.model.buildings]) if self.model.buildings else 0,
            'map_scale': self.map_scale
        }

        print("\n=== 3D Model Statistics ===")
        print(f"Buildings: {self.model.stats['total_buildings']}")
        print(f"Canyons: {self.model.stats['total_canyons']}")
        print(f"Threats: {self.model.stats['total_threats']}")
        print(f"Assets: {self.model.stats['total_assets']}")
        print(f"Avg Building Height: {self.model.stats['avg_building_height']:.1f}m")
        print(f"Map Bounds: X({self.model.min_x:.0f} to {self.model.max_x:.0f}) "
              f"Y({self.model.min_y:.0f} to {self.model.max_y:.0f})")
        print(f"Map Scale: {self.map_scale}")

    def build_complete_model(self) -> Lahore3DModel:
        """Build complete 3D model from all available data"""
        print("=" * 50)
        print("Building 3D Model of Lahore")
        print("=" * 50)

        # Try to load all data
        building_loaded = self.load_building_data()
        strategic_loaded = self.load_strategic_data()
        threat_loaded = self.load_threat_data()

        # If no data was loaded, create synthetic data
        if not (building_loaded or strategic_loaded or threat_loaded):
            print("No JSON data found. Creating synthetic model...")
            self.create_synthetic_data()

        # Calculate statistics
        self.calculate_statistics()

        # Debug canyon positions
        self.debug_canyon_positions()

        print("\n3D Model construction complete!")
        return self.model

    def debug_canyon_positions(self):
        """Print canyon positions to verify they're spread out"""
        print("\n=== CANYON POSITION DEBUG ===")

        if not self.model.canyons:
            print("No canyons to debug")
            return

        for i, canyon in enumerate(self.model.canyons[:5]):  # First 5 canyons
            if not canyon.centerline:
                print(f"Canyon {i}: No centerline")
                continue

            x_vals = [p[0] for p in canyon.centerline]
            y_vals = [p[1] for p in canyon.centerline]

            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            print(f"Canyon {i} ({canyon.name}):")
            print(f"  X: {min_x:.1f} to {max_x:.1f} (center: {center_x:.1f})")
            print(f"  Y: {min_y:.1f} to {max_y:.1f} (center: {center_y:.1f})")
            print(f"  Points: {len(canyon.centerline)}, Threat: {canyon.threat_level}")

        # Calculate overall spread
        all_x, all_y = [], []
        for canyon in self.model.canyons:
            for point in canyon.centerline:
                all_x.append(point[0])
                all_y.append(point[1])

        if all_x and all_y:
            print(f"\nOVERALL SPREAD:")
            print(f"  X range: {min(all_x):.1f} to {max(all_x):.1f} (span: {max(all_x) - min(all_x):.1f})")
            print(f"  Y range: {min(all_y):.1f} to {max(all_y):.1f} (span: {max(all_y) - min(all_y):.1f})")

    def process_all_canyons_together(self):
        """Spread canyons across Lahore at strategic locations"""
        print(f"\n=== SPREADING CANYONS ACROSS LAHORE ===")

        if not self.raw_canyon_data:
            print("No raw canyon data to process")
            return

        # Define strategic locations across Lahore (based on your building area)
        # Your building area: X(-577 to 232), Y(-578 to 583)
        lahore_strategic_locations = [
            # (x, y, name, scale_factor)
            (-400, 400, "Northwest_OldCity", 1.0),  # Old City area
            (-200, 200, "North_Central", 0.9),  # North Central
            (0, 0, "City_Center", 1.2),  # City Center
            (200, -200, "Southeast_Gulberg", 1.1),  # Gulberg area
            (-300, -300, "Southwest_Cantt", 0.8),  # Cantt area
            (100, 300, "Northeast_Defence", 0.9),  # Defence area
            (-500, 100, "West_City", 0.7),  # West Lahore
            (150, -400, "South_City", 0.8),  # South Lahore
            (-100, -100, "Central", 1.0),  # Central
            (300, 100, "East_City", 0.9),  # East Lahore
            (-450, -200, "Southwest", 0.8),  # Southwest
            (250, 350, "Northeast", 0.9),  # Northeast
        ]

        print(f"Lahore Strategic Locations: {len(lahore_strategic_locations)}")
        print(f"Canyons to place: {len(self.raw_canyon_data)}")

        # Process each canyon and place it at a different strategic location
        for i, canyon_data in enumerate(self.raw_canyon_data):
            index = canyon_data['index']
            centerline = canyon_data['centerline']
            persistence = canyon_data['persistence']
            threat_level = canyon_data['threat_level']

            # Get strategic location for this canyon
            if i < len(lahore_strategic_locations):
                target_x, target_y, location_name, scale = lahore_strategic_locations[i]
                print(f"Canyon {index} → {location_name} at ({target_x}, {target_y})")
            else:
                # If more canyons than locations, distribute randomly
                import random
                target_x = random.uniform(-500, 200)
                target_y = random.uniform(-500, 500)
                location_name = f"Random_{i}"
                scale = 1.0

            # Calculate the original center of this canyon
            original_center_x, original_center_y = 0, 0
            valid_points = 0

            for point in centerline:
                if len(point) >= 2:
                    original_center_x += point[0]
                    original_center_y += point[1]
                    valid_points += 1

            if valid_points > 0:
                original_center_x /= valid_points
                original_center_y /= valid_points

            # Convert canyon points to new location
            canyon_converted = []
            for point in centerline:
                if len(point) >= 2:
                    utm_x, utm_y = point[0], point[1]
                    height = point[2] if len(point) > 2 else 5

                    # Move canyon to strategic location
                    # 1. Remove original center offset
                    centered_x = utm_x - original_center_x
                    centered_y = utm_y - original_center_y

                    # 2. Apply scale factor (some areas have bigger/smaller canyons)
                    scaled_x = centered_x * scale
                    scaled_y = centered_y * scale

                    # 3. Move to target location
                    final_x = target_x + scaled_x
                    final_y = target_y + scaled_y
                    final_z = max(5, height * 0.2)  # Scale height for visibility

                    canyon_converted.append((final_x, final_y, final_z))

            if len(canyon_converted) < 2:
                print(f"  Skipping canyon {index} - not enough valid points")
                continue

            # Make canyons larger and more visible
            width = min(60, persistence / 50) + 40  # Wider canyons
            depth = min(35, persistence / 80) + 25  # Deeper canyons

            # Bright, highly visible colors
            threat_colors = {
                'high': (1.0, 0.0, 0.0, 0.9),  # Bright red
                'medium': (1.0, 0.6, 0.0, 0.9),  # Bright orange
                'low': (0.0, 0.8, 0.0, 0.9)  # Bright green
            }
            color = threat_colors.get(threat_level, (0.5, 0.5, 0.8, 0.9))

            # Calculate final center for naming
            final_center_x = sum([p[0] for p in canyon_converted]) / len(canyon_converted)
            final_center_y = sum([p[1] for p in canyon_converted]) / len(canyon_converted)

            canyon_obj = Canyon3D(
                centerline=canyon_converted,
                width=width,
                depth=depth,
                color=color,
                name=f"{location_name}_{threat_level}",
                threat_level=threat_level
            )

            self.model.canyons.append(canyon_obj)

            print(f"  Created: {location_name}, Center: ({final_center_x:.0f}, {final_center_y:.0f})")
            print(f"  Size: {width:.1f}x{depth:.1f}, Points: {len(canyon_converted)}")

        print(f"\n=== SUCCESSFULLY SPREAD {len(self.model.canyons)} CANYONS ===")
        print("Canyons are now distributed across strategic Lahore locations!")

        # Clear temporary data
        self.raw_canyon_data = []
        self.all_utm_points = []