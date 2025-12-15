# data_loader.py - Data loading from JSON files

import json


class DataLoader:
    """Handles loading and processing of data files"""

    def __init__(self):
        self.special_assets = []

    def load_special_assets(self, filename='assets/lahore_defense_3d.json'):
        """Load special assets from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.special_assets = data.get('sector_data', [])
                print(f"✓ Loaded {len(self.special_assets)} special assets")

                # Adjust positions for larger city
                self.adjust_asset_positions()
                return self.special_assets
        except FileNotFoundError:
            print("⚠ Defense data not found, using default assets")
            return self.create_default_assets()
        except Exception as e:
            print(f"✗ Error loading assets: {e}")
            return self.create_default_assets()

    def adjust_asset_positions(self):
        """Adjust asset positions for the larger city scale"""
        scale_factor = 1.5  # Scale up positions for larger city
        for asset in self.special_assets:
            if 'position' in asset:
                pos = asset['position']
                if 'x' in pos:
                    pos['x'] *= scale_factor
                if 'y' in pos:
                    pos['y'] *= scale_factor

    def create_default_assets(self):
        """Create default special assets for larger city"""
        city_center = (1500, 1500)  # Adjusted for larger city

        return [
            {
                'name': 'Governor House',
                'position': {'x': city_center[0], 'y': city_center[1], 'z': 80},
                'priority': 0.95,
                'sector': 'Central_Lahore',
                'type': 'government',
                'resources_allocated': 3,
                'description': 'Seat of Punjab Government'
            },
            {
                'name': 'Lahore Fort',
                'position': {'x': city_center[0] - 600, 'y': city_center[1] - 300, 'z': 60},
                'priority': 0.90,
                'sector': 'Walled_City',
                'type': 'cultural',
                'resources_allocated': 2,
                'description': 'UNESCO World Heritage Site'
            },
            {
                'name': 'Badshahi Mosque',
                'position': {'x': city_center[0] - 570, 'y': city_center[1] - 270, 'z': 90},
                'priority': 0.88,
                'sector': 'Walled_City',
                'type': 'religious',
                'resources_allocated': 2,
                'description': '17th Century Mughal Mosque'
            },
            {
                'name': 'Allama Iqbal Airport',
                'position': {'x': city_center[0] + 900, 'y': city_center[1] + 750, 'z': 40},
                'priority': 0.85,
                'sector': 'Other_Sector',
                'type': 'transportation',
                'resources_allocated': 4,
                'description': 'International Airport'
            },
            {
                'name': 'Liberty Market',
                'position': {'x': city_center[0] + 450, 'y': city_center[1] + 150, 'z': 60},
                'priority': 0.70,
                'sector': 'Gulberg',
                'type': 'commercial',
                'resources_allocated': 1,
                'description': 'Major Shopping Area'
            },
            {
                'name': 'Mayo Hospital',
                'position': {'x': city_center[0] + 150, 'y': city_center[1] + 75, 'z': 70},
                'priority': 0.85,
                'sector': 'Central_Lahore',
                'type': 'healthcare',
                'resources_allocated': 2,
                'description': 'Major Public Hospital'
            },
            {
                'name': 'University of Lahore',
                'position': {'x': city_center[0] - 750, 'y': city_center[1] + 450, 'z': 50},
                'priority': 0.75,
                'sector': 'Other_Sector',
                'type': 'education',
                'resources_allocated': 1,
                'description': 'Major University Campus'
            },
            {
                'name': 'Minar-e-Pakistan',
                'position': {'x': city_center[0] - 525, 'y': city_center[1] - 225, 'z': 85},
                'priority': 0.82,
                'sector': 'Walled_City',
                'type': 'cultural',
                'resources_allocated': 2,
                'description': 'National Monument'
            },
            {
                'name': 'Lahore Railway Station',
                'position': {'x': city_center[0] - 300, 'y': city_center[1] + 200, 'z': 45},
                'priority': 0.78,
                'sector': 'Central_Lahore',
                'type': 'transportation',
                'resources_allocated': 2,
                'description': 'Main Railway Station'
            },
            {
                'name': 'Packages Mall',
                'position': {'x': city_center[0] + 600, 'y': city_center[1] + 300, 'z': 55},
                'priority': 0.72,
                'sector': 'Gulberg',
                'type': 'commercial',
                'resources_allocated': 1,
                'description': 'Large Shopping Mall'
            }
        ]