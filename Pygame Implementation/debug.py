# debug_canyons.py
import json
import os

print("=== DEBUGGING CANYON DATA ===")

# Check the strategic_features_3.1.4.json file
filename = 'strategic_features_3.1.4.json'
if os.path.exists(filename):
    print(f"✓ Found {filename}")

    with open(filename, 'r') as f:
        data = json.load(f)

    print(f"File structure: {list(data.keys())}")

    if 'strategic_features' in data:
        features = data['strategic_features']
        print(f"Available features: {list(features.keys())}")

        if 'canyons' in features:
            canyons = features['canyons']
            print(f"\nFound {len(canyons)} canyons")

            # Analyze the first canyon
            if canyons:
                first_canyon = canyons[0]
                print(f"\n=== FIRST CANYON ANALYSIS ===")
                print(f"Keys in canyon: {list(first_canyon.keys())}")

                # Check for centerline
                if 'centerline' in first_canyon:
                    centerline = first_canyon['centerline']
                    print(f"Centerline type: {type(centerline)}")
                    print(f"Centerline length: {len(centerline)}")

                    if centerline:
                        print(f"First point: {centerline[0]}")
                        print(f"Number of points: {len(centerline)}")
                        print(f"Points are lists: {all(isinstance(p, list) for p in centerline)}")
                        print(f"Points have >=2 coordinates: {all(len(p) >= 2 for p in centerline)}")
                else:
                    print("❌ No 'centerline' key found!")

                # Check other important fields
                for key in ['persistence', 'threat_level', 'id', 'type']:
                    if key in first_canyon:
                        print(f"{key}: {first_canyon[key]}")
        else:
            print("❌ No 'canyons' key in strategic_features")
else:
    print(f"❌ {filename} not found in current directory")