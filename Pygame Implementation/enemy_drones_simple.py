# enemy_drones_simple.py - Simple 3D enemy drones for Lahore Urban Defense

import pygame
import random
import math


class SimpleDrone3D:
    """Simple 3D enemy drone - a rectangular prism"""

    def __init__(self, x, y, z, drone_id):
        self.x = x  # X position (center)
        self.y = y  # Y position (center)
        self.z = z  # Base height
        self.drone_id = drone_id

        # 3D dimensions (meters)
        self.width = 3.0  # Width (x-axis)
        self.depth = 3.0  # Depth (y-axis)
        self.height = 6.0  # Height (z-axis)

        # Colors - bright red for visibility
        self.top_color = (255, 80, 80)  # Top face
        self.front_color = (220, 60, 60)  # Front face
        self.side_color = (200, 40, 40)  # Side face

        # Movement
        self.speed = random.uniform(1.0, 2.0)
        self.target_x = x
        self.target_y = y
        self.target_z = z

        # Movement direction
        self.angle = random.uniform(0, 2 * math.pi)

        # Rotation for 3D effect
        self.rotation_angle = random.uniform(0, math.pi / 4)


class SimpleDroneManager:
    """Manages 3 3D red rectangles moving randomly in Lahore sky"""

    def __init__(self, city_width=3000, city_height=3000):
        self.city_width = city_width
        self.city_height = city_height

        # Calculate city center AFTER parameters are set
        self.city_center = (city_width / 2, city_height / 2)

        # Lahore area boundaries (for random movement)
        self.boundaries = {
            'min_x': self.city_center[0] - 1200,  # West boundary
            'max_x': self.city_center[0] + 1200,  # East boundary
            'min_y': self.city_center[1] - 1000,  # South boundary
            'max_y': self.city_center[1] + 1000,  # North boundary
            'min_z': 80,  # Minimum height (above buildings)
            'max_z': 250  # Maximum height
        }

        self.drones = []

        # Create 3 drones
        self.create_drones()

    def create_drones(self):
        """Create 3 3D drones at random positions"""
        print("Creating 3 enemy drones (3D red rectangles)...")

        for i in range(3):
            # Random starting position within Lahore
            x = random.uniform(self.boundaries['min_x'], self.boundaries['max_x'])
            y = random.uniform(self.boundaries['min_y'], self.boundaries['max_y'])
            z = random.uniform(self.boundaries['min_z'], self.boundaries['max_z'])

            drone = SimpleDrone3D(x, y, z, i)

            # Set random movement target
            self.set_random_target(drone)

            self.drones.append(drone)

        print("✓ Created 3 3D red rectangles moving randomly in Lahore sky")

    def set_random_target(self, drone):
        """Set a new random target for drone to move towards"""
        drone.target_x = random.uniform(self.boundaries['min_x'], self.boundaries['max_x'])
        drone.target_y = random.uniform(self.boundaries['min_y'], self.boundaries['max_y'])
        drone.target_z = random.uniform(self.boundaries['min_z'] + 20, self.boundaries['max_z'] - 20)

        # Random speed
        drone.speed = random.uniform(1.0, 2.5)

        # Calculate angle towards target
        dx = drone.target_x - drone.x
        dy = drone.target_y - drone.y
        if dx != 0 or dy != 0:
            drone.angle = math.atan2(dy, dx)

    def update(self):
        """Update drone positions - move them randomly"""
        for drone in self.drones:
            self.update_drone(drone)

    def update_drone(self, drone):
        """Update a single drone's position"""
        # Calculate distance to target
        dx = drone.target_x - drone.x
        dy = drone.target_y - drone.y
        dz = drone.target_z - drone.z

        distance_xy = math.sqrt(dx * dx + dy * dy)
        distance_z = abs(dz)

        # If close to target or random chance, pick new target
        if distance_xy < 25 or random.random() < 0.008:
            self.set_random_target(drone)
            return

        # Move towards target (XY plane)
        if distance_xy > 0:
            move_x = math.cos(drone.angle) * drone.speed
            move_y = math.sin(drone.angle) * drone.speed

            drone.x += move_x
            drone.y += move_y

        # Move vertically (slower)
        if distance_z > 0:
            vertical_speed = drone.speed * 0.4
            if dz > 0:
                drone.z = min(drone.z + vertical_speed, drone.target_z)
            else:
                drone.z = max(drone.z - vertical_speed, drone.target_z)

        # Add some random drift to make movement more interesting
        drone.x += random.uniform(-0.15, 0.15)
        drone.y += random.uniform(-0.15, 0.15)

        # Slight rotation change for visual interest
        drone.rotation_angle += random.uniform(-0.05, 0.05)

        # Keep within boundaries
        drone.x = max(self.boundaries['min_x'], min(self.boundaries['max_x'], drone.x))
        drone.y = max(self.boundaries['min_y'], min(self.boundaries['max_y'], drone.y))
        drone.z = max(self.boundaries['min_z'], min(self.boundaries['max_z'], drone.z))

    def render(self, screen, camera_system):
        """Render all drones as 3D rectangles"""
        screen_width, screen_height = screen.get_size()

        # Sort drones by distance from camera for proper 3D rendering
        sorted_drones = sorted(self.drones,
                               key=lambda d: -((d.x - camera_system.x) ** 2 +
                                               (d.y - camera_system.y) ** 2))

        for drone in sorted_drones:
            self.render_drone_3d(screen, drone, camera_system, screen_width, screen_height)

    def render_drone_3d(self, screen, drone, camera_system, screen_width, screen_height):
        """Render a single drone as a 3D rectangular prism"""
        # Half dimensions
        half_width = drone.width / 2
        half_depth = drone.depth / 2

        # Apply drone rotation for 3D effect
        cos_r = math.cos(drone.rotation_angle)
        sin_r = math.sin(drone.rotation_angle)

        # Define the 4 base corners of the drone (rotated)
        base_corners = []
        for dx, dy in [(-half_width, -half_depth), (half_width, -half_depth),
                       (half_width, half_depth), (-half_width, half_depth)]:
            # Apply rotation
            rx = dx * cos_r - dy * sin_r
            ry = dx * sin_r + dy * cos_r
            base_corners.append((drone.x + rx, drone.y + ry))

        # Project base corners to screen
        base_screen = []
        for x, y in base_corners:
            proj = camera_system.project_3d_to_2d(x, y, drone.z, screen_width, screen_height)
            if proj[0] is None:
                return  # Drone is behind camera
            base_screen.append(proj[:2])

        # Project top corners to screen (base_z + height)
        top_screen = []
        for x, y in base_corners:
            proj = camera_system.project_3d_to_2d(x, y, drone.z + drone.height, screen_width, screen_height)
            if proj[0] is None:
                return
            top_screen.append(proj[:2])

        # Draw 3D sides of the drone
        self.draw_drone_sides(screen, base_screen, top_screen, drone)

        # Draw drone top
        self.draw_drone_top(screen, top_screen, drone)

    def draw_drone_sides(self, screen, base_screen, top_screen, drone):
        """Draw the sides of the 3D drone"""
        # Draw each of the 4 sides
        for i in range(4):
            next_i = (i + 1) % 4

            # Points for this side polygon
            side_points = [
                base_screen[i],
                top_screen[i],
                top_screen[next_i],
                base_screen[next_i]
            ]

            # Choose color based on which side
            if i == 0:  # Front side
                side_color = drone.front_color
            elif i == 2:  # Back side (darker)
                side_color = (max(0, drone.front_color[0] - 40),
                              max(0, drone.front_color[1] - 40),
                              max(0, drone.front_color[2] - 40))
            else:  # Side faces
                side_color = drone.side_color

            # Draw the side polygon
            pygame.draw.polygon(screen, side_color, side_points)

            # Draw side outline
            pygame.draw.polygon(screen, (150, 30, 30), side_points, 1)

    def draw_drone_top(self, screen, top_screen, drone):
        """Draw the top of the 3D drone"""
        # Draw top face
        pygame.draw.polygon(screen, drone.top_color, top_screen)

        # Draw top outline
        pygame.draw.polygon(screen, (180, 50, 50), top_screen, 1)

        # Draw a small "rotor" or antenna on top
        if len(top_screen) >= 4:
            # Calculate center of top
            center_x = sum(p[0] for p in top_screen) / 4
            center_y = sum(p[1] for p in top_screen) / 4

            # Draw small circle/rotor
            rotor_radius = 3
            pygame.draw.circle(screen, (100, 100, 100),
                               (int(center_x), int(center_y)),
                               rotor_radius)
            pygame.draw.circle(screen, (50, 50, 50),
                               (int(center_x), int(center_y)),
                               rotor_radius, 1)


# Test the drone system
if __name__ == "__main__":
    # Quick test
    print("3D Drone Manager Test")
    print("-" * 40)

    manager = SimpleDroneManager()

    print(f"\nDrone Positions (3D):")
    for i, drone in enumerate(manager.drones):
        print(f"Drone {i}: X={drone.x:.1f}, Y={drone.y:.1f}, Z={drone.z:.1f}")
        print(f"       Size: {drone.width}x{drone.depth}x{drone.height}")

    print(f"\nBoundaries:")
    print(f"X: {manager.boundaries['min_x']:.0f} to {manager.boundaries['max_x']:.0f}")
    print(f"Y: {manager.boundaries['min_y']:.0f} to {manager.boundaries['max_y']:.0f}")
    print(f"Z: {manager.boundaries['min_z']:.0f} to {manager.boundaries['max_z']:.0f}")

    print("\nUpdating drones 10 times...")
    for i in range(10):
        manager.update()
        print(
            f"Step {i + 1}: Drone 0 at ({manager.drones[0].x:.1f}, {manager.drones[0].y:.1f}, {manager.drones[0].z:.1f})")