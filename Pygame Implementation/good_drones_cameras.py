# good_drones_cameras.py - Good drones with fixed camera positions

import pygame
import math


class GoodDrone3D:
    """Good drone - a blue rectangular prism"""

    def __init__(self, x, y, z, drone_id):
        self.x = x  # X position (center)
        self.y = y  # Y position (center)
        self.z = z  # Base height
        self.drone_id = drone_id
        self.name = f"Drone_{drone_id + 1}"

        # 3D dimensions (meters)
        self.width = 2.5
        self.depth = 2.5
        self.height = 5.0

        # Colors - blue for good drones
        self.top_color = (80, 120, 255)
        self.front_color = (60, 100, 220)
        self.side_color = (40, 80, 200)

        # Fixed camera positions for each drone
        self.camera_views = [
            # Camera positioned behind and above drone, looking at drone
            {
                'x': x - 20,  # 20m behind
                'y': y - 20,  # 20m to the side
                'z': z + 15,  # 15m above
                'yaw': math.atan2(y - (y - 20), x - (x - 20)),  # Look at drone
                'pitch': -0.4
            },
            # Camera positioned to the side
            {
                'x': x + 25,
                'y': y,
                'z': z + 10,
                'yaw': math.atan2(y - y, x - (x + 25)),
                'pitch': -0.3
            },
            # Camera positioned in front
            {
                'x': x,
                'y': y + 25,
                'z': z + 8,
                'yaw': math.atan2(y - (y + 25), x - x),
                'pitch': -0.35
            }
        ]

        # Default to first camera view
        self.current_camera_view = 0
        self.selected = False


class GoodDroneManager:
    """Manages 3 good blue drones with fixed camera positions"""

    def __init__(self, city_width=3000, city_height=3000):
        self.city_width = city_width
        self.city_height = city_height
        self.city_center = (city_width / 2, city_height / 2)

        # Starting positions for 3 drones
        self.drone_start_positions = [
            # Drone 1: High above city center
            {'x': self.city_center[0], 'y': self.city_center[1], 'z': 250},
            # Drone 2: Near Walled City
            {'x': self.city_center[0] - 600, 'y': self.city_center[1] - 300, 'z': 200},
            # Drone 3: Near Gulberg
            {'x': self.city_center[0] + 450, 'y': self.city_center[1] + 150, 'z': 220}
        ]

        self.drones = []
        self.active_drone_camera = None  # Which drone's camera is active

        # Create 3 drones
        self.create_drones()

    def create_drones(self):
        """Create 3 good drones at strategic positions"""
        print("Creating 3 good drones (blue, stationary)...")

        for i in range(3):
            pos = self.drone_start_positions[i]
            drone = GoodDrone3D(pos['x'], pos['y'], pos['z'], i)

            # Adjust camera positions for each drone's location
            self.adjust_drone_cameras(drone)

            self.drones.append(drone)

        print("✓ Created 3 stationary blue drones")
        print("  Drone 1: Above City Center")
        print("  Drone 2: Walled City area")
        print("  Drone 3: Gulberg area")

    def adjust_drone_cameras(self, drone):
        """Adjust camera positions to look at the drone properly"""
        # Camera 1: Behind-left, looking at drone
        drone.camera_views[0]['x'] = drone.x - 20
        drone.camera_views[0]['y'] = drone.y - 20
        drone.camera_views[0]['z'] = drone.z + 15
        drone.camera_views[0]['yaw'] = math.atan2(drone.y - (drone.y - 20), drone.x - (drone.x - 20))

        # Camera 2: Right side, looking at drone
        drone.camera_views[1]['x'] = drone.x + 25
        drone.camera_views[1]['y'] = drone.y
        drone.camera_views[1]['z'] = drone.z + 10
        drone.camera_views[1]['yaw'] = math.atan2(drone.y - drone.y, drone.x - (drone.x + 25))

        # Camera 3: In front, looking at drone
        drone.camera_views[2]['x'] = drone.x
        drone.camera_views[2]['y'] = drone.y + 25
        drone.camera_views[2]['z'] = drone.z + 8
        drone.camera_views[2]['yaw'] = math.atan2(drone.y - (drone.y + 25), drone.x - drone.x)

    def get_drone_camera_position(self, drone_index, camera_view=0):
        """Get a specific camera position for a drone"""
        if 0 <= drone_index < len(self.drones):
            drone = self.drones[drone_index]
            if 0 <= camera_view < len(drone.camera_views):
                cam = drone.camera_views[camera_view]
                return {
                    'x': cam['x'],
                    'y': cam['y'],
                    'z': cam['z'],
                    'yaw': cam['yaw'],
                    'pitch': cam['pitch'],
                    'drone_name': drone.name,
                    'drone_id': drone.drone_id
                }
        return None

    def activate_drone_camera(self, drone_index, camera_view=0):
        """Activate a drone's camera view"""
        if 0 <= drone_index < len(self.drones):
            self.active_drone_camera = drone_index
            self.drones[drone_index].current_camera_view = camera_view
            self.drones[drone_index].selected = True

            # Deselect other drones
            for i, drone in enumerate(self.drones):
                if i != drone_index:
                    drone.selected = False

            print(f"✓ Activated {self.drones[drone_index].name}'s camera view")
            return True
        return False

    def deactivate_drone_camera(self):
        """Deactivate drone camera view"""
        if self.active_drone_camera is not None:
            drone_name = self.drones[self.active_drone_camera].name
            self.drones[self.active_drone_camera].selected = False
            self.active_drone_camera = None
            print(f"✗ Deactivated {drone_name}'s camera view")
            return True
        return False

    def update(self):
        """Update drone states"""
        pass

    def render(self, screen, camera_system, show_labels=True):
        """Render all good drones"""
        screen_width, screen_height = screen.get_size()

        # Sort drones by distance from camera
        sorted_drones = sorted(self.drones,
                               key=lambda d: -((d.x - camera_system.x) ** 2 +
                                               (d.y - camera_system.y) ** 2))

        for drone in sorted_drones:
            self.render_drone_3d(screen, drone, camera_system, screen_width, screen_height, show_labels)

    def render_drone_3d(self, screen, drone, camera_system, screen_width, screen_height, show_labels):
        """Render a single good drone"""
        # Half dimensions
        half_width = drone.width / 2
        half_depth = drone.depth / 2

        # Define base corners
        base_corners = []
        for dx, dy in [(-half_width, -half_depth), (half_width, -half_depth),
                       (half_width, half_depth), (-half_width, half_depth)]:
            base_corners.append((drone.x + dx, drone.y + dy))

        # Project base corners to screen
        base_screen = []
        for x, y in base_corners:
            proj = camera_system.project_3d_to_2d(x, y, drone.z, screen_width, screen_height)
            if proj[0] is None:
                return
            base_screen.append(proj[:2])

        # Project top corners to screen
        top_screen = []
        for x, y in base_corners:
            proj = camera_system.project_3d_to_2d(x, y, drone.z + drone.height, screen_width, screen_height)
            if proj[0] is None:
                return
            top_screen.append(proj[:2])

        # Draw 3D sides
        for i in range(4):
            next_i = (i + 1) % 4
            side_points = [
                base_screen[i],
                top_screen[i],
                top_screen[next_i],
                base_screen[next_i]
            ]

            if i == 0:
                side_color = drone.front_color
            elif i == 2:
                side_color = (max(0, drone.front_color[0] - 40),
                              max(0, drone.front_color[1] - 40),
                              max(0, drone.front_color[2] - 40))
            else:
                side_color = drone.side_color

            pygame.draw.polygon(screen, side_color, side_points)
            pygame.draw.polygon(screen, (30, 60, 150), side_points, 1)

        # Draw top
        pygame.draw.polygon(screen, drone.top_color, top_screen)
        pygame.draw.polygon(screen, (50, 80, 180), top_screen, 1)

        # Draw selection indicator
        if drone.selected:
            self.draw_selection_indicator(screen, top_screen, drone)

        # Draw label
        if show_labels:
            self.draw_drone_label(screen, drone, top_screen)

    def draw_selection_indicator(self, screen, top_screen, drone):
        """Draw selection indicator around selected drone"""
        if len(top_screen) >= 4:
            center_x = sum(p[0] for p in top_screen) / 4
            center_y = sum(p[1] for p in top_screen) / 4

            pulse_factor = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 3 + 2
            radius = 12 + pulse_factor

            pygame.draw.circle(screen, (0, 255, 255, 150),
                               (int(center_x), int(center_y)),
                               int(radius), 2)

    def draw_drone_label(self, screen, drone, top_screen):
        """Draw label showing drone name"""
        if len(top_screen) >= 4:
            center_x = sum(p[0] for p in top_screen) / 4
            center_y = sum(p[1] for p in top_screen) / 4

            label_y = center_y - 25
            font = pygame.font.Font(None, 20)
            label_text = f"{drone.name}"
            label_color = (0, 200, 255) if drone.selected else (100, 180, 255)

            label_surf = font.render(label_text, True, label_color)
            label_rect = label_surf.get_rect(center=(center_x, label_y))

            label_bg = pygame.Surface((label_rect.width + 6, label_rect.height + 4), pygame.SRCALPHA)
            bg_color = (0, 0, 0, 180) if drone.selected else (0, 0, 0, 120)
            label_bg.fill(bg_color)

            screen.blit(label_bg, (label_rect.x - 3, label_rect.y - 2))
            screen.blit(label_surf, label_rect)