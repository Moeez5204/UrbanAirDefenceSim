# renderer.py - 3D rendering system

import pygame
import math
from dataclasses import dataclass


@dataclass
class Colors:
    """Color definitions for rendering"""
    BACKGROUND = (15, 25, 45)
    GROUND = (35, 45, 30)
    GROUND_DARK = (20, 30, 20)

    ASSET_COLORS = {
        'government': (255, 50, 50),
        'cultural': (255, 150, 50),
        'religious': (255, 200, 50),
        'transportation': (50, 150, 255),
        'commercial': (50, 200, 100),
        'healthcare': (200, 50, 200),
        'infrastructure': (50, 200, 200),
        'education': (200, 100, 50)
    }


class Renderer:
    """Handles 3D rendering of the city with larger buildings"""

    def __init__(self):
        self.colors = Colors()
        self.frame_count = 0

    def setup(self, city_data, camera):
        """Setup renderer with city data and camera"""
        self.city_data = city_data
        self.camera = camera
        print("✓ Renderer setup complete")

    def render(self, screen, city_data, camera):
        """Render the entire scene"""
        self.frame_count += 1
        self.city_data = city_data
        self.camera = camera

        # Clear with background color (sky)
        screen.fill(self.colors.BACKGROUND)

        # Draw ground first (so buildings appear on top)
        self.draw_ground(screen)

        # Draw buildings
        self.draw_buildings(screen)

    def draw_ground(self, screen):
        """Draw simple ground that fills most of the screen"""
        screen_width, screen_height = screen.get_size()

        # Fill the entire screen with ground for testing
        screen.fill(self.colors.GROUND)


    def draw_buildings(self, screen):
        """Draw all buildings with proper 3D effect"""
        screen_width, screen_height = screen.get_size()

        buildings = sorted(self.city_data['buildings'],
                           key=lambda b: -((b.x - self.camera.x) ** 2 +
                                           (b.y - self.camera.y) ** 2)
                           )

        for building in buildings:
            self.draw_building(screen, building, screen_width, screen_height)

    def draw_building(self, screen, building, screen_width, screen_height):
        """Draw a single 3D building - updated for larger buildings"""
        half_width = building.width / 2
        half_depth = building.depth / 2

        angle = math.pi / 8
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        corners_3d = []
        for dx, dy in [(-half_width, -half_depth), (half_width, -half_depth),
                       (half_width, half_depth), (-half_width, half_depth)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners_3d.append((building.x + rx, building.y + ry))

        base_corners = []
        roof_corners = []

        for x, y in corners_3d:
            base_proj = self.camera.project_3d_to_2d(x, y, 0, screen_width, screen_height)
            roof_proj = self.camera.project_3d_to_2d(x, y, building.height, screen_width, screen_height)

            if base_proj[0] is None or roof_proj[0] is None:
                return

            base_corners.append(base_proj)
            roof_corners.append(roof_proj)

        # Draw building sides with gradient shading
        for i in range(4):
            next_i = (i + 1) % 4
            side_points = [
                base_corners[i][:2],
                roof_corners[i][:2],
                roof_corners[next_i][:2],
                base_corners[next_i][:2]
            ]

            shade_factor = 50 if i % 2 == 0 else 30
            side_color = tuple(max(0, c - shade_factor) for c in building.color)
            pygame.draw.polygon(screen, side_color, side_points)

        # Draw roof
        roof_points = [p[:2] for p in roof_corners]
        pygame.draw.polygon(screen, building.color, roof_points)

        # Draw more detailed window patterns for larger buildings
        if building.height > 30:
            self.draw_detailed_windows(screen, building, corners_3d, screen_width, screen_height)

        # Highlight special assets
        if building.is_special_asset:
            self.highlight_special_asset(screen, building, roof_corners)

    def draw_detailed_windows(self, screen, building, corners_3d, screen_width, screen_height):
        """Draw more detailed window patterns on larger buildings"""
        window_color = (180, 180, 140)

        num_floors = max(3, int(building.height / 12))

        for floor in range(1, num_floors):
            floor_height = floor * 12

            floor_corners = []
            for x, y in corners_3d:
                proj = self.camera.project_3d_to_2d(x, y, floor_height, screen_width, screen_height)
                if proj[0] is not None:
                    floor_corners.append(proj[:2])

            if len(floor_corners) == 4:
                pygame.draw.polygon(screen, window_color, floor_corners, 1)

                if building.width > 100:
                    mid_x = sum(p[0] for p in floor_corners) / 4
                    mid_y = sum(p[1] for p in floor_corners) / 4

                    pygame.draw.line(screen, window_color,
                                     floor_corners[0], floor_corners[3], 1)
                    pygame.draw.line(screen, window_color,
                                     floor_corners[1], floor_corners[2], 1)

    def highlight_special_asset(self, screen, building, roof_corners):
        """Highlight special assets with glowing effects"""
        center_x = sum(p[0] for p in roof_corners) / 4
        center_y = sum(p[1] for p in roof_corners) / 4

        asset_color = self.colors.ASSET_COLORS.get(
            building.asset_type, (255, 255, 0)
        )

        pulse = int(math.sin(self.frame_count * 0.1) * 3 + 5)

        pygame.draw.circle(screen, asset_color,
                           (int(center_x), int(center_y)),
                           pulse + 6, 2)

        pygame.draw.circle(screen, (255, 255, 255),
                           (int(center_x), int(center_y)),
                           pulse + 3)

        distance = math.sqrt((building.x - self.camera.x) ** 2 +
                             (building.y - self.camera.y) ** 2)

        if distance < 150 and building.asset_name:
            self.draw_asset_label(screen, building, center_x, center_y)

    def draw_asset_label(self, screen, building, center_x, center_y):
        """Draw label for special asset"""
        import pygame

        font = pygame.font.Font(None, 24)
        label = font.render(building.asset_name, True, (255, 255, 255))

        label_bg = pygame.Surface(
            (label.get_width() + 10, label.get_height() + 10),
            pygame.SRCALPHA
        )
        label_bg.fill((0, 0, 0, 180))

        label_y = center_y - building.height * 0.3

        screen.blit(label_bg, (center_x - label_bg.get_width() // 2, label_y))
        screen.blit(label, (center_x - label.get_width() // 2, label_y + 5))