# ui_elements.py - User interface elements (Updated with drone camera buttons)

import pygame
import math
from dataclasses import dataclass


@dataclass
class UIStyles:
    """UI styling definitions"""
    TEXT_WHITE = (255, 255, 255)
    TEXT_GREEN = (200, 255, 200)
    TEXT_YELLOW = (255, 200, 100)
    TEXT_BLUE = (200, 200, 255)
    TEXT_RED = (255, 100, 100)
    TEXT_ORANGE = (255, 150, 50)
    TEXT_PURPLE = (200, 100, 255)
    TEXT_CYAN = (100, 255, 255)

    BG_DARK = (0, 0, 0, 180)
    BG_LIGHT = (0, 0, 0, 120)

    CROSSHAIR_OUTER = (255, 255, 255, 128)
    CROSSHAIR_INNER = (255, 100, 100)

    # Button colors
    BUTTON_NORMAL = (50, 50, 70, 200)
    BUTTON_HOVER = (70, 70, 100, 200)
    BUTTON_ACTIVE = (100, 150, 200, 200)
    BUTTON_TEXT = (230, 230, 255)
    BUTTON_BORDER = (100, 100, 150)

    # Drone camera button colors
    DRONE_BUTTON_NORMAL = (50, 70, 90, 200)
    DRONE_BUTTON_HOVER = (70, 100, 120, 200)
    DRONE_BUTTON_ACTIVE = (100, 180, 220, 200)
    DRONE_BUTTON_TEXT = (200, 230, 255)


class CameraButton:
    """Camera selection button"""

    def __init__(self, x, y, width, height, text, camera_index, is_drone=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.camera_index = camera_index
        self.is_drone = is_drone
        self.hover = False
        self.active = False

    def draw(self, screen, styles):
        """Draw the button"""
        # Determine button color based on type
        if self.active:
            if self.is_drone:
                color = styles.DRONE_BUTTON_ACTIVE
            else:
                color = styles.BUTTON_ACTIVE
        elif self.hover:
            if self.is_drone:
                color = styles.DRONE_BUTTON_HOVER
            else:
                color = styles.BUTTON_HOVER
        else:
            if self.is_drone:
                color = styles.DRONE_BUTTON_NORMAL
            else:
                color = styles.BUTTON_NORMAL

        # Draw button background
        button_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(button_surface, color, (0, 0, self.rect.width, self.rect.height), border_radius=6)

        # Draw border
        border_color = styles.BUTTON_BORDER
        if self.is_drone:
            border_color = (80, 120, 150)
        pygame.draw.rect(button_surface, border_color,
                         (0, 0, self.rect.width, self.rect.height), 2, border_radius=6)

        screen.blit(button_surface, self.rect)

        # Draw button text
        if self.is_drone:
            font = pygame.font.Font(None, 18)
            text_color = styles.DRONE_BUTTON_TEXT
        else:
            font = pygame.font.Font(None, 20)
            text_color = styles.BUTTON_TEXT

        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        """Check if mouse is hovering over button"""
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover


class UIManager:
    """Manages all UI elements and HUD"""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.styles = UIStyles()

        self.title_font = pygame.font.Font(None, 36)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.button_font = pygame.font.Font(None, 20)

        self.show_help = True
        self.show_minimap = True
        self.show_camera_buttons = True

        self.city_data = None
        self.special_assets = []
        self.camera_system = None
        self.good_drone_manager = None

        self.camera_pos = (0, 0, 0)
        self.camera_yaw = 0
        self.camera_pitch = 0
        self.flight_mode = True
        self.current_sector = "Unknown"
        self.fps = 60.0

        # Camera buttons
        self.camera_buttons = []
        self.drone_camera_buttons = []
        self.setup_camera_buttons()

    def setup(self, city_data, special_assets):
        """Setup UI with city data"""
        self.city_data = city_data
        self.special_assets = special_assets
        print("✓ UI setup complete")

    def setup_camera_buttons(self):
        """Setup camera selection buttons"""
        button_width = 120
        button_height = 35
        button_spacing = 8

        # Regular camera buttons (on right side)
        start_x = self.screen_width - button_width - 20
        start_y = 100

        camera_names = [
            "Center View",
            "Walled City",
            "Gulberg",
            "Cantonment",
            "High Altitude",
            "Edge View"
        ]

        for i in range(6):
            y = start_y + i * (button_height + button_spacing)
            button = CameraButton(
                start_x, y, button_width, button_height,
                camera_names[i], i, is_drone=False
            )
            self.camera_buttons.append(button)

        # Drone camera buttons (below regular cameras)
        drone_start_y = start_y + 6 * (button_height + button_spacing) + 20

        drone_names = [
            "Drone 1 View",
            "Drone 2 View",
            "Drone 3 View"
        ]

        for i in range(3):
            y = drone_start_y + i * (button_height + button_spacing)
            button = CameraButton(
                start_x, y, button_width, button_height,
                drone_names[i], i, is_drone=True
            )
            self.drone_camera_buttons.append(button)

        # Add a "Free Camera" button at the bottom
        free_button_y = drone_start_y + 3 * (button_height + button_spacing) + 10
        self.free_button = CameraButton(
            start_x, free_button_y, button_width, button_height,
            "Free Camera", -1, is_drone=False
        )
        self.free_button.active = True  # Start with free camera active

    def set_camera_system(self, camera_system):
        """Set the camera system reference"""
        self.camera_system = camera_system

    def set_good_drone_manager(self, drone_manager):
        """Set the good drone manager reference"""
        self.good_drone_manager = drone_manager

    def update(self, camera_pos, camera_yaw, camera_pitch, flight_mode, current_sector, fps):
        """Update UI state"""
        self.camera_pos = camera_pos
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        self.flight_mode = flight_mode
        self.current_sector = current_sector
        self.fps = fps

        # Update button states based on camera system
        if self.camera_system:
            # Update free button
            self.free_button.active = (self.camera_system.current_fixed_camera is None)

            # Update regular camera buttons
            for i, button in enumerate(self.camera_buttons):
                button.active = (self.camera_system.current_fixed_camera == i)

            # Update drone camera buttons
            if self.good_drone_manager and hasattr(self.good_drone_manager, 'active_drone_camera'):
                active_drone = self.good_drone_manager.active_drone_camera
                for i, button in enumerate(self.drone_camera_buttons):
                    button.active = (active_drone == i)

    def render(self, screen):
        """Render all UI elements"""
        self.draw_crosshair(screen)
        self.draw_hud(screen)

        if self.show_minimap:
            self.draw_minimap(screen)

        if self.show_camera_buttons:
            self.draw_camera_buttons(screen)

    def draw_camera_buttons(self, screen):
        """Draw camera selection buttons"""
        # Draw regular camera title
        title = self.small_font.render("CITY CAMERAS", True, self.styles.TEXT_BLUE)
        title_x = self.screen_width - 140
        screen.blit(title, (title_x, 75))

        # Draw drone camera title
        if len(self.drone_camera_buttons) > 0:
            drone_title = self.small_font.render("DRONE CAMERAS", True, self.styles.TEXT_CYAN)
            drone_title_y = self.drone_camera_buttons[0].rect.y - 20
            screen.blit(drone_title, (title_x, drone_title_y))

        # Draw all buttons
        mouse_pos = pygame.mouse.get_pos()

        # Update and draw regular camera buttons
        for button in self.camera_buttons:
            button.check_hover(mouse_pos)
            button.draw(screen, self.styles)

        # Update and draw drone camera buttons
        for button in self.drone_camera_buttons:
            button.check_hover(mouse_pos)
            button.draw(screen, self.styles)

        # Update and draw free camera button
        self.free_button.check_hover(mouse_pos)
        self.free_button.draw(screen, self.styles)

    def handle_click(self, mouse_pos):
        """Handle mouse clicks on camera buttons"""
        if not self.camera_system:
            return None

        result = None

        # Check regular camera buttons
        for button in self.camera_buttons:
            if button.rect.collidepoint(mouse_pos):
                if not button.active:
                    self.camera_system.switch_to_fixed_camera(button.camera_index)
                    # Deactivate drone camera if active
                    if self.good_drone_manager:
                        self.good_drone_manager.deactivate_drone_camera()
                result = f"city_camera_{button.camera_index}"
                break

        # Check drone camera buttons
        for button in self.drone_camera_buttons:
            if button.rect.collidepoint(mouse_pos):
                if self.good_drone_manager:
                    if not button.active:
                        self.good_drone_manager.activate_drone_camera(button.camera_index)
                        # Switch camera to drone view
                        if self.camera_system:
                            self.camera_system.current_fixed_camera = None  # Exit any city camera mode
                    else:
                        # If already active, deactivate and go to free camera
                        self.good_drone_manager.deactivate_drone_camera()
                result = f"drone_camera_{button.camera_index}"
                break

        # Check free camera button
        if self.free_button.rect.collidepoint(mouse_pos):
            if not self.free_button.active:
                self.camera_system.switch_to_free_camera()
                # Deactivate drone camera if active
                if self.good_drone_manager:
                    self.good_drone_manager.deactivate_drone_camera()
            result = "free_camera"

        return result

    def draw_crosshair(self, screen):
        """Draw aiming crosshair"""
        center_x, center_y = self.screen_width // 2, self.screen_height // 2

        size = 15
        pygame.draw.line(screen, self.styles.CROSSHAIR_OUTER,
                         (center_x - size, center_y),
                         (center_x + size, center_y), 2)
        pygame.draw.line(screen, self.styles.CROSSHAIR_OUTER,
                         (center_x, center_y - size),
                         (center_x, center_y + size), 2)

        pygame.draw.circle(screen, self.styles.CROSSHAIR_INNER,
                           (center_x, center_y), 2)

    def draw_hud(self, screen):
        """Draw heads-up display information"""
        pos_text = f"Position: ({self.camera_pos[0]:.0f}, {self.camera_pos[1]:.0f}, {self.camera_pos[2]:.0f})"
        dir_text = f"Direction: {math.degrees(self.camera_yaw):.0f}°"

        # Get camera mode text
        camera_mode = "Free Camera"
        camera_color = self.styles.TEXT_GREEN

        if self.camera_system:
            if self.camera_system.current_fixed_camera is not None:
                camera_mode = self.camera_system.get_current_camera_name()
                camera_color = self.styles.TEXT_YELLOW
            elif self.good_drone_manager and self.good_drone_manager.active_drone_camera is not None:
                drone = self.good_drone_manager.drones[self.good_drone_manager.active_drone_camera]
                camera_mode = f"Drone View ({drone.name})"
                camera_color = self.styles.TEXT_CYAN

        mode_text = f"Camera: {camera_mode}"
        sector_text = f"Sector: {self.current_sector}"

        pos_surf = self.font.render(pos_text, True, self.styles.TEXT_GREEN)
        dir_surf = self.font.render(dir_text, True, self.styles.TEXT_GREEN)
        mode_surf = self.font.render(mode_text, True, camera_color)
        sector_surf = self.font.render(sector_text, True, self.styles.TEXT_YELLOW)

        screen.blit(pos_surf, (10, self.screen_height - 90))
        screen.blit(dir_surf, (10, self.screen_height - 60))
        screen.blit(mode_surf, (10, self.screen_height - 30))

        screen.blit(sector_surf, (self.screen_width - sector_surf.get_width() - 140, 40))

        fps_text = self.font.render(f"FPS: {self.fps:.1f}", True, self.styles.TEXT_WHITE)
        screen.blit(fps_text, (self.screen_width - 100, 10))

        if self.show_help:
            self.draw_help_panel(screen)

    def draw_help_panel(self, screen):
        """Draw help/controls panel"""
        help_lines = [
            "CONTROLS:",
            "WSAD - Move forward/backward/strafe",
            "Mouse - Look around",
            "Space - Move UP",
            "Shift - Move DOWN",
            "F - Toggle flight/walk mode",
            "M - Toggle minimap",
            "H - Toggle help",
            "R - Reset position",
            "1-6 - Switch cameras",
            "7-9 - Drone cameras",
            "0 - Free camera",
            "ESC - Exit"
        ]

        max_width = 0
        for line in help_lines:
            text_surf = self.small_font.render(line, True, self.styles.TEXT_WHITE)
            max_width = max(max_width, text_surf.get_width())

        panel_width = max_width + 20
        panel_height = len(help_lines) * 25 + 10

        help_bg = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        help_bg.fill(self.styles.BG_DARK)

        screen.blit(help_bg, (10, 10))

        for i, line in enumerate(help_lines):
            color = self.styles.TEXT_YELLOW if i == 0 else self.styles.TEXT_WHITE
            text_surf = self.small_font.render(line, True, color)
            screen.blit(text_surf, (20, 20 + i * 25))

    def draw_minimap(self, screen):
        """Draw minimap in top-right corner"""
        minimap_size = 180
        minimap_x = self.screen_width - minimap_size - 150
        minimap_y = 10

        minimap_bg = pygame.Surface((minimap_size, minimap_size), pygame.SRCALPHA)
        minimap_bg.fill(self.styles.BG_DARK)
        screen.blit(minimap_bg, (minimap_x, minimap_y))

        if self.city_data:
            city_width = self.city_data['width']
            city_height = self.city_data['height']
            scale = minimap_size / max(city_width, city_height)

            # Draw buildings only
            if 'buildings' in self.city_data:
                for building in self.city_data['buildings']:
                    x = minimap_x + building.x * scale
                    y = minimap_y + building.y * scale
                    size = max(1, building.width * scale * 0.15)

                    color = building.color
                    if building.is_special_asset:
                        asset_colors = {
                            'government': (255, 50, 50),
                            'cultural': (255, 150, 50),
                            'religious': (255, 200, 50),
                            'transportation': (50, 150, 255),
                            'commercial': (50, 200, 100),
                            'healthcare': (200, 50, 200),
                            'infrastructure': (50, 200, 200),
                            'education': (200, 100, 50)
                        }
                        color = asset_colors.get(building.asset_type, (255, 255, 0))

                    pygame.draw.rect(screen, color,
                                     (x - size / 2, y - size / 2, size, size))

        cam_x = minimap_x + self.camera_pos[0] * 0.06
        cam_y = minimap_y + self.camera_pos[1] * 0.06

        pygame.draw.circle(screen, self.styles.TEXT_WHITE,
                           (int(cam_x), int(cam_y)), 3)
        pygame.draw.circle(screen, (0, 0, 0),
                           (int(cam_x), int(cam_y)), 1)

        dir_length = 8
        dir_x = cam_x + math.cos(self.camera_yaw) * dir_length
        dir_y = cam_y + math.sin(self.camera_yaw) * dir_length
        pygame.draw.line(screen, self.styles.TEXT_RED,
                         (cam_x, cam_y), (dir_x, dir_y), 2)

        pygame.draw.rect(screen, (100, 100, 100),
                         (minimap_x, minimap_y, minimap_size, minimap_size), 1)

        title = self.small_font.render("MAP", True, self.styles.TEXT_BLUE)
        screen.blit(title, (minimap_x + minimap_size // 2 - title.get_width() // 2,
                            minimap_y - 15))

    def toggle_help(self):
        """Toggle help panel visibility"""
        self.show_help = not self.show_help

    def toggle_minimap(self):
        """Toggle minimap visibility"""
        self.show_minimap = not self.show_minimap