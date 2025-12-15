# main.py - Main entry point (Updated with drone camera buttons)

import pygame
import sys
from data_loader import DataLoader
from city_generator import CityGenerator
from camera_system import CameraSystem
from renderer import Renderer
from ui_elements import UIManager
from enemy_drones_simple import SimpleDroneManager
from good_drones_cameras import GoodDroneManager


class LahoreUrbanDefense:
    """Main application class for larger city"""

    def __init__(self, width=1600, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Lahore Urban Defense - With Drone Cameras")

        # Initialize components
        self.data_loader = DataLoader()
        self.city_generator = CityGenerator()
        self.camera = CameraSystem()
        self.renderer = Renderer()
        self.ui = UIManager(width, height)
        self.enemy_drone_manager = SimpleDroneManager()
        self.good_drone_manager = GoodDroneManager()

        # Game state
        self.clock = pygame.time.Clock()
        self.running = True
        self.frame_count = 0

        # Initialize systems
        self.init_systems()

    def init_systems(self):
        """Initialize all game systems"""
        print("=" * 60)
        print("LAHORE URBAN DEFENSE - DRONE CAMERA SYSTEM")
        print("=" * 60)
        print("Features:")
        print("  • 3 Enemy Drones (red) - moving randomly")
        print("  • 3 Good Drones (blue) - stationary")
        print("  • 6 Fixed City Camera Positions")
        print("  • 3 Drone Camera Positions (buttons 7-9)")
        print("=" * 60)

        # Load data
        self.special_assets = self.data_loader.load_special_assets()

        # Generate city
        self.city_data = self.city_generator.generate_city(self.special_assets)

        # Setup camera
        self.camera.setup(self.city_data['center'], self.width, self.height)

        # Setup renderer
        self.renderer.setup(self.city_data, self.camera)

        # Setup UI with references
        self.ui.setup(self.city_data, self.special_assets)
        self.ui.set_camera_system(self.camera)
        self.ui.set_good_drone_manager(self.good_drone_manager)

        print("\nCONTROLS:")
        print("  Mouse - Look around")
        print("  WSAD/Space/Shift - Move (free camera only)")
        print("  F - Toggle flight mode (free camera only)")
        print("  R - Reset position")
        print("  M - Toggle minimap")
        print("  H - Toggle help")
        print("  1-6 - City cameras")
        print("  7-9 - Drone cameras")
        print("  0 - Free camera")
        print("  Click buttons on right for cameras")
        print("  ESC - Exit")
        print("=" * 60 + "\n")

    def switch_to_drone_camera(self, drone_index):
        """Switch to viewing from a drone's camera"""
        if self.good_drone_manager.activate_drone_camera(drone_index):
            # Get the drone camera position
            cam_data = self.good_drone_manager.get_drone_camera_position(drone_index)
            if cam_data:
                # Update the main camera to drone camera position
                self.camera.x = cam_data['x']
                self.camera.y = cam_data['y']
                self.camera.z = cam_data['z']
                self.camera.yaw = cam_data['yaw']
                self.camera.pitch = cam_data['pitch']
                self.camera.current_fixed_camera = None  # Not in city camera mode
                return True
        return False

    def handle_events(self):
        """Handle all PyGame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle keyboard events
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_h:
                    self.ui.toggle_help()
                elif event.key == pygame.K_m:
                    self.ui.toggle_minimap()
                elif event.key == pygame.K_f:
                    if self.camera.current_fixed_camera is None and self.good_drone_manager.active_drone_camera is None:
                        self.camera.toggle_flight_mode()
                elif event.key == pygame.K_r:
                    if self.camera.current_fixed_camera is None and self.good_drone_manager.active_drone_camera is None:
                        self.camera.reset_position()
                # City camera keys (1-6)
                elif pygame.K_1 <= event.key <= pygame.K_6:
                    camera_index = event.key - pygame.K_1
                    self.camera.switch_to_fixed_camera(camera_index)
                    self.good_drone_manager.deactivate_drone_camera()
                # Drone camera keys (7-9)
                elif pygame.K_7 <= event.key <= pygame.K_9:
                    drone_index = event.key - pygame.K_7  # 7 -> 0, 8 -> 1, 9 -> 2
                    if drone_index < 3:  # Make sure it's a valid drone
                        self.switch_to_drone_camera(drone_index)
                        self.camera.current_fixed_camera = None  # Not in city camera mode
                # 0 for free camera
                elif event.key == pygame.K_0:
                    self.camera.switch_to_free_camera()
                    self.good_drone_manager.deactivate_drone_camera()

            # Handle mouse wheel for speed adjustment (free camera only)
            elif event.type == pygame.MOUSEWHEEL:
                if self.camera.current_fixed_camera is None and self.good_drone_manager.active_drone_camera is None:
                    self.camera.adjust_speed(event.y)

            # Handle mouse clicks for camera buttons
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()
                    result = self.ui.handle_click(mouse_pos)

                    # Handle the button click result
                    if result:
                        if result.startswith("city_camera_"):
                            # City camera was clicked
                            pass  # Already handled by UI
                        elif result.startswith("drone_camera_"):
                            # Drone camera was clicked
                            drone_index = int(result.split("_")[2])
                            self.switch_to_drone_camera(drone_index)
                        elif result == "free_camera":
                            # Free camera button was clicked
                            self.camera.switch_to_free_camera()
                            self.good_drone_manager.deactivate_drone_camera()

    def update(self):
        """Update game state"""
        # Update camera (handles mouse look)
        self.camera.update()

        # If in drone camera mode, we might need to update camera position
        # (but drones are stationary, so camera position doesn't change)

        # Update enemy drones (they move)
        self.enemy_drone_manager.update()

        # Update good drones (stationary)
        self.good_drone_manager.update()

        # Update UI
        self.ui.update(
            camera_pos=(self.camera.x, self.camera.y, self.camera.z),
            camera_yaw=self.camera.yaw,
            camera_pitch=self.camera.pitch,
            flight_mode=self.camera.flight_mode,
            current_sector=self.get_current_sector(),
            fps=self.clock.get_fps()
        )

        self.frame_count += 1

    def get_current_sector(self):
        """Determine which sector the camera is in"""
        for sector in self.city_data['sectors']:
            distance = ((self.camera.x - sector.center_x) ** 2 +
                        (self.camera.y - sector.center_y) ** 2) ** 0.5
            if distance < sector.radius:
                return sector.name
        return "Unknown"

    def render(self):
        """Render the scene"""
        # Clear screen and render city
        self.renderer.render(self.screen, self.city_data, self.camera)

        # Render enemy drones (red, moving)
        self.enemy_drone_manager.render(self.screen, self.camera)

        # Render good drones (blue, stationary)
        self.good_drone_manager.render(self.screen, self.camera, show_labels=True)

        # Render UI
        self.ui.render(self.screen)

        # Update display
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()
        sys.exit()


def main():
    """Main entry point"""
    game = LahoreUrbanDefense(width=1600, height=900)
    game.run()


if __name__ == "__main__":
    main()