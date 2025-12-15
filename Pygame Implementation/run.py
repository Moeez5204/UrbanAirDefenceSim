# run.py - Simple runner

from main import LahoreUrbanDefense

if __name__ == "__main__":
    print("=" * 60)
    print("LAHORE URBAN DEFENSE - WITH DRONE CAMERA BUTTONS")
    print("=" * 60)
    print("Features:")
    print("  3 RED Enemy Drones - Moving randomly")
    print("  3 BLUE Good Drones - Stationary")
    print("  6 Fixed City Cameras (Buttons 1-6)")
    print("  3 Drone Cameras (Buttons 7-9)")
    print("  Click buttons on right side for all cameras")
    print("=" * 60)
    print("\nPress 7, 8, or 9 for drone camera views")
    print("Or click the blue drone camera buttons")
    print("=" * 60)

    game = LahoreUrbanDefense(width=1600, height=900)
    game.run()