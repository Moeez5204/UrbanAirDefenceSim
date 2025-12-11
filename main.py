import os
import subprocess
import sys

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    subprocess.run([sys.executable, os.path.join(current_dir, 'CityModelReconstruct.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'AlphaComplexConstruction.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'PersistentHomolgyAnalysis.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'StrategicFeatureExtraction.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'TopologyEnhancedKalmanFilter.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'IMM_Filter.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'LTSM.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'ThreatsScoreCalculator.py')])
    subprocess.run([sys.executable, os.path.join(current_dir, 'AdaptiveSectorDefenseAllocation.py')])



if __name__ == "__main__":
    main()