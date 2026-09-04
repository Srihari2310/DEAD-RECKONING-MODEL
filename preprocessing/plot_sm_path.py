import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define project directories
PROJECT_DIR = r"D:\Workspace\DEAD RECKONING MODEL"
S_DIR = os.path.join(PROJECT_DIR, "S-Dataset")

def load_file(path):
    """Load CSV with BOM/Latin1 encoding fallback and strip header spaces."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_csv(path, encoding='latin1')
    
    df.columns = df.columns.str.strip()
    return df

def main():
    # 1. Find and load S-M.csv
    pattern = os.path.join(S_DIR, "**", "S-M.*")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        print("Error: Could not find S-M file in S-Dataset folder.")
        return
    
    s_path = matches[0]
    print(f"Loading S-M dataset from: {s_path}")
    df = load_file(s_path)
    
    # 2. Extract Acceleration, Gravity, and Gyroscope
    accel_cols = ['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)', 'ACCELEROMETER Z (m/s²)']
    gravity_cols = ['GRAVITY X (m/s²)', 'GRAVITY Y (m/s²)', 'GRAVITY Z (m/s²)']
    gyro_yaw_col = 'GYROSCOPE Yaw (rad/s)'  # Yaw rotation rate around Z-axis

    # Fallback column search if names differ slightly
    missing = [c for c in accel_cols + gravity_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        print("Available columns:", list(df.columns))
        return

    # 3. Calculate Linear Acceleration (Accelerometer - Gravity)
    lin_accel_x = df['ACCELEROMETER X (m/s²)'].values - df['GRAVITY X (m/s²)'].values
    lin_accel_y = df['ACCELEROMETER Y (m/s²)'].values - df['GRAVITY Y (m/s²)'].values
    
    # Gyroscope Yaw rate (rad/s)
    gyro_yaw = df[gyro_yaw_col].values if gyro_yaw_col in df.columns else np.zeros(len(df))

    # 4. Dead Reckoning Kinematics
    dt = 0.1  # 10 Hz sampling rate (0.1 seconds per row)
    num_samples = len(df)
    
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    v_forward = 0.0
    yaw = 0.0

    # Accumulate velocity, heading, and 2D positions
    for i in range(1, num_samples):
        # Update Heading (Yaw) using Gyroscope
        yaw += gyro_yaw[i] * dt
        
        # In phone frame, Y is usually forward/longitudinal acceleration
        a_forward = lin_accel_y[i]
        
        # Integrate acceleration to velocity (with non-negative speed clamp)
        v_forward = max(0.0, v_forward + a_forward * dt)
        
        # Project forward displacement into 2D World Frame (X, Y)
        dx = v_forward * np.sin(yaw) * dt
        dy = v_forward * np.cos(yaw) * dt
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy

    # 5. Plot the 2D Path
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, label='Reconstructed Path (S-M IMU)', color='blue', linewidth=1.5)
    
    # Mark start and end
    plt.scatter(x[0], y[0], color='green', s=120, zorder=5, label='Start Point (0,0)')
    plt.scatter(x[-1], y[-1], color='red', s=120, zorder=5, label='End Point')

    plt.title('2D Vehicle Trajectory Reconstructed from S-M IMU Data', fontsize=12)
    plt.xlabel('East Displacement X (meters)', fontsize=10)
    plt.ylabel('North Displacement Y (meters)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend(loc='best')
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()