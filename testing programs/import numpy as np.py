import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATE SYNTHETIC SENSOR DATA (10 Hz)
# ==========================================
dt = 0.1  # 10 Hz sampling rate (0.1s per frame)
t = np.arange(0, 60, dt)  # 60-second trip
num_samples = len(t)

# Standard Earth gravity (Z-axis)
gravity_x = np.zeros(num_samples)
gravity_y = np.zeros(num_samples)
gravity_z = np.full(num_samples, 9.81)

# Simulate accelerometer readings (m/s²)
# Vehicle accelerates forward for 20s, turns left for 20s, then drives straight
accel_x = np.zeros(num_samples)  # Lateral (Side-to-side)
accel_y = np.zeros(num_samples)  # Longitudinal (Forward/Backward)
accel_z = np.full(num_samples, 9.81)  # Vertical (Gravity + bumps)

# Segment 1 (0 to 20s): Forward acceleration
accel_y[0:200] = 0.8  

# Segment 2 (20s to 40s): Constant speed forward + Left turn (Lateral Accel)
accel_y[200:400] = 0.1
accel_x[200:400] = -0.5  # Negative X = Leftward force

# Segment 3 (40s to 60s): Straight drive + Braking
accel_y[400:600] = -0.4

# Add small realistic sensor noise
np.random.seed(42)
accel_x += np.random.normal(0, 0.05, num_samples)
accel_y += np.random.normal(0, 0.05, num_samples)

# ==========================================
# 2. ISOLATE LINEAR ACCELERATION
# ==========================================
# Subtract gravity vector to isolate motion acceleration
lin_accel_x = accel_x - gravity_x
lin_accel_y = accel_y - gravity_y
lin_accel_z = accel_z - gravity_z

# ==========================================
# 3. DEAD RECKONING KINEMATICS (2D PATH)
# ==========================================
# Initialize position, velocity, and orientation (yaw angle)
x, y = np.zeros(num_samples), np.zeros(num_samples)
vx, vy = 0.0, 0.0
yaw = 0.0  # Heading angle in radians

for i in range(1, num_samples):
    # Forward acceleration in car's body frame
    a_forward = lin_accel_y[i]
    a_lateral = lin_accel_x[i]

    # Simple turning estimation: yaw rate derived from lateral force
    # (In real data, this comes from Gyroscope Yaw)
    yaw_rate = -a_lateral * 0.1  
    yaw += yaw_rate * dt

    # Forward velocity integration
    v_forward = max(0.0, vy + a_forward * dt)  # Non-holonomic: car doesn't fly/slide backward easily
    vy = v_forward

    # Convert body-frame displacement into world (X, Y) coordinates
    dx = v_forward * np.sin(yaw) * dt
    dy = v_forward * np.cos(yaw) * dt

    # Accumulate 2D trajectory position
    x[i] = x[i-1] + dx
    y[i] = y[i-1] + dy

# ==========================================
# 4. PLOT 2D PATH WITH MATPLOTLIB
# ==========================================
plt.figure(figsize=(8, 8))
plt.plot(x, y, color='blue', linewidth=2, label='Reconstructed Vehicle Path')

# Mark start and end points
plt.scatter(x[0], y[0], color='green', s=100, zorder=5, label='Start Point (0,0)')
plt.scatter(x[-1], y[-1], color='red', s=100, zorder=5, label='End Point')

plt.title('2D Vehicle Path Reconstruction from Acceleration & Gravity', fontsize=12)
plt.xlabel('East Displacement X (meters)', fontsize=10)
plt.ylabel('North Displacement Y (meters)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')  # Keep 1:1 aspect ratio so curves look natural
plt.legend(loc='best')

plt.show()