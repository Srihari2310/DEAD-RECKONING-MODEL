import numpy as np
import pandas as pd

# import your Step 1 loading + alignment functions
from step1_load_verify_M import (
    find_trip_file, load_file, find_col, align_shift, S_DIR, V_DIR
)

def main():
    s_path = find_trip_file(S_DIR, "S-M")
    v_path = find_trip_file(V_DIR, "V-M")
    s_df = load_file(s_path); s_df.columns = [c.strip() for c in s_df.columns]
    v_df = load_file(v_path); v_df.columns = [c.strip() for c in v_df.columns]
    s_df, v_df = align_shift(s_df, v_df, shift=23)  # confirmed shift from Step 1

    accel_cols = [find_col(s_df, ["accelerometer", a]) for a in ["x", "y", "z"]]
    grav_cols  = [find_col(s_df, ["gravity", a]) for a in ["x", "y", "z"]]
    heading_col = find_col(v_df, ["heading"])
    speed_col   = find_col(v_df, ["velocity"])

    accel = s_df[accel_cols].to_numpy()
    grav  = s_df[grav_cols].to_numpy()
    linear_accel = accel - grav   # Step 3 of your architecture: gravity removed

    # --- Static leveling: average gravity direction over whole trip ---
    g_mean = grav.mean(axis=0)
    g_mean = g_mean / np.linalg.norm(g_mean)
    print("Gravity (down) direction:", g_mean)

    # --- Dynamic yaw: use straight-line, >15km/h segments ---
    speed = v_df[speed_col].to_numpy()
    heading = np.radians(v_df[heading_col].to_numpy())
    heading_rate = np.abs(np.gradient(np.unwrap(heading)))
    straight_mask = (speed > 15) & (heading_rate < 0.01)  # slow-changing heading = straight

    print(f"Straight-line samples used for yaw calib: {straight_mask.sum()} / {len(speed)}")

    fwd_ref = np.stack([np.cos(heading[straight_mask]), np.sin(heading[straight_mask])], axis=1)
    phone_xy = linear_accel[straight_mask][:, :2]

    # least-squares: find rotation angle aligning phone_xy to fwd_ref
    num = np.sum(phone_xy[:, 0]*fwd_ref[:, 1] - phone_xy[:, 1]*fwd_ref[:, 0])
    den = np.sum(phone_xy[:, 0]*fwd_ref[:, 0] + phone_xy[:, 1]*fwd_ref[:, 1])
    yaw_offset = np.arctan2(num, den)
    print(f"Estimated phone mounting yaw offset: {np.degrees(yaw_offset):.2f} degrees")

    # --- Build full 3x3 rotation matrix (levels + de-yaws) ---
    down = g_mean
    arbitrary = np.array([1, 0, 0]) if abs(down[0]) < 0.9 else np.array([0, 1, 0])
    right = np.cross(down, arbitrary); right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    R_level = np.vstack([forward, right, down])  # phone -> leveled frame

    cz, sz = np.cos(-yaw_offset), np.sin(-yaw_offset)
    R_yaw = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_final = R_yaw @ R_level  # phone -> vehicle frame

    vehicle_accel = linear_accel @ R_final.T
    s_df["accel_forward"]  = vehicle_accel[:, 0]
    s_df["accel_lateral"]  = vehicle_accel[:, 1]
    s_df["accel_vertical"] = vehicle_accel[:, 2]

    print("\nVehicle-frame accel stats:")
    print(s_df[["accel_forward","accel_lateral","accel_vertical"]].describe())

    import os
    os.makedirs("preprocessing/output", exist_ok=True)
    s_df.to_csv("preprocessing/output/S-M_calibrated.csv", index=False)
    v_df.to_csv("preprocessing/output/V-M_aligned.csv", index=False)
    print("\nSaved calibrated S-M and aligned V-M to preprocessing/output/")

if __name__ == "__main__":
    main()