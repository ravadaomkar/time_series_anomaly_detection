"""
Synthetic NASA C-MAPSS FD001 Data Generator
Mimics exact structure: 100 train engines, 100 test engines
21 sensors, 3 operational settings, single fault mode (HPC degradation)
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_cmapss_fd001(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Column names matching C-MAPSS exactly
    columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    # Sensors that are constant/near-constant in real FD001 (will be dropped)
    # sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
    # sensor_6 has minimal variation
    
    def generate_engine_trajectory(unit_id, max_cycles=None, is_train=True):
        """Generate realistic degradation trajectory"""
        if max_cycles is None:
            # Train: run to failure, ~128-362 cycles in real data
            max_cycles = np.random.randint(128, 363)
        else:
            # Test: truncated, add RUL later
            pass
            
        cycles = np.arange(1, max_cycles + 1)
        n_cycles = len(cycles)
        
        # Operational settings (constant for FD001 - sea level conditions)
        op1 = np.full(n_cycles, 0.0)  # Altitude ~0
        op2 = np.full(n_cycles, 0.0)  # Mach ~0
        op3 = np.full(n_cycles, 100.0)  # Throttle ~100%
        
        # Base sensor values (healthy state)
        base_sensors = {
            1: 518.67,   # T2 - constant
            2: 642.36,   # T24
            3: 1587.70,  # T30
            4: 1400.46,  # T50
            5: 14.62,    # P2 - constant
            6: 21.61,    # Ps30 - minimal variation
            7: 553.91,   # phi
            8: 2388.05,  # NRf
            9: 9050.17,  # NRc
            10: 1.30,    # BPR - constant
            11: 47.20,   # farB
            12: 521.31,  # htBleed
            13: 2388.08, # Nf_dmd
            14: 8133.52, # W31
            15: 8.3629,  # W32
            16: 397.00,  # T48 - constant
            17: 2388.00, # T50
            18: 38.90,   # P50 - constant
            19: 23.2884, # P30 - constant
            20: 43.12,   # Nf
            21: 38.14    # 
        }
        
        # Degradation profile: starts healthy, degrades after certain point
        health_threshold = int(0.3 * n_cycles)  # Degradation starts at 30% of life
        
        data = []
        for i, cycle in enumerate(cycles):
            # Degradation factor: 0 = healthy, 1 = failed
            if i < health_threshold:
                degradation = 0
            else:
                degradation = ((i - health_threshold) / (n_cycles - health_threshold)) ** 2
            
            row = [unit_id, cycle, op1[i], op2[i], op3[i]]
            
            for s in range(1, 22):
                base = base_sensors[s]
                noise = np.random.normal(0, base * 0.005)  # 0.5% noise
                
                # Apply degradation to relevant sensors
                if s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 20, 21]:
                    # These sensors drift with degradation
                    drift = degradation * base * np.random.uniform(0.02, 0.15)
                    # Direction: some increase, some decrease
                    if s in [2, 3, 4, 11]:  # Temperatures increase
                        value = base + drift + noise
                    else:  # Others decrease or mixed
                        value = base - drift * np.random.uniform(0.5, 1.5) + noise
                else:
                    # Constant sensors with just noise
                    value = base + noise
                
                row.append(value)
            
            data.append(row)
        
        return pd.DataFrame(data, columns=columns)
    
    # Generate TRAIN data (100 engines, run to failure)
    print("Generating training data...")
    train_dfs = []
    train_ruls = {}  # For verification: actual max cycle per engine
    
    for unit in range(1, 101):
        df = generate_engine_trajectory(unit, is_train=True)
        train_dfs.append(df)
        train_ruls[unit] = df['time_cycles'].max()
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df.to_csv(f"{output_dir}/train_FD001.txt", sep=' ', index=False, header=False)
    print(f"Train: {len(train_df)} records, engines: {train_df['unit_id'].nunique()}")
    
    # Generate TEST data (100 engines, truncated)
    print("Generating test data...")
    test_dfs = []
    test_ruls = []
    
    for unit in range(1, 101):
        # Full trajectory length
        full_length = np.random.randint(128, 363)
        # Truncation point (leave some RUL)
        truncate_at = np.random.randint(int(0.5 * full_length), int(0.85 * full_length))
        actual_rul = full_length - truncate_at
        
        df = generate_engine_trajectory(unit, max_cycles=truncate_at, is_train=False)
        test_dfs.append(df)
        test_ruls.append(actual_rul)
    
    test_df = pd.concat(test_dfs, ignore_index=True)
    test_df.to_csv(f"{output_dir}/test_FD001.txt", sep=' ', index=False, header=False)
    
    # Save RUL file
    rul_df = pd.DataFrame({'RUL': test_ruls})
    rul_df.to_csv(f"{output_dir}/RUL_FD001.txt", sep=' ', index=False, header=False)
    
    print(f"Test: {len(test_df)} records, engines: {test_df['unit_id'].nunique()}")
    print(f"RUL range: {min(test_ruls)} - {max(test_ruls)}")
    
    return train_df, test_df, rul_df

if __name__ == "__main__":
    generate_cmapss_fd001()