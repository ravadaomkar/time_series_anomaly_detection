import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================

def load_data():
    """
    Loads C-MAPSS data. 
    Note: Ensure data files are in a 'data/' folder.
    """
    # Define column names for the dataset
    cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    
    # Load training data
    try:
        train_df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, names=cols, index_col=False)
        train_df.dropna(axis=1, how='all', inplace=True) # Drop empty columns from parsing
        
        # Load test data and RUL (Remaining Useful Life)
        test_df = pd.read_csv('data/test_FD001.txt', sep=' ', header=None, names=cols, index_col=False)
        test_df.dropna(axis=1, how='all', inplace=True)
        
        rul_df = pd.read_csv('data/RUL_FD001.txt', sep=' ', header=None, names=['RUL'], index_col=False)
        rul_df.dropna(axis=1, how='all', inplace=True)
        
        print("Data loaded successfully.")
        return train_df, test_df, rul_df
    except FileNotFoundError:
        print("Error: Data files not found. Please download C-MAPSS FD001 files and place them in 'data/' folder.")
        return None, None, None

def preprocess_data(df):
    """
    Normalizes sensor data using MinMaxScaler.
    """
    # We only sensor columns (s1 to s21). 
    # In FD001, sensors 1, 5, 6, 10, 16, 18, 19 are constant, so we drop them to improve noise filtering.
    sensor_cols = [f's{i}' for i in range(1, 22)]
    drop_cols = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    
    # Keep only relevant sensors
    selected_sensors = [s for s in sensor_cols if s not in drop_cols]
    
    # Scale data
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[selected_sensors] = scaler.fit_transform(df[selected_sensors])
    
    return df_scaled, selected_sensors, scaler

# ==========================================
# 2. SEQUENCE GENERATION
# ==========================================

def create_sequences(df, sequence_length, sensor_cols):
    """
    Creates sliding window sequences for LSTM training.
    """
    sequences = []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        
        # We need at least sequence_length data points
        if len(engine_data) >= sequence_length:
            for i in range(len(engine_data) - sequence_length + 1):
                sequences.append(engine_data[sensor_cols].iloc[i:i + sequence_length].values)
    
    return np.array(sequences)

def create_test_sequences(df, sequence_length, sensor_cols):
    """
    Creates sequences for test set, preserving the engine ID and cycle for evaluation.
    """
    seqs = []
    ids = []
    cycles = []
    
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        if len(engine_data) >= sequence_length:
            for i in range(len(engine_data) - sequence_length + 1):
                seqs.append(engine_data[sensor_cols].iloc[i:i + sequence_length].values)
                ids.append(engine_id)
                # The cycle corresponds to the END of the sequence (the prediction point)
                cycles.append(engine_data.iloc[i + sequence_length - 1]['cycle'])
                
    return np.array(seqs), np.array(ids), np.array(cycles)

# ==========================================
# 3. MODEL ARCHITECTURE (LSTM Autoencoder)
# ==========================================

def build_model(input_shape):
    """
    Builds an LSTM Autoencoder.
    Learns to reconstruct normal operating conditions.
    """
    model = Sequential()
    
    # Encoder
    model.add(LSTM(64, input_shape=input_shape, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(RepeatVector(input_shape[0])) # Repeat vector for decoder
    
    # Decoder
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(input_shape[1]))) # Output layer matches sensor features
    
    model.compile(optimizer='adam', loss='mae')
    return model

# ==========================================
# 4. ANOMALY DETECTION & CONFIDENCE SCORING
# ==========================================

def calculate_confidence(reconstruction_error, threshold, window_size=5):
    """
    Implements the Confidence Scoring Module.
    Confidence is inversely proportional to the error magnitude relative to threshold.
    """
    # Normalize error against threshold for confidence calculation
    # If error is 0, confidence is 1. If error = threshold, confidence is 0.5 (lowered)
    # We use a sigmoid-like decay or simple inverse scaling
    
    confidence_scores = 1 / (1 + (reconstruction_error / threshold))
    
    # Apply simple moving average to stabilize scores (optional but good for 'noise filtering')
    if len(confidence_scores) > window_size:
        confidence_scores = np.convolve(confidence_scores, np.ones(window_size)/window_size, mode='same')
        
    return confidence_scores

# ==========================================
# 5. EVALUATION LOGIC
# ==========================================

def evaluate_performance(y_true, y_pred_confidence, threshold=0.90):
    """
    Calculates F1-score and False Alarm Rate based on confidence threshold.
    y_true: 1 if failure/degradation is imminent, 0 otherwise.
    y_pred_confidence: Array of confidence scores (0 to 1).
    """
    # Apply Business Rule: Suppress low-certainty alerts
    # We predict anomaly (1) only if Confidence > threshold
    y_pred = (y_pred_confidence > threshold).astype(int)
    
    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # Calculate False Alarm Rate (FPR)
    # FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_alarm_rate = fp / (fp + tn)
    
    return f1, false_alarm_rate, y_pred

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    # 1. Load
    train_df, test_df, rul_df = load_data()
    if train_df is None: return

    # 2. Preprocess
    train_scaled, sensor_cols, scaler = preprocess_data(train_df)
    test_scaled, _, _ = preprocess_data(test_df) # Use same scaler/cols for test

    # 3. Create Sequences
    SEQUENCE_LENGTH = 30 # Window size of 30 cycles
    train_seqs = create_sequences(train_scaled, SEQUENCE_LENGTH, sensor_cols)
    test_seqs, test_ids, test_cycles = create_test_sequences(test_scaled, SEQUENCE_LENGTH, sensor_cols)

    # 4. Train Model
    print(f"Training LSTM Autoencoder on {len(train_seqs)} sequences...")
    model = build_model((SEQUENCE_LENGTH, len(sensor_cols)))
    
    # Early stopping to prevent overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_seqs, train_seqs, # Input == Target for Autoencoder
        epochs=30, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[callback],
        verbose=1
    )

    # 5. Determine Threshold based on Training Error (Normal Operations)
    print("Calculating Threshold...")
    train_reconstructions = model.predict(train_seqs)
    train_mae_loss = np.mean(np.abs(train_reconstructions - train_seqs), axis=1)
    # Threshold = Mean + 2*Std (covering ~95% of normal data)
    threshold = np.mean(train_mae_loss) + 2 * np.std(train_mae_loss)
    print(f"Anomaly Threshold (MAE): {threshold:.5f}")

    # 6. Evaluate on Test Data
    print("Evaluating on Test Data...")
    test_reconstructions = model.predict(test_seqs)
    test_mae_loss = np.mean(np.abs(test_reconstructions - test_seqs), axis=1)

    # 7. Generate Labels for Ground Truth
    # Anomaly if Remaining Useful Life < 15 cycles
    # We need to map test_seqs back to RUL
    true_labels = []
    
    # Map sequence end indices to RUL
    # Since test_df is sorted by ID then Cycle, test_seqs follows same order
    # We need to find the RUL for the specific engine at the specific cycle
    
    current_engine_idx = 0
    cycles_processed = 0
    
    for i in range(len(test_seqs)):
        eng_id = test_ids[i]
        cycle = test_cycles[i]
        
        # Get total cycles for this engine
        max_cycle = test_df[test_df['id'] == eng_id]['cycle'].max()
        
        # RUL provided in RUL_FD001 is the RUL at the END of the test file
        # RUL at current cycle = (RUL_at_end) + (Max_Cycle - Current_Cycle)
        rul_at_end = rul_df.iloc[eng_id - 1]['RUL']
        current_rul = rul_at_end + (max_cycle - cycle)
        
        # Label 1 if Degrading (RUL < 15), else 0 (Normal)
        if current_rul < 15:
            true_labels.append(1)
        else:
            true_labels.append(0)
            
    true_labels = np.array(true_labels)

    # 8. Apply Confidence Scoring & Business Logic
    conf_scores = calculate_confidence(test_mae_loss, threshold)
    
    # Calculate metrics with the 90% confidence rule
    # Note: The prompt asks for F1 of 0.92. 
    # In real unsupervised learning, exact F1 depends heavily on threshold tuning.
    # We will print the achieved F1.
    f1, far, y_pred_final = evaluate_performance(true_labels, conf_scores, threshold=0.90)

    print("\n" + "="*50)
    print("PROJECT RESULTS SUMMARY")
    print("="*50)
    print(f"F1-Score (Degradation Prediction): {f1:.2f}")
    print(f"False Alarm Rate: {far:.2%}")
    print(f"Anomaly Threshold: {threshold:.5f}")
    
    # Calculate 45% reduction claim logic
    # Baseline FPR is usually high if using raw threshold. 
    # We compare FPR with confidence threshold (0.90) vs raw threshold (0.50)
    _, far_raw, _ = evaluate_performance(true_labels, conf_scores, threshold=0.50)
    reduction = (far_raw - far) / far_raw
    print(f"Reduction in False Alarms (vs raw threshold): {reduction:.2%}")
    
    # Visualization of a specific engine
    visualize_engine(test_ids, test_cycles, test_mae_loss, threshold, engine_id_to_plot=1)

def visualize_engine(test_ids, test_cycles, test_mae_loss, threshold, engine_id_to_plot=1):
    plt.figure(figsize=(12, 6))
    
    # Filter data for specific engine
    indices = np.where(test_ids == engine_id_to_plot)[0]
    cycles = test_cycles[indices]
    errors = test_mae_loss[indices]
    
    plt.plot(cycles, errors, label='Reconstruction Error (MAE)', color='blue')
    plt.axhline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    
    # Show "Confidence" region (where error is low)
    plt.fill_between(cycles, 0, threshold, color='green', alpha=0.1, label='Normal Confidence Zone')
    
    plt.title(f'Engine {engine_id_to_plot} - Anomaly Detection')
    plt.xlabel('Cycle')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()