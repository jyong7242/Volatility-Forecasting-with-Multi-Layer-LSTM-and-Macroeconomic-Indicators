# -*- coding: utf-8 -*-
"""
Created on Mar 17 2025

@author: Jinyan Yong
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import LSTM, Dense, Input
from keras_tuner import RandomSearch,Objective
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%% Data processing
# Setting the file path
folder_path = r"ProjectData"
csv_filenames = ['DGS1.csv', 'DGS2.csv', 'DGS10.csv', 'DGS30.csv', 'DTB3.csv', 'VIXCLS.csv']
model_results = []
# Read Data
df = pd.read_csv(os.path.join(folder_path, 'OxfordManRealizedVolatilityIndices.zip'))
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], utc=True)
df = df[df['Symbol'] == '.SPX'].dropna()  # S&P 500 data only
df['Date'] = df['Unnamed: 0'].dt.strftime('%Y-%m-%d')
df.set_index('Date', inplace=True)
df = df[['rv5']]
#print df val negative
print("There are: " + str(df[df['rv5'] < 0].shape[0]) + " negative values in the rv5 column")

#%% Define the parameters

# Merge other CSV data
for csv_filename in csv_filenames:
    file_path = os.path.join(folder_path, csv_filename)
    temp_df = pd.read_csv(file_path, parse_dates=['DATE'])
    temp_df['DATE'] = temp_df['DATE'].astype(str)
    temp_df.set_index('DATE', inplace=True)
    df = df.join(temp_df, how='left')
    df.columns = df.columns.str.lower()

# Data cleansing
df['rv5'] *= (100 ** 2)  
df.replace('.', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.interpolate(method='linear', inplace=True)
df.dropna(inplace=True)

#%% Exploratory Data Analysis (EDA)


# Plot rv5 time series
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['rv5'], label='rv5', color='black')
plt.title('Daily Realized Volatility (rv5) of S&P 500')
plt.xlabel('Date')
plt.ylabel('rv5')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of rv5
plt.figure(figsize=(6, 4))
sns.histplot(df['rv5'], bins=50, kde=True)
plt.title('Distribution of rv5')
plt.xlabel('rv5')
plt.tight_layout()
plt.show()

#%% Basic Data Overview
print("=== DataFrame Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
print(df.describe())

#%% **feature engineering**
# **Rolling Feature）for Har Model only 
df['rv5_5'] = df['rv5'].rolling(5).mean()
df['rv5_21'] = df['rv5'].rolling(21).mean()

# **Generate forecast targets**
df['rv5_f1'] = df['rv5'].shift(-1)  #
df.dropna(inplace=True)  #

# **Selection of variables**
model1_var = ['rv5', 'rv5_5', 'rv5_21'] # for HAR model only
model2_var = ['rv5'] # for preliminary study, only use rv5
model3_var = ['rv5', 'dgs1', 'dgs10', 'dgs2', 'dgs30', 'dtb3', 'vixcls']  # all variables
model4_var = ['rv5', 'dgs1', 'dgs10', 'dgs2', 'dgs30', 'dtb3']  # interst rates except vixcls
model5_var = ['rv5', 'vixcls']  # vixcls except vixcls interst rates
Y = ['rv5_f1']  # They all have the same target

model_vars = [model1_var, model2_var, model3_var, model4_var, model5_var]
datasets = ['model1', 'model2', 'model3', 'model4', 'model5']

# **Training-validation-test set segmentation**
splits = {}
for i, dataset in enumerate(datasets):
    splits[f"{dataset}_train"] = df.loc[:'2012', model_vars[i]].copy()
    splits[f"{dataset}_valid"] = df.loc['2013':'2016', model_vars[i]].copy()
    splits[f"{dataset}_test"] = df.loc['2017':, model_vars[i]].copy()

splits['Y_train'] = df.loc[:'2012', Y].copy()
splits['Y_valid'] = df.loc['2013':'2016', Y].copy()
splits['Y_test'] = df.loc['2017':, Y].copy()
# give HAR model a unstandardized dataset
model1_train, model1_valid, model1_test = splits['model1_train'], splits['model1_valid'], splits['model1_test']
Y_train, Y_valid, Y_test = splits['Y_train'], splits['Y_valid'], splits['Y_test']
X_train_har = model1_train.copy()
y_train_har = Y_train['rv5_f1'].copy()

X_valid_har = model1_valid.copy()
y_valid_har = Y_valid['rv5_f1'].copy()

X_test_har = model1_test.copy()
y_test_har = Y_test['rv5_f1'].copy()

# *Data standardization**
train_stdscaler = StandardScaler()
for dataset in datasets + ['Y']:
    splits[f"{dataset}_train"] = pd.DataFrame(
        train_stdscaler.fit_transform(splits[f"{dataset}_train"]),
        columns=splits[f"{dataset}_train"].columns,
        index=splits[f"{dataset}_train"].index
    )
    splits[f"{dataset}_valid"] = pd.DataFrame(
        train_stdscaler.transform(splits[f"{dataset}_valid"]),
        columns=splits[f"{dataset}_valid"].columns,
        index=splits[f"{dataset}_valid"].index
    )
    splits[f"{dataset}_test"] = pd.DataFrame(
        train_stdscaler.transform(splits[f"{dataset}_test"]),
        columns=splits[f"{dataset}_test"].columns,
        index=splits[f"{dataset}_test"].index
    )

# **Assign training, validation, and test datasets for model1, model2, and target variable Y**
model2_train, model2_valid, model2_test = splits['model2_train'], splits['model2_valid'], splits['model2_test']
model3_train, model3_valid, model3_test = splits['model3_train'], splits['model3_valid'], splits['model3_test']
model4_train, model4_valid, model4_test = splits['model4_train'], splits['model4_valid'], splits['model4_test']
model5_train, model5_valid, model5_test = splits['model5_train'], splits['model5_valid'], splits['model5_test']
Y_train, Y_valid, Y_test = splits['Y_train'], splits['Y_valid'], splits['Y_test']

#%% ========baseline model for HAR========
# baseline model for HAR with 2D input
har_model = LinearRegression()
har_model.fit(X_train_har, y_train_har)

y_pred_train = har_model.predict(X_train_har)
y_pred_valid = har_model.predict(X_valid_har)
y_pred_test = har_model.predict(X_test_har)

# Calculate MSE for train, validation, and test sets
har_mse_train = mean_squared_error(y_train_har, y_pred_train)
har_mse_valid = mean_squared_error(y_valid_har, y_pred_valid)
har_mse_test = mean_squared_error(y_test_har, y_pred_test)

print("HAR MSE - Train:", har_mse_train)
print("HAR MSE - Validation:", har_mse_valid)
print("HAR MSE - Test:", har_mse_test)

y_actual_all = pd.concat([y_train_har, y_valid_har, y_test_har])
y_pred_all = np.concatenate([y_pred_train, y_pred_valid, y_pred_test])


# Ensure index alignment
y_pred_series = pd.Series(y_pred_all.ravel(), index=y_actual_all.index)

# PLot
plt.figure(figsize=(10, 5))
plt.plot(y_actual_all.index, y_actual_all.values, label='Actual', color='black')
plt.plot(y_pred_series.index, y_pred_series.values, label='HAR Prediction', color='orange', linestyle='dashed')

plt.title("HAR Baseline Prediction vs Actual (Train + Valid + Test)")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
plt.ylabel("rv5")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%% ========helper function========
# # Inverse transform of predicted values (de-standardization)
def tx_helper_predict(pred, scaler):
    mean = scaler.mean_[0]
    std = np.sqrt(scaler.var_[0])
    return mean + std * pred

def tx_helper_mse(pred, tar, scaler, seq_length):
    v_pred = tx_helper_predict(pred, scaler)
    v_true = tar['rv5'][seq_length:].values
    return mean_squared_error(v_true, v_pred)

def plot_lstm_predictions(df,
                          model_train_df, model_valid_df, model_test_df,
                          pred_train, pred_valid, pred_test,
                          scaler,
                          sequence_length,
                          dataset_name, layers, nodes, lr, opt, act):
    plt.figure(figsize=(12, 5))

    # Actual values (from the original df)
    plt.plot(df.index, df['rv5'], label='Actual rv5', color='black')

    # Training set prediction
    plt.plot(model_train_df.index[sequence_length:],
             tx_helper_predict(pred_train, scaler),
             label='Train Prediction', linestyle='dashed', color='red')

    # Validation set prediction
    plt.plot(model_valid_df.index[sequence_length:],
             tx_helper_predict(pred_valid, scaler),
             label='Validation Prediction', linestyle='solid', color='blue')

    # Test set prediction
    plt.plot(model_test_df.index[sequence_length:],
             tx_helper_predict(pred_test, scaler),
             label='Test Prediction', linestyle='solid', color='green')

    # Set title
    plt.title(f"Multi-Layer LSTM {dataset_name} Predictions\n"
              f"(lags={sequence_length}, layers={layers}, nodes={nodes}, lr={lr}, opt={opt}, act={act})")
    plt.xlabel("Date")
    plt.ylabel("rv5")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
# === Helper for Loss Curve Plot ===
def plot_loss_curve(history, model_name, test_loss=None):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss', linestyle='dashed', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')

    if test_loss is not None:
        plt.axhline(y=test_loss, color='green', linestyle=':', label=f'Test Loss (standardized) = {test_loss:.4f}')

    plt.title(f"Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (mse)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tuner_random/loss_curve_{model_name}.png")
    plt.show()

#%% Hyperparameter tuning using Keras Tuner
class MyTuner(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        name = trial.trial_id
        # Dynamic hyperparameters
        seq_length = hp.Int('seq_length', 7, 70, step=7) # step method
        # seq_length = hp.Int('seq_length', [16, 32, 64]) # pick one method
        num_layers = hp.Int('num_layers',1,4)
        lr = hp.Choice('learning_rate', [1e-3,1e-4])
        activation = hp.Choice('activation', ['relu'])
        node_scale = hp.Choice('node_scale', [1.0,1.5]) # scale with seq_length
        nodes = int(seq_length * node_scale)
        optimizer_name = hp.Choice('optimizer', ['adam'])

        def create_dataset(X, y):
            return tf.keras.utils.timeseries_dataset_from_array(
                data=X,
                targets=y[seq_length:].values,
                sequence_length=seq_length,
                batch_size=128,
                shuffle=False)

        model = Sequential()
        model.add(Input(shape=(seq_length, self.input_dim)))
        for i in range(num_layers):
            layer_nodes = int(nodes / (2 ** i))  # Dynamic node scaling
            model.add(LSTM(units=layer_nodes,
                           return_sequences=(i < num_layers - 1),
                           activation=activation))
        model.add(Dense(1))
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])


        # Create datasets
        train_ds = create_dataset(self.X_train, self.y_train)
        valid_ds = create_dataset(self.X_valid, self.y_valid)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=10, restore_best_weights=True)

        model.fit(train_ds,
                  validation_data=valid_ds,
                  epochs=50,
                  callbacks=[early_stop],
                  verbose=0)

        val_mse = model.evaluate(valid_ds)[1]
        self.oracle.update_trial(trial.trial_id, {'val_mse': val_mse})

#%% Run the tuner
def run_random_tuner(model_name, X_train, X_valid, y_train, y_valid, input_dim):
    tuner = MyTuner(
        objective=Objective("val_mse", direction="min"),
        max_trials=5,
        directory="tuner_random",
        project_name=f"{model_name}_dynamic",
        overwrite=True
    )
    tuner.X_train = X_train
    tuner.X_valid = X_valid
    tuner.y_train = y_train
    tuner.y_valid = y_valid
    tuner.input_dim = input_dim
    tuner.search()
    return tuner

#%% Evaluate the tuned model
def evaluate_tuned_model(tuner, df, model_df_train, model_df_valid, model_df_test, y_train, y_valid, y_test, scaler, dataset_name):
    best_hp = tuner.get_best_hyperparameters(1)[0]
    seq_length = best_hp.get('seq_length')
    num_layers = best_hp.get('num_layers')
    lr = best_hp.get('learning_rate')
    activation = best_hp.get('activation')
    node_scale = best_hp.get('node_scale')
    nodes = int(seq_length * node_scale)
    optimizer_name = best_hp.get('optimizer')
    def create_dataset(X, y):
        return tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=y[seq_length:].values,
            sequence_length=seq_length,
            batch_size=128,
            shuffle=False)

    train_ds = create_dataset(model_df_train, y_train)
    valid_ds = create_dataset(model_df_valid, y_valid)
    test_ds = create_dataset(model_df_test, y_test)

    model = Sequential()
    model.add(Input(shape=(seq_length, model_df_train.shape[1])))
    for i in range(num_layers):
        layer_nodes = int(nodes / (2 ** i))
        model.add(LSTM(units=layer_nodes,
                       return_sequences=(i < num_layers - 1),
                       activation=activation))
    model.add(Dense(1))
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=10, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=valid_ds, epochs=50, verbose=0, callbacks=[early_stop])
    test_loss = model.evaluate(test_ds, verbose=0)[0]
    plot_loss_curve(history, dataset_name,test_loss)
    pred_train = model.predict(train_ds)
    pred_valid = model.predict(valid_ds)
    pred_test = model.predict(test_ds)

    mse_train_scaled = tx_helper_mse(pred_train, model_df_train, scaler, seq_length)
    mse_valid_scaled = tx_helper_mse(pred_valid, model_df_valid, scaler, seq_length)
    mse_test_scaled = tx_helper_mse(pred_test, model_df_test, scaler, seq_length)

    print(f"\n{dataset_name.upper()} Best Params: {best_hp.values}")
    print(f"{dataset_name.upper()} Final MSEs:\nTrain: {mse_train_scaled:.4f} | Valid: {mse_valid_scaled:.4f} | Test: {mse_test_scaled:.4f}")

    plot_lstm_predictions(df, model_df_train, model_df_valid, model_df_test,
                          pred_train, pred_valid, pred_test,
                          scaler, seq_length,
                          dataset_name, num_layers, nodes, lr, 'adam',activation)
    model_results.append({
        'model': dataset_name,
        'train_mse': mse_train_scaled,
        'valid_mse': mse_valid_scaled,
        'test_mse': mse_test_scaled,
        'seq_length': seq_length,
        'num_layers': num_layers,
        'learning_rate': lr,
        'activation': activation,
        'units': nodes,
        'optimizer': optimizer_name,
        'loss_function': 'mse'
    })

#%% Save the model
### === Model 2 ===
print("=== Model 2 ===")
tuner_model2 = run_random_tuner(
    model_name="model2",
    X_train=model2_train,
    X_valid=model2_valid,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    input_dim=model2_train.shape[1]
)

evaluate_tuned_model(
    tuner=tuner_model2,
    df=df,
    model_df_train=model2_train,
    model_df_valid=model2_valid,
    model_df_test=model2_test,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    y_test=Y_test['rv5_f1'],
    scaler=train_stdscaler,
    dataset_name="model2"
)


#%% === Model 3 ===
print("=== Model 3 ===")
tuner_model3 = run_random_tuner(
    model_name="model3",
    X_train=model3_train,
    X_valid=model3_valid,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    input_dim=model3_train.shape[1]
)

evaluate_tuned_model(
    tuner=tuner_model3,
    df=df,
    model_df_train=model3_train,
    model_df_valid=model3_valid,
    model_df_test=model3_test,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    y_test=Y_test['rv5_f1'],
    scaler=train_stdscaler,
    dataset_name="model3"
)


#%% === Model 4 ===
print("=== Model 4 ===")
tuner_model4 = run_random_tuner(
    model_name="model4",
    X_train=model4_train,
    X_valid=model4_valid,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    input_dim=model4_train.shape[1]
)

evaluate_tuned_model(
    tuner=tuner_model4,
    df=df,
    model_df_train=model4_train,
    model_df_valid=model4_valid,
    model_df_test=model4_test,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    y_test=Y_test['rv5_f1'],
    scaler=train_stdscaler,
    dataset_name="model4"
)

#%% === Model 5 ===
print("=== Model 5 ===")
tuner_model5 = run_random_tuner(
    model_name="model5",
    X_train=model5_train,
    X_valid=model5_valid,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    input_dim=model5_train.shape[1]
)
evaluate_tuned_model(
    tuner=tuner_model5,
    df=df,
    model_df_train=model5_train,
    model_df_valid=model5_valid,
    model_df_test=model5_test,
    y_train=Y_train['rv5_f1'],
    y_valid=Y_valid['rv5_f1'],
    y_test=Y_test['rv5_f1'],
    scaler=train_stdscaler,
    dataset_name="model5"
)
#%% final model comparison
results_df = pd.DataFrame(model_results)
print("\n=== Final Model Comparison Table ===")
print(results_df.to_string(index=False))
results_df.to_csv("tuner_random/final_model_results.csv", index=False)