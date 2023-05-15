import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.stats import mode
import features as feat
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import butter, filtfilt

sample_rate = 100

# Load the dataset
data_folder = 'data'
df = pd.DataFrame()

# iterate over csvs
print('> Loading data from all CSVs . . .')
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        current_df = pd.read_csv(os.path.join(data_folder, file_name))
        
        # Get fall indices based on name, remove ".csv"
        start_index, end_index = map(int, file_name[:-4].split('-'))

        # Assign labels based on the indices
        current_df['Activity'] = 0
        current_df.loc[start_index:end_index, 'Activity'] = 1
        
        # Add curr dataframe to global one
        df = pd.concat([df, current_df], ignore_index=True)

df['magnitude'] = (df['x']**2 + df['y']**2 + df['z']**2)**0.5

def butter_lowpass(cutoff, fs):
    order = 4
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs)
    y = filtfilt(b, a, data)
    return y

# Use cutoff of 3 to get rid of high frequency noise
cutoff = 3.0

# Apply lowpass filter
print("> Applying lowpass filter . . .")
df['x'] = butter_lowpass_filter(df['x'], cutoff, sample_rate)
df['y'] = butter_lowpass_filter(df['y'], cutoff, sample_rate)
df['z'] = butter_lowpass_filter(df['z'], cutoff, sample_rate)
df['magnitude'] = butter_lowpass_filter(df['magnitude'], cutoff, sample_rate)

# Check for NaN values
df.isnull().sum()
# Drop rows with NaN values
df = df.dropna()

features = pd.DataFrame()
window_size = 100

# Extract features for each window
print('> Extracting features for each window . . .')
for i in range(0, len(df)-window_size, window_size):
    X = df['x'].values[i: i + window_size]
    Y = df['y'].values[i: i + window_size]
    Z = df['z'].values[i: i + window_size]
    Mag = df['magnitude'].values[i: i + window_size]
    activities = df['Activity'].values[i: i + window_size]
    
    # Get the mode of the activities in the window
    activity = mode(activities, keepdims=False)[0]
    
    # Create a new row as a DataFrame and concatenate to features DataFrame
    new_row = pd.DataFrame([{
        'X_mean': feat.calculate_mean(X),
        'X_std': feat.calculate_variance(X),
        'X_min': feat.calculate_min(X),
        'X_max': feat.calculate_max(X),
        'Y_mean': feat.calculate_mean(Y),
        'Y_std': feat.calculate_variance(Y),
        'Y_min': feat.calculate_min(Y),
        'Y_max': feat.calculate_max(Y),
        'Z_mean': feat.calculate_mean(Z),
        'Z_std': feat.calculate_variance(Z),
        'Z_min': feat.calculate_min(Z),
        'Z_max': feat.calculate_max(Z),
        'Mag_mean': feat.calculate_mean(Mag),
        'Mag_std': feat.calculate_variance(Mag),
        'Mag_min': feat.calculate_min(Mag),
        'Mag_max': feat.calculate_max(Mag),
        'X_median': feat.calculate_median(X),
        'Y_median': feat.calculate_median(Y),
        'Z_median': feat.calculate_median(Z),
        'Mag_median': feat.calculate_median(Mag),
        'X_fft': feat.compute_fft_features(X),
        'Y_fft': feat.compute_fft_features(Y),
        'Z_fft': feat.compute_fft_features(Z),
        'Mag_fft': feat.compute_fft_features(Mag),
        'X_entropy': feat.compute_ent_features(X),
        'Y_entropy': feat.compute_ent_features(Y),
        'Z_entropy': feat.compute_ent_features(Z),
        'Mag_entropy': feat.compute_ent_features(Mag),
        'X_peaks': feat.compute_peak_features(X),
        'Y_peaks': feat.compute_peak_features(Y),
        'Z_peaks': feat.compute_peak_features(Z),
        'Mag_peaks': feat.compute_peak_features(Mag),
        'Activity': activity
    }])
    features = pd.concat([features, new_row], ignore_index=True)

# Split the data
print('> Splitting data into vectors and labels . . .')
X = features.drop('Activity', axis=1)
y = features['Activity']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import sys

tree = None
cv = KFold(n_splits=10, shuffle=True, random_state=None)

total_accuracy = 0
total_precision = 0
total_recall = 0

print('> Testing decision tree classifier . . .')
for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    tree.fit(X_train, y_train)
    # print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = tree.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred)
    # print(conf)

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # print("Precision = ", precision)
    # print("Recall = ", recall)
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall

print("The average accuracy is {}".format(total_accuracy / 10.0))
print("The average precision is {}".format(total_precision / 10.0)) 
print("The average recall is {}".format(total_recall / 10.0)) 

# print("> Training decision tree classifier on entire dataset...")
# tree.fit(X, y)

forest = None
cv = KFold(n_splits=10, shuffle=True, random_state=None)

total_accuracy = 0
total_precision = 0
total_recall = 0

print('Testing random forest classifier . . .')
for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    forest = RandomForestClassifier(criterion="entropy", max_depth=3)
    # print("Fold {} : Training random forest classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    forest.fit(X_train, y_train)
    # print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = forest.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred)
    # print(conf)

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # print("Precision = ", precision)
    # print("Recall = ", recall)
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall

print("The average accuracy is {}".format(total_accuracy / 10.0))
print("The average precision is {}".format(total_precision / 10.0)) 
print("The average recall is {}".format(total_recall / 10.0)) 
