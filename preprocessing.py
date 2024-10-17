import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def onehotencode(y_synth):
  y_synth = np.array(y_synth)
  start_end_ids = y_synth[:,:2]

  patterns = y_synth[:,2]

  mapping = {
      0: ([0, 0, 0, 0, 1, 0]),
      1: ([0, 0, 0, 1, 0, 0]),
      2: ([0, 0, 1, 0, 0, 0]),
      3: ([0, 1, 0, 0, 0, 0]),
      4: ([1, 0, 0, 0, 0, 0]),
      5: ([0, 0, 0, 1, 0, 1]),
      6: ([0, 0, 1, 0, 0, 1]),
      7: ([0, 1, 0, 0, 0, 1]),
      8: ([1, 0, 0, 0, 0, 1]),
  }
  mapped_patterns = np.array([mapping[pattern] for pattern in patterns])
  pattern_type = mapped_patterns[:,:-1]
  pattern_rising = mapped_patterns[:,-1]

  return start_end_ids, pattern_type, pattern_rising

def preprocess(X, y):

  start_end_ids, pattern_type, pattern_rising  = onehotencode(y)

  X_train_processed=[]
  scale={}
  for i in range(len(X)):
    open_scaler_i = StandardScaler()
    X_train_processed.append(open_scaler_i.fit_transform(X[i]))
    scale[f"open_scaler_{len(X[i])}"] = open_scaler_i

  maxlen = 450

  X_train_preprocessed = pad_sequences(X_train_processed, maxlen = maxlen, dtype='float32', padding='post', value=-1)

  X_train_preprocessed = tf.convert_to_tensor(X_train_preprocessed, np.float32)
  y_train_pattern_type = tf.convert_to_tensor(pattern_type, np.int16)
  y_train_pattern_rising = tf.convert_to_tensor(pattern_rising, np.int16)
  y_train_dates = tf.convert_to_tensor(start_end_ids, np.int16)

  return X_train_preprocessed, y_train_pattern_type, y_train_pattern_rising, y_train_dates

def preprocess_X(X):

  assert len(X)<451, "The maximum days possible are 450"

  open_scaler = StandardScaler()
  X_train_processed = open_scaler.fit_transform(X)

  maxlen = 450

  X_train_preprocessed = pad_sequences([X_train_processed], maxlen = maxlen, dtype='float32', padding='post', value=-1)

  if len(X_train_preprocessed.shape) == 2:
    X_train_preprocessed = tf.convert_to_tensor([X_train_preprocessed], np.float32)
  elif len(X_train_preprocessed.shape) == 3:
    X_train_preprocessed = tf.convert_to_tensor(X_train_preprocessed, np.float32)
  else:
    return "Errorrrrrr"

  return X_train_preprocessed, open_scaler