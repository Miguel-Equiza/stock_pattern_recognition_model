from synthetic_data_generator import *
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import *

def gen_synthetic_x_y(l=round(0.4*50),pattern=["rising_wedge","falling_wedge","double_top","double_bottom"]):
  X=[]
  y=[]
  pat={"rising_wedge":1,
          "falling_wedge":2,
          "double_top":3,
          "double_bottom":4,
          "ascending_triangle":5,
          "descending_triangle":6,
          "h&s_top":7,
          "h&s_bottom":8
      }
  for i in range(l):

    func={"rising_wedge":rising_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "falling_wedge":falling_wedge(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "double_top":double_top(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "double_bottom":double_bottom(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "ascending_triangle":ascending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "descending_triangle":descending_triangle(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "h&s_top":head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0),
        "h&s_bottom":inv_head_shoulders(n=500, mu=3, sigma=5, h=1, noise_level=2, ar1 = 0.95, ar2 = 0)
    }

    date, ope, hig, low, close, start, end = func[pattern]
    ope=np.array(ope)
    hig=np.array(hig)
    low = np.array(low)
    close = np.array(close)

    X.append(np.column_stack((ope, hig, low, close)))
    y.append((start, end, pat[pattern]))


  return X, y

def get_synthetic_data():

  amount={"rising_wedge":556,
      "falling_wedge":268,
      "double_top": 393,
      "double_bottom": 348,
      "ascending_triangle":86,
      "descending_triangle":98,
      "h&s_top":86,
      "h&s_bottom":43}

  X_ris_wedg, y_ris_wedg = gen_synthetic_x_y(l=round(amount["rising_wedge"]*0.4), pattern="rising_wedge")
  X_fal_wedg, y_fal_wedg = gen_synthetic_x_y(l=round(amount["falling_wedge"]*0.4), pattern="falling_wedge")
  X_d_top, y_d_top = gen_synthetic_x_y(l=round(amount["double_top"]*0.4), pattern="double_top")
  X_d_bottom, y_d_bottom = gen_synthetic_x_y(l=round(amount["double_bottom"]*0.4), pattern="double_bottom")
  X_ris_wedg, y_ris_wedg = gen_synthetic_x_y(l=round(amount["ascending_triangle"]), pattern="ascending_triangle")
  X_fal_wedg, y_fal_wedg = gen_synthetic_x_y(l=round(amount["descending_triangle"]), pattern="descending_triangle")
  X_d_top, y_d_top = gen_synthetic_x_y(l=round(amount["h&s_top"]), pattern="h&s_top")
  X_d_bottom, y_d_bottom = gen_synthetic_x_y(l=round(amount["h&s_bottom"]), pattern="h&s_bottom")

  X_synthetic = X_ris_wedg + X_fal_wedg + X_d_top + X_d_bottom + X_ris_wedg + X_fal_wedg + X_d_top + X_d_bottom
  y_synthetic = y_ris_wedg + y_fal_wedg + y_d_top + y_d_bottom + y_ris_wedg + y_fal_wedg + y_d_top + y_d_bottom

  return X_synthetic, y_synthetic



def upside_down(Xs, ys):
  new_Xs, new_ys = [], []

  inverse_patterns = {
      1: 2,
      2: 1,
      3: 4,
      4: 3,
      5:6,
      6:5,
      7:8,
      8:7
  }
  for X, y in zip(Xs, ys):
    start, end, pattern = y
    if pattern != 0:
      new_pattern = inverse_patterns[pattern]
      pattern_max_0 = X - np.max(X[start:end])
      pattern_upside_down = -pattern_max_0
      new_Xs.append(pattern_upside_down)
      new_ys.append((start, end, new_pattern))

  X_whole = new_Xs + Xs
  y_whole = new_ys + ys
  return X_whole, y_whole

def data_augmentation(Xs: list[np.ndarray], ys: list[tuple[int, int, int]]):

  new_Xs, new_ys = [], []

  for X, y in zip(Xs, ys):
    size = len(X)
    start, end, pattern = y

    if pattern > 4:
      n = 10
    else:
      n = 1

    start_margin, end_margin = round(start * (2/3)), round((size - end )*(2/3))
    if start_margin > 4 and end_margin > 4:

      for _ in range(n):
        margin_left = np.random.randint(4, start_margin)
        margin_right = np.random.randint(4, end_margin)
        new_Xs.append(X[start - margin_left : end + margin_right])
        new_ys.append((margin_left, margin_left + end - start, pattern))


    if start > 9: # then the size is enough to have a time-series before
      new_Xs.append(X[:start])
      new_ys.append((-1,-1,0))

    if size - end > 9:
      new_Xs.append(X[end:])
      new_ys.append((-1,-1,0))

  return new_Xs, new_ys

def augmentate(X, y):
    X_augmented, y_augmented = data_augmentation(X, y)
    X_all, y_all = upside_down(X_augmented, y_augmented)

    return X_all, y_all

def get_real_X_y(pattern_list = ["rising_wedge", "falling_wedge", "double_top", "double_bottom", "ascending_triangle","descending_triangle","h&s_top","h&s_bottom"]):

  X=[]
  y=[]

  for pattern in pattern_list:
    directory_path = f'data/patterns/{pattern}'
    for root, dirs, files in os.walk(directory_path):
      for filename in files:
        df = pd.read_csv(f"{directory_path}/{filename}")
        X.append(np.array(df[["Open", "High", "Low", "Close"]]))
        y.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))

  return X, y

def get_data(synth=True):
  X, y = get_real_X_y(pattern_list = ["rising_wedge", "falling_wedge", "double_top", "double_bottom", "ascending_triangle","descending_triangle","h&s_top","h&s_bottom"])

  if synth:

    X_synthetic, y_synthetic = get_synthetic_data()

    X = X + X_synthetic
    y = y + y_synthetic

  X_all, y_all = augmentate(X, y)

  X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2)
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

  X_train_preprocessed, y_train_pattern_type, y_train_pattern_rising, y_train_dates = preprocess(X_train, y_train)
  X_test_preprocessed, y_test_pattern_type, y_test_pattern_rising, y_test_dates = preprocess(X_test, y_test)
  X_val_preprocessed, y_val_pattern_type, y_val_pattern_rising, y_val_dates = preprocess(X_val, y_val)


  return X_train_preprocessed, y_train_pattern_type, y_train_pattern_rising, y_train_dates, X_test_preprocessed, y_test_pattern_type, y_test_pattern_rising, y_test_dates, X_val_preprocessed, y_val_pattern_type, y_val_pattern_rising, y_val_dates
