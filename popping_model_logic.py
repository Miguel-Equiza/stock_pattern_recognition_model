import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import math
import ipywidgets as widgets
import random
import pandas as pd
from preprocessing import *
from candle_types_detector import *

candle_types = ['doji',
       'o_marubozu', 'gravestone', 'dragonfly_doji', 'hammer', 'c_marubozu',
       'bear_engulfing', 'bull_engulfing', 'tweezer_bottom', 'tweezer_top',
       'morning_star', 'evening_star', 'three_inside_up', 'three_inside_down',
       'three_black_crows', 'three_white_soldiers']

def create_candlestick_chart(fig, df):
  arrow_annotation=[]

  df = df[df.iloc[:, 5:].any(axis=1)]
  df['candle'] = df.apply(lambda row: ' '.join([col for col in df.columns[5:] if row[col]]), axis=1)
  _ = [arrow_annotation.append({'x': row['date'],'y': row['high'],'text': row["candle"],'showarrow': True,'arrowhead': 2,'ax': 0,'ay': -40}) for index, row in df.iterrows()]

  with fig.batch_update():
    fig.layout.annotations = arrow_annotation
    fig.update_layout(xaxis_rangeslider_visible=False)

  return fig

def divide_fragments(one_data_fragment, start, end):
    fragment_before_pattern = one_data_fragment[0,:start]
    fragment_before_pattern = tf.convert_to_tensor([fragment_before_pattern], np.float32)
    fragment_before_pattern = pad_sequences(fragment_before_pattern, maxlen = 450, dtype='float32', padding='post', value=-1)

    fragment_after_pattern = one_data_fragment[0,end:]
    fragment_after_pattern = tf.convert_to_tensor([fragment_after_pattern], np.float32)
    fragment_after_pattern = pad_sequences(fragment_after_pattern, maxlen = 450, dtype='float32', padding='post', value=-1)

    return tf.convert_to_tensor([fragment_before_pattern, fragment_after_pattern], np.float32)

def popping_pattern(data_preprocessed, pretrained_model_p, pretrained_model_d, scaler):

    ends_index=[0]

    data_fragments = tf.convert_to_tensor([data_preprocessed], np.float32)

    n_original_days=0
    while list(np.array(data_preprocessed[0][n_original_days])) != [-1, -1, -1, -1]:
        n_original_days += 1

    pattern_list=[]

    i = 0
    saturation = 0

    while (data_fragments.shape[0]-1) >= i:
        if i == 0:
            fragments_list = []

        one_data_fragment = data_fragments[i]
        print(data_fragments.shape)

        n_fragment_days = 1
        while list(one_data_fragment[0][n_fragment_days]) != [-1, -1, -1, -1]:
            n_fragment_days += 1

        if n_fragment_days < round(n_original_days*0.1):
            saturation += 1
        if saturation == data_fragments.shape[0]:
            break

        y_p = pretrained_model_p.predict(one_data_fragment) 
        pattern = list(np.round(y_p[0])[0].astype(np.int16))
        binary = list(np.round(y_p[1])[0].astype(np.int16))

        y_d = pretrained_model_d.predict(one_data_fragment)[1]
        start = np.round(y_d[0][0]).astype(np.int16)
        end = np.round(y_d[0][1]).astype(np.int16)


        if (pattern == [0, 0, 0, 0, 1]) | (pattern == [0, 0, 0, 0, 0]) | (n_fragment_days < round(n_original_days*0.1)) | (start > end) | (start < 0) | (end < 0):
            start = round(n_fragment_days/2)
            end = round(n_fragment_days/2)
            pattern_list.append([0, 0, [0, 0, 0, 0, 1],[0]])

        if (start != end) & (start > 0) & (end > 0) & (start < end):
            if end > n_fragment_days:
                end = n_fragment_days

            global_end = end + ends_index[i]
            global_start = start + ends_index[i]

            if global_end > n_original_days:
                global_end = n_original_days

            if (global_start < global_end) & ((global_end - global_start)>5):
                pattern_list.append([global_start, global_end, pattern, binary])

        else:
            global_end = end + ends_index[i]

        position = (i*2) + 1
        ends_index.insert(position, global_end)

        if len(pattern_list)>1:
            if pattern_list[-1] == pattern_list[-2]:
                pattern_list.pop(-1)
                break

        if (i == 0) & ((data_fragments.shape[0] - 1) == 0):
            data_fragments = divide_fragments(one_data_fragment, start, end)
            continue

        if (data_fragments.shape[0] - 1) > 0:
            if i == 0:
                fragments_list=[]

            if n_fragment_days < round(n_original_days*0.1):
                saturate_model = pad_sequences(tf.convert_to_tensor([[[-1,-1,-1,-1],[-1,-1,-1,-1]]], np.float32), maxlen = 450, dtype='float32', padding='post', value=-1)
                saturate_model = tf.convert_to_tensor([saturate_model, saturate_model], np.float32)
                fragments_list.append(saturate_model)

            else:
                sub_fragments = divide_fragments(one_data_fragment, start, end)

                fragments_list.append(sub_fragments)

            if (i > 0) & ((data_fragments.shape[0] - 1) > 0) & (i == (data_fragments.shape[0] - 1)):
                data_fragments = tf.concat(fragments_list, axis=0)
                i = 0
                continue

        if ((data_fragments.shape[0]-1) > 0):
            i = i + 1

    return pattern_list


def get_chart_p(ticker, start_date, end_date, cdle_patterns):
  df = yf.download(ticker, start=start_date, end=end_date)

  pretrained_model_p = tf.keras.models.load_model("model_p_final")
  pretrained_model_d = tf.keras.models.load_model("model_p_d_final")

  df = candle_type_detection(df)
  columns_to_select = ['date', 'open', 'high', 'low', 'close'] + cdle_patterns
  df = df[columns_to_select]

  data = np.array(df.iloc[:,1:5])
  data_preprocessed, scaler = preprocess_X(data)
  ax = popping_pattern(data_preprocessed, pretrained_model_p, pretrained_model_d, scaler)

  pattern_map = {
    ((0, 0, 0, 0, 0), (1,)):"No pattern",
    ((0, 0, 0, 0, 0), (0,)):"No pattern",
    ((0, 0, 0, 1, 0), (0,)):"Rising wedge",
    ((0, 0, 1, 0, 0), (0,)):"Falling wedge",
    ((0, 1, 0, 0, 0), (0,)):"Double top",
    ((1, 0, 0, 0, 0), (0,)):"Double bottom",
    ((0, 0, 0, 1, 0), (1,)):"Ascending triangle",
    ((0, 0, 1, 0, 0), (1,)):"Descending triangle",
    ((0, 1, 0, 0, 0), (1,)):"Head and shoulders",
    ((1, 0, 0, 0, 0), (1,)):"Inv. head and shoulders"
  }

  ax_df = pd.DataFrame(ax, columns=['start_id', 'end_id', 'pattern', "rising"])
  ax_df[['start_id', 'end_id']] = ax_df[['start_id', 'end_id']].apply(lambda x: np.round(x).astype(np.int16))
  ax_df['pattern'] = ax_df.apply(lambda x: pattern_map[tuple(x[-2:])], axis=1)
  ax_df = ax_df.drop(columns=['rising'])

  ax_df['upper_bound'] = ax_df.apply(lambda x: np.round(df.loc[x['start_id']:x['end_id']+2, 'high'].max(), 1), axis=1)
  ax_df['lower_bound'] = ax_df.apply(lambda x: np.round(df.loc[x['start_id']:x['end_id']+2, 'low'].min(), 1), axis=1)
  ax_df["start_id"] = ax_df.apply(lambda x: str(df.iloc[x['start_id']]["date"]).replace(" 00:00:00",""), axis=1)
  ax_df["end_id"] = ax_df.apply(lambda x: str(df.iloc[x['end_id']-1]["date"]).replace(" 00:00:00",""), axis=1)

  ax_df["x_rectangle"] = ax_df.apply(lambda x: [x['start_id'], x['end_id'], x['end_id'], x['start_id'], x['start_id']], axis=1)
  ax_df["y_rectangle"] = ax_df.apply(lambda x: [x['lower_bound'], x['lower_bound'], x['upper_bound'], x['upper_bound'], x['lower_bound']], axis=1)

  return df, ax_df

def plotting(ticker, start_date, end_date, with_candle = True, cdle_patterns = ['doji', 'o_marubozu']):

  assert all(elem in candle_types for elem in cdle_patterns), f"Valid candle types are: {candle_types}"
  assert isinstance(cdle_patterns, list), "cdle_patterns should be a list"

  df, ax_df = get_chart_p(ticker, start_date, end_date, cdle_patterns)
  data=[]
  _ = [data.append(go.Scatter(x=row["x_rectangle"], y=row["y_rectangle"], mode='lines', fill='toself', fillcolor=f'rgba({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)}, 0.1)', name=row["pattern"])) for index, row in ax_df.iterrows() if row["pattern"]!="No pattern"]

  if data==[]:
      print("Couldn't find relevant patterns, try another date range or ticker")

  data.append(go.Candlestick(x=df.iloc[:,0],
          open=df.iloc[:,1],
          high=df.iloc[:,2],
          low=df.iloc[:,3],
          close=df.iloc[:,4]))

  fig = go.Figure(data=data)

  if with_candle==True:

    fig = create_candlestick_chart(fig, df)

  fig.show()
