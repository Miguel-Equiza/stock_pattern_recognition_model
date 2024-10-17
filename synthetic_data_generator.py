
import numpy as np
import pandas as pd
import scipy
import plotly.graph_objects as go


def get_X_start_end(n_days):

  length_pattern = np.random.randint(50,n_days/3+1) 
  start_pattern = np.random.randint(1,n_days-length_pattern-1)  
  end_pattern = start_pattern + length_pattern
  days_after_pattern = n-end_pattern  

  X = pd.to_datetime(np.arange(n_days), unit='D',
    origin=pd.Timestamp('2017-05-08'))

  return X, start_pattern, end_pattern, length_pattern, days_after_pattern

def get_normal_noise(n_days, mu, sigma, ar1, ar2):
  open_noise = []
  y0 = 0
  for i in range(n_days):

    if i == 0:
      open_noise.append(0)
      y1 = open_noise[-1]
    else:
      open_noise.append(ar1*y1 + ar2*y0 + scipy.stats.norm.rvs(mu, sigma))
      y0 = y1
      y1 = open_noise[-1]
  return open_noise

def create_noise_around_pattern(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2):
  before_pattern_noise = get_normal_noise(start_pattern, mu, sigma, ar1, ar2)

  after_pattern_noise = get_normal_noise(days_after_pattern, mu, sigma, ar1, ar2)

  # This is to make the last element before the pattern have the same value as the first of the pattern, and the same with after
  before_pattern_noise.reverse()
  before_pattern_noise = list(np.array(before_pattern_noise) + open_pattern[0])
  after_pattern_noise = list(np.array(after_pattern_noise) +open_pattern[-1])

  open_all = before_pattern_noise + open_pattern + after_pattern_noise

  return open_all

def create_close(open_all):
  return open_all[1:]+[open_all[-1]]

def create_low_high(open_all, close_all, max_h):

  low_all = []
  high_all = []
  for i in range(len(open_all)):

    low_all.append(min([close_all[i],open_all[i]]) - np.random.uniform(0,max_h))
    high_all.append(max([close_all[i],open_all[i]]) + np.random.uniform(0,max_h))

  return low_all, high_all

def fig_plot(dates, open_all, high_all, low_all, close_all, start_pattern, end_pattern):
  fig = go.Figure(data=[go.Candlestick(x=dates,
              open=open_all,
              high=high_all,
              low=low_all,
              close=close_all)])
  fig.add_vline(x = dates[start_pattern])
  fig.add_vline(x = dates[end_pattern])
  fig.show()

def create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h):

  open_all = create_noise_around_pattern(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2)
  close_all = create_close(open_all)
  low_all, high_all = create_low_high(open_all, close_all, max_h)

  return open_all, high_all, low_all, close_all


def double_bottom(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x = np.linspace(-1.4,3.4,length_pattern)
  noise = np.random.normal(0,noise_level,length_pattern)

  b = np.random.uniform(3.9,4.1)
  c = np.random.uniform(4,7)
  d = np.random.uniform(1,5)

  pattern_function = x**4 - b*(x**3)+ x**2 + c*x - d
  open_pattern = 4*pattern_function+2*noise
  open_pattern = list(open_pattern + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def double_top(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x = np.linspace(-1.4,3.4,length_pattern)
  noise = np.random.normal(0,noise_level,length_pattern)

  b = np.random.uniform(3.9,4.1)
  c = np.random.uniform(4,7)
  d = np.random.uniform(1,5)

  pattern_function = -x**4 + b*(x**3)- x**2 - c*x + d
  open_pattern = 4*pattern_function+2*noise
  open_pattern = list(open_pattern + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern


def bearish_flag(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  a = np.random.uniform(2,8)
  x_bearish = np.linspace(-10,0,round(length_pattern/3))
  x_flag = np.linspace(0,20,round((length_pattern/3)*2))
  noise_bearish = np.random.normal(0,noise_level,round(length_pattern/3))
  noise_flag = np.random.normal(0,(a/4)*noise_level,round((length_pattern/3)*2))

  m = np.random.uniform(np.sqrt(3),5)
  bearish_pattern = m*abs(x_bearish)
  bearish_pattern = 4*bearish_pattern + 2*noise_bearish

  n = np.random.uniform(0.2,0.8)
  flag_pattern = n*x_flag + 3*np.sin(x_flag)
  flag_pattern = 4*flag_pattern + 2*noise_flag

  open_pattern = list(bearish_pattern) + list(flag_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def bullish_flag(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  a = np.random.uniform(2,8)
  x_bullish = np.linspace(-10,0,round(length_pattern/3))
  x_flag = np.linspace(-0.5*np.pi,20,round((length_pattern/3)*2))
  noise_bullish = np.random.normal(0,noise_level,round(length_pattern/3))
  noise_flag = np.random.normal(0,(a/4)*noise_level,round((length_pattern/3)*2))

  m = np.random.uniform(np.sqrt(3),5)
  bullish_pattern = m*abs(x_bullish)
  bullish_pattern = 4*bullish_pattern + 3*noise_bullish

  n = np.random.uniform(0.2,0.8)
  flag_pattern = -n*x_flag - 3*np.sin(x_flag)
  flag_pattern = 4*flag_pattern + 2*noise_flag

  open_pattern = list(bullish_pattern) + list(flag_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def bullish_pennant(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-10,0,round(length_pattern/3))
  noise_1 = np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  m = np.random.uniform(2,5)
  bullish_pattern = 4*m*(x_1) + 3*noise_1

  n = np.random.uniform(0.2,0.8)
  x_2=np.linspace(-0.5*np.pi,30,round((length_pattern/3)*2))
  noise_2 = np.random.normal(0,noise_level,round((length_pattern/3)*2))
  x2_2 = np.flip(x_2)
  pennant_pattern = 2*(-0.1*x_2 - 3*np.sin(x_2)*((x2_2**2)/200)) + 2*noise_2
  pennant_pattern = pennant_pattern - pennant_pattern[0]

  open_pattern = list(bullish_pattern) + list(pennant_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def bearish_pennant(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-10,0,round(length_pattern/3))
  noise_1 = np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  m = np.random.uniform(2,5)
  bearish_pattern = -4*m*(x_1) - 3*noise_1

  n = np.random.uniform(0.2,0.8)
  x_2=np.linspace(-0.5*np.pi,30,round((length_pattern/3)*2))
  noise_2 = np.random.normal(0,noise_level,round((length_pattern/3)*2))
  x2_2 = np.flip(x_2)
  pennant_pattern = 2*(0.1*x_2 + 3*np.sin(x_2)*((x2_2**2)/200)) + 2*noise_2
  pennant_pattern = pennant_pattern - pennant_pattern[0]

  open_pattern = list(bearish_pattern) + list(pennant_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def descending_triangle(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-10,0,round(length_pattern/3))
  noise_1 = np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  m = np.random.uniform(2,5)
  bearish_pattern = -4*m*(x_1) - 3*noise_1

  n = np.random.uniform(0.2,0.8)
  noise_2 = np.random.normal(0,noise_level,round((length_pattern/3)*2))
  x_2 =np.linspace(-20,0.5*np.pi,round((length_pattern/3)*2))
  triangle_pattern = abs((np.sin(x_2)*x_2+x_2)) + 2*noise_2

  open_pattern = list(bearish_pattern) + list(triangle_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def ascending_triangle(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-10,0,round(length_pattern/3))
  noise_1 = np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  m = np.random.uniform(2,5)
  ascending_pattern = 4*m*(x_1) - 3*noise_1

  n = np.random.uniform(0.2,0.8)
  noise_2 = np.random.normal(0,noise_level,round((length_pattern/3)*2))
  x_2 =np.linspace(-20,0.5*np.pi,round((length_pattern/3)*2))
  triangle_pattern = -abs((np.sin(x_2)*x_2+x_2)) + 2*noise_2

  open_pattern = list(ascending_pattern) + list(triangle_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def cup_handle(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-10,0,round(length_pattern/3))
  noise_1 = np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  m = np.random.uniform(0.8,5)
  cup_pattern = 4*m*(x_1) - 3*noise_1

  n = np.random.uniform(0.2,0.8)
  x_2 = np.linspace(np.pi,2*np.pi,round((length_pattern/3)*2))
  noise_2 = np.random.normal(0,noise_level,round((length_pattern/3)*2))
  handle_pattern = 5*15*np.sin(x_2) + 3*noise_2

  open_pattern = list(cup_pattern) + list(handle_pattern)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def head_shoulders(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-0.5*np.pi,2*np.pi,round(length_pattern/3))

  shoulder_1 = 7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  shoulder_1[-1]=0
  shoulder_1[0]=0

  shoulder_2 = 7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  shoulder_2 = list(shoulder_2)
  shoulder_2.reverse()
  shoulder_2[-1]=0
  shoulder_2[0]=0

  x_2 = np.linspace(0,np.pi,round(length_pattern/3))
  m = np.random.uniform(11,25)
  head = m*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  head[-1]=0
  head[0]=0

  open_pattern = list(shoulder_1) + list(head) + list(shoulder_2)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def inv_head_shoulders(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-0.5*np.pi,2*np.pi,round(length_pattern/3))

  shoulder_1 = -7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  shoulder_1[-1]=0
  shoulder_1[0]=0

  shoulder_2 = -7*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  shoulder_2 = list(shoulder_2)
  shoulder_2.reverse()
  shoulder_2[-1]=0
  shoulder_2[0]=0

  x_2 = np.linspace(0,np.pi,round(length_pattern/3))
  m = np.random.uniform(11,25)
  head = -m*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/3))
  head[-1]=0
  head[0]=0

  open_pattern = list(shoulder_1) + list(head) + list(shoulder_2)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def falling_wedge(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  m1 = np.random.uniform(1.6,2.2)
  m2 = np.random.uniform(1.6,2.2)
  m3 = np.random.uniform(1.6,2.2)
  n1 = np.random.uniform(1,1.5)
  n2 = np.random.uniform(1,1.5)

  rep=int(length_pattern/5)
  x_1 = np.linspace(0,1.5/(m1+n1),rep)
  x_2 = np.linspace(1.5/(m1+n1),2.5/(m2+n1),rep)
  x_3 = np.linspace(2.5/(m2+n1),4/(m2+n2),rep)
  x_4 = np.linspace(4/(m2+n2),5/(m3+n2),rep)
  x_5 = np.linspace(5/(m3+n2),7/(m3+n2),rep)

  y1 = -m1*x_1
  y2 = n1*x_2-1.5
  y3 = -m2*x_3+1
  y4 = n2*x_4-3
  y5 = -m3*x_5+2

  open_pattern = list(y1) + list(y2) + list(y3) + list(y4) + list(y5)
  open_pattern = list(50*np.array(open_pattern)+np.random.normal(0,noise_level,len(open_pattern)) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def rising_wedge(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  m1 = np.random.uniform(1.6,2.2)
  m2 = np.random.uniform(1.6,2.2)
  m3 = np.random.uniform(1.6,2.2)
  n1 = np.random.uniform(1,1.5)
  n2 = np.random.uniform(1,1.5)

  rep=int(length_pattern/5)
  x_1 = np.linspace(0,1.5/(m1+n1),rep)
  x_2 = np.linspace(1.5/(m1+n1),2.5/(m2+n1),rep)
  x_3 = np.linspace(2.5/(m2+n1),4/(m2+n2),rep)
  x_4 = np.linspace(4/(m2+n2),5/(m3+n2),rep)
  x_5 = np.linspace(5/(m3+n2),7/(m3+n2),rep)

  y1 = m1*x_1
  y2 = -n1*x_2+1.5
  y3 = m2*x_3-1
  y4 = -n2*x_4+3
  y5 = m3*x_5-2

  open_pattern = list(y1) + list(y2) + list(y3) + list(y4) + list(y5)
  open_pattern = list(50*np.array(open_pattern)+np.random.normal(0,noise_level,len(open_pattern)) + np.random.uniform(-25,25))


  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def tripple_bottom(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-0.5*np.pi,np.pi,round(length_pattern/5))
  x_2 = np.linspace(np.pi, 2*np.pi,round(length_pattern/5))
  x_3 = np.linspace(0,np.pi,round(length_pattern/5))

  m = np.random.uniform(8,15)

  y_1 = -m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_1[-1]=0

  y_2 = -(m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_2[-1]=0
  y_2[0]=0

  y_3 = -m*np.sin(x_3) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_3[-1]=0
  y_3[0]=0

  y_4 = -(m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_4[0]=0
  y_4[-1]=0

  y_5 = -m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_5 = list(y_5)
  y_5.reverse()
  y_5[0]=0

  open_pattern = list(y_1) + list(y_2) + list(y_3) + list(y_4) + list(y_5)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))

  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern

def tripple_top(n_days=500, mu=3, sigma=5, max_h=1, noise_level=2, ar1 = 0.95, ar2 = 0):

  X, start_pattern, end_pattern, length_pattern, days_after_pattern = get_X_start_end(n_days)


  x_1 = np.linspace(-0.5*np.pi,np.pi,round(length_pattern/5))
  x_2 = np.linspace(np.pi, 2*np.pi,round(length_pattern/5))
  x_3 = np.linspace(0,np.pi,round(length_pattern/5))

  m = np.random.uniform(8,15)

  y_1 = m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_1[-1]=0

  y_2 = (m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_2[-1]=0
  y_2[0]=0

  y_3 = m*np.sin(x_3) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_3[-1]=0
  y_3[0]=0

  y_4 = (m/3)*np.sin(x_2) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_4[0]=0
  y_4[-1]=0

  y_5 = m*np.sin(x_1) + np.random.normal(0,0.7*noise_level,round(length_pattern/5))
  y_5 = list(y_5)
  y_5.reverse()
  y_5[0]=0

  open_pattern = list(y_1) + list(y_2) + list(y_3) + list(y_4) + list(y_5)
  open_pattern = list(np.array(open_pattern) + np.random.uniform(-25,25))

  open_all, high_all, low_all, close_all = create_high_low_close(start_pattern, days_after_pattern, open_pattern, mu, sigma, ar1, ar2, max_h)

  return X, open_all, high_all, low_all, close_all, start_pattern, end_pattern