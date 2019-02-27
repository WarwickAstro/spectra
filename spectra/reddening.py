import numpy as np

__all__ = [
  "A_curve"
]

def A_curve(x, R=3.1):
  """
  Calculate CCM 1989 extinction curve. x is in units of 1/um.
  """
  def Av_IR(x):
    """
    0.3 <= x/um < 1.1
    """
    a = 0.574 * x**1.61
    b =-0.527 * x**1.61
    return a, b

  def Av_opt(x):
    """
    1.1 <= x/um < 3.3
    """
    y = x-1.82
    poly_a = [1, +0.17699, -0.50447, -0.02427, +0.72085, +0.01979, -0.77530, +0.32999][::-1]
    poly_b = [0, +1.41338, +2.28305, +1.07233, -5.38434, -0.62251, +5.30260, -2.09002][::-1]
    a = np.polyval(poly_a, y)
    b = np.polyval(poly_b, y)
    return a, b

  def Av_UV(x):
    """
    3.3 <= x/um < 8.0
    """
    poly_Fa = [0, 0, -0.04473, -0.009779][::-1]
    poly_Fb = [0, 0, +0.21300, +0.120700][::-1]
    Fa = np.polyval(poly_Fa, x-5.9)
    Fb = np.polyval(poly_Fb, x-5.9)
    if isinstance(x, np.ndarray):
      Fa[x < 5.9] = 0
      Fb[x < 5.9] = 0
    elif isinstance(x, (int, float)):
      if x < 5.9:
        Fa = Fb = 0
    else:
      raise TypeError
    a =  1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + Fa
    b = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + Fb
    return a, b

  FIR = (x < 0.3)
  IR  = (x >= 0.3) & (x<1.1)
  opt = (x >= 1.1) & (x<3.3)
  UV  = (x >= 3.3) & (x<8.0)
  FUV = (x >= 8.0)

  a = np.zeros_like(x)
  b = np.zeros_like(x)
  a[IR ], b[IR ] = Av_IR( x[IR ])
  a[opt], b[opt] = Av_opt(x[opt])
  a[UV ], b[UV ] = Av_UV( x[UV ])
  a[FIR], b[FIR] = Av_IR(0.3)
  a[FUV], b[FUV] = Av_UV(8.0)
  A = a + b/R
  return A

