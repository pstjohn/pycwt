"""
Sample script to test the wavelet methods on characterizing two sets of
circadian data.
"""

import numpy as np
import pylab as plt
from pycwt import wavelet
from pycwt import proportional_ridge

# filename = 'circ_easy.csv'
filename = 'circ_hard.csv'
# For reproducibility
# np.random.seed(1234)
# 
# 
# # Set real parameters for the sinusoid
# true_logamp = 1.2
# true_decay = 0.01
# true_phase = np.pi/4
# 
# x = np.linspace(0, (10 + np.random.rand())*24., num=300)
# 
# true_period = 24 + 1*np.sin(6*np.pi*x/len(x))
# 
# # Simulate the true trajectory
# y_real = (np.exp(true_logamp - true_decay*x) *
#           np.cos(2*np.pi*x/true_period + true_phase))
# 
# # Add some noise
# y_err = y_real + 0.05*y_real.max()*np.random.randn(len(x))
# 
import pandas as pd
# data = pd.DataFrame({'x' : x, 'y' : y_err})
data = pd.read_csv(filename)

import CommonFiles.Bioluminescence as bl

bclass = bl.Bioluminescence(data.x, data.y, period_guess=24.)
bclass.detrend()
x = bclass.x
y = bclass.y

std = data.y.std()                   # Standard deviation
std2 = std ** 2                      # Variance
var = (y - y.mean()) / std # Normalizing

N = var.size                         # Number of measurements
# x = data.x
dt = x[1] - x[0]

dj = 0.01                            # Four sub-octaves per octaves
s0 = -1 #2 * dt                      # Starting scale, here 6 months
J = 7 / dj                      # Seven powers of two with dj sub-octaves
try: alpha, _, _ = wavelet.ar1(var)       # Calculate red-noise for current spectrum
except Warning: alpha = 0.0               # Lag-1 autocorrelation for white noise

mother = wavelet.Morlet(4)          # Morlet mother wavelet with wavenumber=6
# mother = wavelet.Paul(m=4.)          # Morlet mother wavelet with wavenumber=6

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0,
                                                      J, mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)
power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
period = 1. / freqs

slevel = 0.95                        # Significance level

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=slevel,
                                         wavelet=mother)

sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95                # Where ratio > 1, power is significant


# Calculate ridge
ridge = proportional_ridge(wave, coi, freqs, max_dist=0.01)


# Plot results

fig_sig = plt.figure()
ax_sig = fig_sig.add_subplot(111)

ax_sig.plot(data.x, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax_sig.plot(data.x, var, 'k', linewidth=1.5)

fig_cwt = plt.figure()
ax_cwt = fig_cwt.add_subplot(111)

levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
# ax_cwt.contourf(data.x, np.log2(period), np.log2(power),
#                 np.log2(levels), extend='both')
x = data.x
y = np.log2(period)
X, Y = np.meshgrid(x, y)
ax_cwt.pcolormesh(X, Y, np.log2(power), cmap='RdBu')
ax_cwt.contour(data.x, np.log2(period), sig95, [-99, 1], colors='k',
               linewidths=2.)
ax_cwt.plot(x, y[ridge], 'r')
# ax_cwt.plot(x, np.log2(true_period), 'g')
ax_cwt.fill(
    np.concatenate([data.x, data.x[-1:]+dt,
                    data.x[-1:] + dt,data.x[:1] - dt,
                    data.x[:1]-dt]),
    np.log2(np.concatenate([coi,[1e-9], period[-1:], period[-1:],
                            [1e-9]])),
    'k', alpha=0.3, hatch='x')

Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))

ax_cwt.set_ylim(np.log2([period.min(), period.max()]))
ax_cwt.set_yticks(np.log2(Yticks))
ax_cwt.set_yticklabels(Yticks)
ax_cwt.set_xlim([data.x.min(), data.x.max()])
ax_cwt.invert_yaxis()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, power[ridge], label='Amplitude')
# ax.plot(x, np.abs(wave[ridge]))

plt.show()
