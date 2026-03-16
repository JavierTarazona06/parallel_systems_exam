import pylab as plt
import numpy as np
import time
import numba

@numba.njit(parallel=True)
def mandelbrot_iter(c_arr: np.ndarray, loop: int) -> np.ndarray:
    color = np.ones(np.shape(c_arr), np.int64) + 5
    for i in numba.prange(c_arr.shape[0]):
        for j in range(c_arr.shape[1]):
            c0 = c_arr[i, j]
            z = 0.j
            for n in range(loop):
                z = z*z + c0
                if np.abs(z)>2:
                    color[i,j] = (100*np.minimum(color[i,j], n))/loop
                    break
    return color

# initial values 
loop = 1000 # number of interations
div = 1500 # divisions
# all possible values of c
c = np.linspace(-2,2,div)[:,np.newaxis] + 1j*np.linspace(-1.5,1.5,div)[np.newaxis,:]
# Array that will hold colors for plot, initial value set here will be
# the color of the points in the mandelbrot set, i.e. where the series
# converges.
# For the code below to work, this initial value must at least be 'loop'.
# Here it is loop + 5
start = time.time()
color = mandelbrot_iter(c, loop)
end = time.time()
print(f"Time taken: {end - start} seconds")

plt.rcParams['figure.figsize'] = [12, 7.5]
# contour plot with real and imaginary parts of c as axes
# and colored according to 'color'
plt.contourf(c.real, c.imag, color)
plt.xlabel("Real($c$)")
plt.ylabel("Imag($c$)")
plt.xlim(-2,2)
plt.ylim(-1.5,1.5)
plt.savefig("plot.png")
plt.show()
