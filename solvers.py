
#!/bin/python3
from __future__ import print_function
import sys

# Change this to True to enable cuda
USE_CUPY = False

# We want to use cupy when possible, however,  you can also just use numpy
if '-c' in sys.argv or USE_CUPY:
    import cupy as np
    import cusignal as signal
    np.cuda.Device(0).use()
    USE_CUPY = True
    print("Using cupy")
else:
    print("Using Numpy")
    import numpy as np
    from scipy import signal

# not everything is implemented in cupy, so np_real is for the other stuff
import numpy as np_real
import time
from skimage.feature import blob_log
from pandas import DataFrame
import seaborn as sns
import scipy
import math_helper
import random
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature 
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    #__builtin__.print('My overridden print() function!')
    #return __builtin__.print(*args, **kwargs)
    pass

#def get_real_numpy_array(x):
#    if USE_CUPY:
#        return np_real.array(np.asnumpy(x))
#    else:
#        return x

#def convert_to_cupy_array(x):
#    if USE_CUPY:
#        return np.array(x)
#    else:
#        return x 

class ConfigError(Exception):
    """ Base class for config problems """


class EquationBase:
    def __init__(self):
        self._energy_callbacks = []
        self._derivative_callbacks = []

    def add_energy_callback(self, callback):
        self._energy_callbacks.append(callback)


class EquationHelperMixin:
    def laplacian_2d(self, matrix):
        # nine-point-stencil kernel
        #kernel = np.array([[1, 4, 1], [4, -20,4], [1, 4, 1]])  # 9 point stencil
        kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        # periodic padding
        temp_mat = np.pad(matrix, (1,1), 'wrap')
        return signal.convolve2d(temp_mat, kernel, mode='valid')


class OverdampedPhi4(EquationHelperMixin, EquationBase):
    def __init__(self, parameter):
        self._parameter = parameter
        super(OverdampedPhi4, self).__init__()

    def derivative(self, timestep, y):
        lap1 = self.laplacian_2d(y[0])
        lap2 = self.laplacian_2d(y[1])
        g = self._parameter

        eq1 = -g*y[0] + lap1 - y[0]**3 - y[0] * y[1]**2
        eq2 = -g*y[1] + lap2 - y[1]**3 - y[1] * y[0]**2

        shape = y.shape
        temp = np.zeros(shape)

        temp[0] = eq1
        temp[1] = eq2

        for callback in self._derivative_callbacks:
            callback(timestep, y, temp)

        return temp

    def energy(self, timestep, field_matrix):
        x1 = field_matrix[0]
        x2 = field_matrix[1]

        en = -1 * 0.5 * x1 * x1 + -1*0.5*x2*x2
        en2 = 0.25 * x1 * x1 * x1 * x1 + 0.25 * x2 * x2 * x2 * x2 + 0.5 *x1 * x1 * x2 * x2
        en3 = 0.5*self.laplacian_2d(x1) + 0.5 * self.laplacian_2d(x2)
        ke = 0.5*x1*x1 + 0.5*x2 * x2
        en = np.sum(en)
        en2 = np.sum(en2)
        en3 = np.sum(en3)
        ke = np.sum(ke)
        en_tot = en + en2 + en3
        for callback in self._energy_callbacks:
            callback(timestep, field_matrix, en, en2, en3, 0, en_tot)

        return en, en2, en3, ke, en_tot

    def get_noise(self, shape):
        return np.random.normal(loc=0, size=shape, scale=0.03)

class UnderdampedPhi4(EquationHelperMixin, EquationBase):
    def __init__(self, parameter, overdamped):
        self._parameter = parameter
        self._overdamped = overdamped
        self._use_overdamped = True
        self._use_thermalize = True
        super(UnderdampedPhi4, self).__init__()

    def derivative(self, timestep, y):
        #lap1 = self.laplacian_2d(y[0])
        #lap2 = self.laplacian_2d(y[2])
        #if self._use_overdamped:
        #    return self._overdamped.derivative(timestep, y)

        if self._use_thermalize:
            g = -self._parameter
            self.g = g
            self._overdamped._parameter = g
        else:
            g = self._parameter
            self.g = g
            self._overdamped._parameter = g

        if self._use_overdamped:
            return self._overdamped.derivative(timestep, y)

        #breakpoint()
        x1 = y[0]
        x2 = y[2]
        y1 = y[1]
        y2 = y[3]
        # Key
        lap1 = 1*self.laplacian_2d(x1)
        lap2 = 1*self.laplacian_2d(x2)

        x1_new = y1
        y1_new = -g*x1 + lap1 -x1**3 - x1*x2**2
        x2_new = y2
        y2_new = -g*x2 + lap2 -x2**3 - x2*x1**2

        eq1 = x1_new
        eq2 = y1_new
        eq3 = x2_new
        eq4 = y2_new

        shape = y.shape
        temp = np.zeros(shape)

        temp[0] = eq1
        temp[1] = eq2
        temp[2] = eq3
        temp[3] = eq4

        for callback in self._derivative_callbacks:
            callback(timestep, y, temp)

        return temp

    def energy(self, timestep, field_matrix):
        if self._use_overdamped:
            #breakpoint()
            return self._overdamped.energy(timestep, field_matrix)

        if self._use_thermalize:
            g = -self._parameter
        else:
            g = self._parameter

        x1 = field_matrix[0]
        x2 = field_matrix[2]
        y1 = field_matrix[1]
        y2 = field_matrix[3]

        en = g*0.5*x1*x1 + g*0.5*x2*x2
        en2 = 0.25 * x1*x1*x1*x1 + 0.25*x2*x2*x2*x2 + 0.5 * x1*x1*x2*x2

        x1right = np.roll(x1, 1, axis=1) - x1
        x1up = np.roll(x1, 1, axis=0) - x1

        x2right = np.roll(x2, 1, axis=1) - x2
        x2up = np.roll(x2, 1, axis=0) - x2
        en3 = 1*(0.5 * x1right**2 + 0.5*x1up**2 + 0.5*x2right**2 + 0.5*x2up**2)
        ke = 0.5*y1*y1 + 0.5*y2*y2

        en = np.sum(en)
        en2 = np.sum(en2)
        en3 = np.sum(en3)
        ke = np.sum(ke)
        en_tot = en + en2 + en3 + ke
        for callback in self._energy_callbacks:
            callback(timestep, field_matrix, en, en2, en3, ke, en_tot)
        #0.01 -30706.270671903876 16010.30128604571 6690.843958067564 8122.966690256067 0.0017981149668192115

        norm = 256**2
        return en / norm, en2 / norm, en3 / norm, ke / norm, en_tot / norm

    def get_next_verlet(self, field_matrix, dt):

        fields = field_matrix
        x1 = fields[0]
        x2 = fields[2]
        y1 = fields[1]
        y2 = fields[3]

        if self._use_thermalize:
            g = -self._parameter
        else:
            g = self._parameter

        lap1 = 1*self.laplacian_2d(x1)
        lap2 = 1*self.laplacian_2d(x2)

        a1 = -g*x1 + lap1 - x1**3 - x1*x2**2
        a2 = -g*x2 + lap2 - x2**3 - x2*x1**2

        x1_next = x1 + y1*dt + 0.5*a1*dt**2
        x2_next = x2 + y2*dt + 0.5*a2*dt**2

        lap1_next = self.laplacian_2d(x1_next)
        lap2_next = self.laplacian_2d(x2_next)

        a1_next =  -g*x1_next + lap1_next - x1_next**3 - x1_next*x2_next**2
        a2_next = -g*x2_next + lap2_next - x2_next**3 - x2_next*x1_next**2

        y1_next = y1 + 0.5 * (a1 + a1_next) * dt
        y2_next = y2 + 0.5 * (a2 + a2_next) * dt

        res = np.copy(field_matrix)
        res[0] = x1_next
        res[2] = x2_next
        res[1] = y1_next
        res[3] = y2_next
        return res

    def get_noise(self, shape):
        return np.zeros(shape)

    def clear_thermalize(self):
        #self.parameter = -1 * self.parameter
        self._use_thermalize = False

    def clear_use_overdamped(self):
        self._use_overdamped = False


class SimulationConfig:
    """ Configuration for PDE simulation """
    def __init__(self, config):

        if not isinstance(config, dict):
            raise ConfigError("Config must be a dictionary")

        # Grab mandatory parameters
        self._name = config.get('name')
        self._dimensions = config.get('dimensions')
        self._fields = config.get('fields')
        self._ranges = config.get('ranges')

        self._observables = config.get('observables')
        self._seed = config.get('seed')
        self._steps = config.get('steps')
        self._noises = config.get('noises')

        self._da = config.get('da')

        # Check existance
        member_variables = [attr for attr in dir(self)
                            if not callable(getattr(self, attr))
                            and not attr.startswith("__")]

        error_list = []
        for member in member_variables:
            if getattr(self, member) is None:
                error_list.append("Required Config Item {} does not exist".format(member))

        errors = '\n'.join(error_list)
        if len(error_list) > 0:
            raise ConfigError(errors)
    
        # Type checks
        assert isinstance(self._name, str)
        assert isinstance(self._dimensions, int)
        assert isinstance(self._fields, int)
        assert isinstance(self._ranges, tuple)
        assert isinstance(self._observables, list)
        assert isinstance(self._seed, int)
        assert isinstance(self._steps, int)
        assert isinstance(self._noises, int)
        assert isinstance(self._da, list)

    @property
    def name(self):
        return self._name

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def fields(self):
        return self._fields

    @property
    def ranges(self):
        return self._ranges

    @property
    def observables(self):
        return self._observables

    @property
    def seed(self):
        return self._seed

    @property
    def steps(self):
        return self._steps

    @property
    def noises(self):
        return self._noises

    @property
    def da(self):
        return self._da


class Integrator:
    """ Base class for all Numerical Integration methods.
        This sets up all of the fields needed for integration.
    """

    def __init__(self, config):
        # We need one field of shape ranges for each da
        # However, we should also be able to pass an initializer
        # The shape here should be the steps, dimension.x, dimension.y)
        self._config = config
        shape = config.ranges
        self._field_matrix = []
        for i in range(0, config.fields):
            #self._field_matrix.append(np.zeros(shape, dtype=np.float64))
            self._field_matrix.append(np.random.normal(loc=0, size=shape, scale=0.03))

        #self._field_matrix =  np.random.normal(loc=0, size=shape, scale=0.01)
        self._field_matrix = np.array(self._field_matrix)
        ##Next we need to setup the random number streams
        #self._rngs = []
        #for i in range(0, config.noises):
        #    gen = np.random.Generator.from_seed(config.seed + i*1000)
        #    self._rngs.append(gen)

        self._time = 0

        self._history = []

    def get_field_matrixes(self):
        """
        returns the field matrixes
        """
        return self._field_matrix

    def update_field_matrixes(self, matrixes):
        """
        Updates the field matrixes
        """
        #self._history.append(matrixes)
        self._field_matrix = matrixes

    def periodic_padding(self, matrix):
        a,b = matrix.shape
        temp_matrix = np.tile(matrix, (a,b))
        # Now we carve off shape-1 from each end
        a_start = a-1
        a_end = a_start + a+2
        b_start = b-1
        b_end = b_start + b+2
        return temp_matrix[a_start:a_end, b_start:b_end]


class VelocityVerlet(Integrator):
    """ Velocity Verlet is just the full kinematics equation, excluding higher order jerk terms """

    def __init__(self, config, model):
        if not isinstance(config, SimulationConfig):
            raise ConfigError("Must pass an instance of SimulationConfig")

        self._derivative = model.derivative
        self._energy = model.energy
        self._model = model
        self._time = 1
        self._steps = 1
       # self.previous_field = np.copy
        super().__init__(config)
        self._previous_field = np.zeros(self._field_matrix.shape)
        self._previous_velocity = np.zeros(self._field_matrix.shape)
        self._previous_position = np.zeros(self._field_matrix.shape)

    def step(self, _dt):
        start_time = time.time()
        # Thermalize on the CPU

        if self._model._use_overdamped:
            dt = 0.02
        else:
            dt = _dt # TOOD: Make this configurable

        fields = self._field_matrix
        res = self._model.get_next_verlet(fields, dt)
        noise = self._model.get_noise(res.shape)
        self.update_field_matrixes(res + noise)
        en, en2, en3, ke, tot = self._energy(self._steps, self._field_matrix)

        self._time += dt
        self._steps += 1
        if self._steps % 1000 == 0:
            print("{} {} {} {} {} {}".format(dt, en, en2, en3, ke, tot))
            print("--- %s seconds ---\n" %(time.time() - start_time))
        return en, en2, en3, ke, tot, self._time, self._steps

    def clear_thermalize(self):
        #breakpoint()
        self._model.clear_thermalize()

    def clear_use_overdamped(self):
        self._model.clear_use_overdamped()

    def get_engine_parameters(self):
        if self._model._use_thermalize:
            a = "thermalizing"
        else:
            a = "not-thermalizing"

        if self._model._use_overdamped:
            b = "overdamped"
        else:
            b = "underdamped"

        c =  self._model._parameter

        return a, b, c

    def change_parameter(self):
        self._model.change_parameter()
            
class RK4(Integrator):
    """ Implementation of a fourth order Runge-Kutta """

    def __init__(self, config, model):
        if not isinstance(config, SimulationConfig):
            raise ConfigError("Must pass an instance of SimulationConfig")

        self._derivative = model.derivative
        self._energy = model.energy
        self._model = model
        self._time = 1
        self._steps = 1
        super().__init__(config)

    def step(self, _dt):
        start_time = time.time()
        # Thermalize on the CPU

        if self._model._use_overdamped:
            dt = 0.02
        else:
            dt = _dt # TOOD: Make this configurable

        a1 = self._field_matrix
        d1 = self._derivative(self._steps, a1) # full time step
        k1 = a1 # + d1        
        d2 = self._derivative(self._steps, a1) * dt / 2
        k2 = a1 + d2
        d3 = self._derivative(self._steps, k2) * dt / 2
        k3 = k2 + d3
        d4 = self._derivative(self._steps, k3) * dt
        k4 = k3 + d4
        #breakpoint()
        res = 1/6 * (k1 + 2 * k2 + 2*k3 + k4)
        #res = a1 + d1 # euler
        #res = a1 + d1*dt
        # update hte field matrix
        noise = self._model.get_noise(res.shape)
        self.update_field_matrixes(res + noise)

        # calculate energy
        en, en2, en3, ke, tot = self._energy(self._steps, self._field_matrix)
        # update time
        if self._steps % 1000 == 0:
            print("{} {} {} {} {} {}".format(dt, en, en2, en3, ke, tot))
            print("--- %s seconds ---\n" %(time.time() - start_time))

        self._time += dt
        self._steps += 1
        return en, en2, en3, ke, tot, self._time, self._steps

    def change_parameter(self):
        self._model.change_parameter()
        
    def clear_thermalize(self):
        #breakpoint()
        self._model.clear_thermalize()

    def clear_use_overdamped(self):
        self._model.clear_use_overdamped()

    def get_engine_parameters(self):
        if self._model._use_thermalize:
            a = "thermalizing"
        else:
            a = "not-thermalizing"

        if self._model._use_overdamped:
            b = "overdamped"
        else:
            b = "underdamped"

        c =  self._model._parameter

        return a, b, c
            
