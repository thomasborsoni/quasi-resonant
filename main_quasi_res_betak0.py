import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions_quasi_res_betak0 import one_step
import time
from numba import njit, prange



# Parameters of the model

# beta_k = 0, beta_i = - alpha - 1, gamma = 2 alpha + 5


# We assume density 1, mass = 1 and alpha = 0 and percent_coll = 1

eps = .01

# Numerical Parameters

N = 10**7               # number of particles

nbtimesteps = 3000        # number of steps


@njit(parallel = True)
def one_step_opti(V_vector, q_vector, half_N, exp_eps):

    # Compute exp(-eps) once and for all

    exp_minus_eps = 1 / exp_eps

    # Compute max quantities

    Ekmax = ((np.max(V_vector[:,0]) - np.min(V_vector[:,0]))**2 + (np.max(V_vector[:,1]) - np.min(V_vector[:,1]))**2 + (np.max(V_vector[:,2]) - np.min(V_vector[:,2]))**2) / 4

    Imax = np.max(q_vector)

    tau_min = 1 / (2 * np.sqrt(Ekmax) * (Ekmax + 2*Imax)**2)

    # initialize sigma
    sigma = np.empty(3)

    # Choose colliding pairs

    selected_pairs = rd.randint(0,2*half_N,2*half_N)

    random_numbers_for_test = rd.rand(half_N)

    for k in prange(half_N):

        n = selected_pairs[k]
        m = selected_pairs[k+half_N]

        v = V_vector[n]
        q = q_vector[n]

        v_star = V_vector[m]
        q_star = q_vector[m]


        Ek = np.sum((v-v_star)**2) / 4
        sumI = q + q_star

        Energy = Ek + sumI


        if random_numbers_for_test[k] < np.sqrt(Ek/Ekmax) * ((Ek + sumI)/(Ekmax + 2*Imax))**2  :

            random_numbers = rd.rand(4)

        # Then a collision happens, update the states

            R0 = sumI / Energy
            R0m = R0 / (R0 + exp_eps*(1-R0))
            R0p = R0 / (R0 + exp_minus_eps*(1-R0))

            R = R0m + (R0p - R0m) * random_numbers[0]

            q_vector[n] = random_numbers[1] * R * Energy
            q_vector[m] = (1 - random_numbers[1]) * R * Energy


            # Draw an angle uniformly on the sphere

            theta = 6.283185307179586 * random_numbers[2]            # 2 * pi

            cos_phi = 2 * random_numbers[3] - 1
            sin_phi = np.sqrt(1 - cos_phi**2)

            sigma[0] = np.cos(theta) * sin_phi
            sigma[1] = np.sin(theta) * sin_phi
            sigma[2] = cos_phi


            # update the velocities accordingly

            mean_v = (v + v_star) / 2
            change_sigma = np.sqrt((1-R)*Energy) * sigma

            V_vector[n] = mean_v + change_sigma
            V_vector[m] = mean_v - change_sigma

    return V_vector, q_vector, tau_min


# Choice of initialization

init_exp = True            # if False, init with uniform distributions
Tk_init = 1
Ti_init = 1.3

# Initial conditions

if init_exp :
    V_vector0 = rd.normal(0,np.sqrt(Tk_init),(N,3))
    q_vector0 = rd.exponential(Ti_init,N)

else :
    V_vector0 = (rd.rand(N,3) - .5)
    q_vector0 = rd.rand(N) * 10

# Zeroth temperature

E0 = (np.sum((V_vector0 - np.mean(V_vector0))**2) / 2  + np.sum(q_vector0) ) / N
Tf = E0 / (3 / 2 + 1)

# exponential of epsilon

exp_eps = np.exp(eps)

# other params

hundredth_iteration = int(max(nbtimesteps/100,1))

current_time = 0

# Run

V_vector = V_vector0.copy()
q_vector = q_vector0.copy()

Ti0 = np.sum(q_vector0) / N

Temps = [Ti0]
Time_vector = [current_time]


for k in range(nbtimesteps):

    # t1 = time.time()
    # V_vector, q_vector, dt = one_step(1,V_vector,q_vector,1,0,eps)
    t2 = time.time()
    V_vector, q_vector, dt = one_step_opti(V_vector, q_vector, N//2, exp_eps)
    t3 = time.time()

    # print(t2-t1)


    current_time += dt

    # keep track of timestep
    if k%hundredth_iteration==0:

        print(str(int(100*k/nbtimesteps)) + ' %')
        print(t3-t2)

        Time_vector.append(current_time)

        # Compute temperature

        Temps.append(np.sum(q_vector) / N)


#### Plot Temperatures

Ti_array = np.array(Temps)
Tf_array = Tf * np.ones_like(Ti_array)


plt.figure()
plt.plot(Time_vector,Ti_array,label='$T_i$ DSMC')
# plt.plot(Time_vector,Tf_array,label='$T_{eq}$')

Tmax = Time_vector[-1]

# Solve corresponding temperature equation

Cte_T = eps**2 * 4 / np.sqrt(np.pi) * 3

def func_equation(t,Ti):

    Tk = 2 / 3 * ((3/2  + 1) * Tf -  Ti)

    return  Cte_T * Tk**(1.5) * Ti *(Tk - Ti)


theory_Ti = solve_ivp(func_equation, [0,Tmax], [Ti0], dense_output=True, method='Radau').sol

vect_t = np.linspace(0,Tmax,1000)

thTivect = theory_Ti(vect_t)[0]

plt.plot(vect_t, thTivect, '--',label = '$T_i$ th.')
plt.legend()
plt.show()


