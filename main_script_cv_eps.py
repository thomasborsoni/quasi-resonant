import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit, prange
import time

# Parameters of the model: density = 1, mass = 1, delta = 2 (thus gamma = 5), beta_k = 0, beta_i = - 1.

# #### User's choices #####

# Choose epsilon
eps = .1

# Numerical parameters
N = 10**5                      # number of particles
nbTimeSteps = 400000          # number of steps

# Initial temperatures
Tk_init = 1
Ti_init = 50


# # Functions


# The numba library allows for a considerable speed up (of a factor 400 up to 700 on tests I performed): a factor 40 to 60 by compiling and a factor 10 to 20 by parallelizing. The first is achieved simply by using the decorator @jit(nopython=True) at the beginning of the function and the latter is achieved simply by using @njit(parallel=True) instead, and prange instead of range in the loop we parallelize.


# For a noticeable speedup, we compile the computation of Ekmax with numba
@njit
def compute_Ekmax(V_vector):

    return  ((np.max(V_vector[:,0]) - np.min(V_vector[:,0]))**2 + (np.max(V_vector[:,1]) - np.min(V_vector[:,1]))**2 + (np.max(V_vector[:,2]) - np.min(V_vector[:,2]))**2) * 0.25

# Hereafter we define the function that does one step
@njit(parallel = True)
def collision_step(V_vector, q_vector, inv_maxes_sqrd, number_pairs, N, exp_eps):

    # Initialize sigma and create the random numbers to be used for the acceptation/rejection
    sigma = np.empty(3)
    random_numbers_for_test = rd.rand(number_pairs)

    # Loop on potential candidates for collision
    for k in prange(number_pairs):

        # Randomly select a pair as potential candidate for collision
        n, m  = rd.randint(0, N, 2)

        # Extract v and v_star
        v = V_vector[n]
        v_star = V_vector[m]

        # Compute relative kinetic and internal energies
        Ek = np.sum((v - v_star)**2) * 0.25
        Ei = q_vector[n] + q_vector[m]

        Energy = Ek + Ei

        # Test if the collision indeed happens of not - notice that if
        # n = m then Ek = 0 and rejection is ensured
        test_nb = random_numbers_for_test[k]

        # Accept / reject test
        if test_nb * test_nb < Ek * Energy * Energy * Energy * Energy * inv_maxes_sqrd:

            ### The pair is selected for collision, update the states

            # Compute R', R_eps^-, R_eps^+
            R_prime = Ek / Energy
            R_eps_plus = exp_eps * R_prime / (exp_eps * R_prime + (1-R_prime))
            R_eps_minus = R_prime / (R_prime + (1-R_prime) * exp_eps)

            # Draw R according to the correct law
            U = rd.rand()
            R = R_eps_minus * (1 - U) + R_eps_plus * U

            # Draw r uniformly
            r = rd.rand()

            # Update the internal states
            q_vector[n] = r * (1-R) * Energy
            q_vector[m] = (1 - r) * (1-R) * Energy


            # Draw sigma uniformly on the 2D-sphere
            theta = 6.283185307179586 * rd.rand()  # 2 * pi

            cos_phi = 2 * rd.rand() - 1
            sin_phi = np.sqrt(1 - cos_phi * cos_phi)
            
            sigma[0] = np.cos(theta) * sin_phi
            sigma[1] = np.sin(theta) * sin_phi
            sigma[2] = cos_phi

            # Update the velocities
            mean_v = (v + v_star) * 0.5
            deviation = np.sqrt(R * Energy) * sigma

            V_vector[n] = mean_v + deviation
            V_vector[m] = mean_v - deviation

    return V_vector, q_vector


#%% DSMC simulation

### Initialize

# Maxwellian initial distribution
V_vector0 = rd.normal(0,np.sqrt(Tk_init),(N,3))
q_vector0 = rd.exponential(Ti_init,N)

# Actual initial internal and equilibrium temperatures
Ti0 = np.sum(q_vector0) / N
Tf = (np.sum((V_vector0 - np.mean(V_vector0,axis=0))**2) / (2*N)  + Ti0) / (5/2)

# Compute once and for all exp(epsilon)
exp_eps = np.exp(eps)

# Other parameters
thousandth_iteration = int(max(nbTimeSteps/1000,1))
number_pairs = N // 2

# Initialize molecules
V_vector = V_vector0.copy()
q_vector = q_vector0.copy()

# Initialize the list of internal temperatures and the list of times
DSMC_Ti_vector = [Ti0]
current_time = 0
DSMC_time_vector = [current_time]

# Compute initial Ekmax and Imax
previous_Ekmax = compute_Ekmax(V_vector)
Ekmax = previous_Ekmax * 1.02           # take 2% margin for the first iterations
previous_Imax = np.max(q_vector)
Imax = previous_Imax * 1.02             # take 2% margin for the first iterations
inv_maxes = 1 / ((Ekmax + 2 * Imax)**2 * np.sqrt(Ekmax))
inv_maxes_sqrd = inv_maxes**2


### Run

# Start simulation
simulation_time_start = time.time()
count_communication = 0

for k in range(1,nbTimeSteps):

    # Perform collisions: update velocity and internal vectors
    V_vector, q_vector = collision_step(V_vector, q_vector, inv_maxes_sqrd, number_pairs, N, exp_eps)

    # Update current time
    current_time += 0.5 * inv_maxes
    

    # Every nbtimestep/1000 iterations, store time, temperature, update Ekmax and Imax
    if k%thousandth_iteration==0:

        # Store current time
        DSMC_time_vector.append(current_time)

        # Store current internal temperature
        DSMC_Ti_vector.append(np.sum(q_vector) / N)
        
        # Compute max quantities - this sole computation is about 30% slower than
        # an iteration of the function collision_step
        new_Ekmax = compute_Ekmax(V_vector)
        new_Imax = np.max(q_vector)
    
        # Update max quantities by also maxing with the previous maxes to take some margin
        Ekmax = max(previous_Ekmax, new_Ekmax)
        Imax = max(previous_Imax, new_Imax)
        inv_maxes = 1 / ((Ekmax + 2 * Imax)**2 * np.sqrt(Ekmax))
        inv_maxes_sqrd = inv_maxes * inv_maxes
    
        # Update previous_maxes
        previous_Ekmax = new_Ekmax
        previous_Imax = new_Imax

        # Update counter for communication
        count_communication += 1

        # Communicate the advancement of the simulation
        if count_communication==10:
        
            # Print percentage of step done
            print(str(int(100*k/nbTimeSteps)) + ' % completed')

            # Print remaining simulation time (minutes and seconds)
            sim_time = (time.time() - simulation_time_start)*(nbTimeSteps/k - 1)

            print('Remaining time: ' + str(int(sim_time)//60) + ' minutes and ' + str(int(sim_time)%60) + ' seconds')

            count_communication = 0


# End simulation
sim_time = time.time() - simulation_time_start

# Print simulation time
print('Simulation took ' + str((int(sim_time)%3600)//60) + ' minutes and ' + str(int(sim_time)%60) + ' seconds')



# %% Landau-Teller ODE system

## Solve the corresponding Landau-Teller ODE system

# Compute the constant appearing in the Landau-Teller ODE
Cte_T = eps**2 * 12 / np.sqrt(np.pi)

# Right-hand side of the Landau-Teller ODE
def rhs_Landau_Teller(t,Ti):

    # Deduce kinetic temperature from equilibrium and internal temperatures
    Tk = 2 / 3 * (2.5 * Tf - Ti)

    return  Cte_T * np.abs(Tk)**(1.5) * Ti * (Tk - Ti)

# Solve the Landau-Teller equation with same initial internal temperature as DSMC
Landau_Teller_Ti = solve_ivp(rhs_Landau_Teller, [DSMC_time_vector[0],DSMC_time_vector[-1]], [Ti0], dense_output=True, method='Radau').sol

# Choose 1000 time points for the plot ranging from initial to final times of DSMC simulation
Landau_Teller_time_vector = np.linspace(DSMC_time_vector[0],DSMC_time_vector[-1],1000)

# Compute the solution on the selected time points
Landau_Teller_Ti_vector = Landau_Teller_Ti(Landau_Teller_time_vector)[0]

# %% Plots

# Plot the DSMC and Landau-Teller internal temperatures and the equilibrium temperature 
plt.figure()
plt.plot(DSMC_time_vector, DSMC_Ti_vector, color='black', linewidth = 3 , label='$T_i$ - DSMC')

plt.plot(Landau_Teller_time_vector, Landau_Teller_Ti_vector, color='white', linestyle='dashed', linewidth = 1, label = '$\overline{T}_i$ - Landau-Teller')

#plt.plot(Landau_Teller_time_vector, Tf * np.ones_like(Landau_Teller_time_vector),  '-.',label='$T_{eq}$', color='grey')

plt.legend(facecolor="lightgrey")

plt.xlabel("Time")
plt.ylabel("Temperature")


# Save plot

# plt.savefig('plot_betak0_eps10.pdf')


# Compute the relative error
Landau_Teller_Ti_full_vector = Landau_Teller_Ti(DSMC_time_vector)[0]

relative_error = np.abs(np.array(DSMC_Ti_vector) - Landau_Teller_Ti_full_vector) / Landau_Teller_Ti_full_vector

# Plot the relative error in semi-log scale except the first value (as it is zero)
plt.figure()
plt.semilogy(DSMC_time_vector[1:], relative_error[1:], color='black', linewidth = .7)

plt.xlabel("Time")
plt.ylabel("Relative error")


# plt.savefig('relative_error.pdf',bbox_inches = 'tight')

plt.show()








