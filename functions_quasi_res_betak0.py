import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import gamma as gammafunc
from numba import jit


@jit(nopython=True)
def compute_Bmax(Ekmax, Imax):

    return  2 * Ekmax**(.5) * (Ekmax + 2*Imax)**2

@jit(nopython=True)
def compute_ratio(Ek, sumI, Ekmax, Imax):

    return np.sqrt(Ek/Ekmax) * ((Ek + sumI)/(Ekmax + 2*Imax))**2


@jit(nopython=True)
def update_states(v, v_star, sumI, Energy, mass, alpha, eps):

    # Start with updating the internal energies
    R0 = sumI / Energy
    R0m = R0 / (R0 + np.exp(eps)*(1-R0))
    R0p = R0 / (R0 + np.exp(-eps)*(1-R0))

    U = rd.rand()
    R = (R0m**(alpha + 1) + (R0p**(alpha + 1) - R0m**(alpha + 1))*U )**(1 / (alpha + 1))

    r = rd.rand()         # draw uniformly another number for the repartition


    I_prime = r * R * Energy
    I_prime_star = (1 - r) * R * Energy

    q_prime = I_prime**(alpha + 1)
    q_prime_star = I_prime_star**(alpha + 1)

    # Draw an angle uniformly on the sphere

    theta = 2 * np.pi * rd.rand()
    phi = np.arccos(2 * rd.rand() - 1)

    sigma = np.zeros(3)
    sigma[0] = np.cos(theta) * np.sin(phi)
    sigma[1] = np.sin(theta) * np.sin(phi)
    sigma[2] = np.cos(phi)




    # update the velocities accordingly

    v_prime = (v + v_star) / 2 + np.sqrt((1-R)*Energy / mass) * sigma
    v_prime_star = v + v_star - v_prime


    return v_prime, v_prime_star, q_prime, q_prime_star



@jit(nopython=True)
def one_step(percent_coll, V_vectorin, q_vectorin, mass, alpha, eps):

    # Set variables
    V_vector = V_vectorin.copy()
    q_vector = q_vectorin.copy()
    N = len(q_vector)

    # Compute max quantities

    Ekmax = mass / 4 * ((np.max(V_vector[:,0]) - np.min(V_vector[:,0]))**2 + (np.max(V_vector[:,1]) - np.min(V_vector[:,1]))**2 + (np.max(V_vector[:,2]) - np.min(V_vector[:,2]))**2)


    Imax = np.max(q_vector**(1/(alpha+1)))

    tau_min = 1 / compute_Bmax(Ekmax, Imax)

    # Choose colliding pairs

    selected_pairs = rd.choice(N**2, int(N * percent_coll * .5))         # with redundancy 2)

    for k in selected_pairs:

        n = k // N                         # choice 2)
        m = k%N


        v = V_vector[n]
        q = q_vector[n]

        v_star = V_vector[m]
        q_star = q_vector[m]


        Ek = mass / 4 * np.sum((v-v_star)**2)
        sumI = q**(1/(alpha+1)) + q_star**(1/(alpha+1))

        Energy = Ek + sumI

        ratio = compute_ratio(Ek, sumI, Ekmax, Imax)

        if rd.rand() < ratio :

        # Then a collision happens, update the states

            v_prime, v_prime_star, q_prime, q_prime_star = update_states(v, v_star, sumI, Energy, mass, alpha, eps)

            V_vector[n] = v_prime
            V_vector[m] = v_prime_star
            q_vector[n] = q_prime
            q_vector[m] = q_prime_star


    return V_vector, q_vector, percent_coll * tau_min



@jit(nopython=True)
def one_step_opti(V_vector, q_vector, N, exp_eps):

    # Compute exp(-eps) once and for all

    exp_minus_eps = 1 / exp_eps

    # Compute max quantities

    Ekmax = ((np.max(V_vector[:,0]) - np.min(V_vector[:,0]))**2 + (np.max(V_vector[:,1]) - np.min(V_vector[:,1]))**2 + (np.max(V_vector[:,2]) - np.min(V_vector[:,2]))**2) / 4

    Imax = np.max(q_vector)

    tau_min = 1 / (2 * np.sqrt(Ekmax) * (Ekmax + 2*Imax)**2)


    # Choose colliding pairs

    selected_pairs = rd.choice(N*(N-1)//2, N//2)

    # selected_pairs = rd.randint()
    random_numbers_for_test = rd.rand(N//2)

    for k in range(N//2):

        n = int(np.floor(np.sqrt(2*selected_pairs[k] + .25) - .5) )    # find n such that n(n+1)/2 <= k < (n+1)(n+2)/2
        m = selected_pairs[k] - n*(n+1)//2


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

            sigma = np.empty(3)
            sigma[0] = np.cos(theta) * sin_phi
            sigma[1] = np.sin(theta) * sin_phi
            sigma[2] = cos_phi


            # update the velocities accordingly

            mean_v = (v + v_star) / 2
            change_sigma = np.sqrt((1-R)*Energy) * sigma

            V_vector[n] = mean_v + change_sigma
            V_vector[m] = mean_v - change_sigma

    return V_vector, q_vector, tau_min
