#=======================================================
# KURAMOTO MODEL Modified by Connor Braun, May 2020
#=======================================================

# Kuramoto class originally authored by Dr. Dawid Laszuk, modified by Connor Braun for research/personal interest purposes
#----------- REFERENCE------------------------------------------------------------
#   D. Laszuk, "Python implementation of Kuramoto systems," 2017-,
#   [Online] Available: http://www.laszukdawid.com/codes
#-----------ORIGINAL PROGRAM LICENCE----------------------------------------------
# This program is free software on GNU General Public Licence version 3.
# For details of the copyright please see: http://www.gnu.org/licenses/.
#---------------------------------------------------------------------------------
# See this link for a *very* detailed guide on the Kuramoto model: http://scala.uc3m.es/publications_MANS/PDF/finalKura.pdf
# See this page for a quick explanation of the Kuramoto Model: http://go.owu.edu/~physics/StudentResearch/2005/BryanDaniels/index.html

import time
import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import ode
from datetime import datetime
from cmath import *
from notion.client import NotionClient

class Kuramoto:
    _noises = { 'logistic': np.random.logistic,
            'normal': np.random.normal,
            'uniform': np.random.uniform,
            'custom': None
        }
    noise_types = _noises.keys()
    # -----------------------------
    # KURAMOTO OBJECT CONSTRUCTOR
    # -----------------------------
    def __init__(self, initialValuesDictionary, noise=None, alpha=0):
        # Construct a Kuramoto object using a dictionary that maps Y0 to initial phase array, W to natural frequency array and K to a coupling matrix.
        # Y0 and W arrays are of dimension 1 x n, where n = number of oscillators in network
        self.dtype = np.float32 # we'll use this attribute to change the precision of the unit W's later
        self.dt = 1 #initialize to prevent division by 0
        self.initialPhases = np.array(initialValuesDictionary['Y0'])
        self.naturalFrequencies = np.array(initialValuesDictionary['W'])
        self.adjacencyMatrix = np.array(initialValuesDictionary['K'])
        self.numberOfOscillators = len(self.naturalFrequencies)
        #UNSURE WHAT mOrder MEANS, USED IN JACOBIAN COMPUTATION
        self.mOrder = self.adjacencyMatrix.shape[0] # idk WHY we'd have >1 matrix, but when we do (sps: A1 and A2) we pass in K = np.dstack((A1, A2)).T s.t. K.shape[0] returns the number of concatenated matrices (2, in this case).
        self.noise = noise
        self.alpha = alpha
    @property
    def noise(self):
        # Sets perturbations added to the system at each timestamp. Noise function can be manually defined or selected from predefined by assgining corresponding name. List of available pertrubations is reachable through `noise_types`
        return self._noise
    @noise.setter
    def noise(self, _noise):
        self._noise = None
        self.noise_parameters = None
        self.noise_type = 'custom'
        
        # If a function was passed
        if callable(_noise):
            self._noise = _noise
        # If a string was passed
        elif isinstance(_noise, str):
            if _noise.lower() not in self.noise_types:
                self.noise_type = None
                raise NameError('No such noise method comes predefined in the model, acceptable strings include: logistic, normal or uniform')
            self.noise_type = _noise.lower()
            self.update_noise_parameters(self.dt)

            noise_function = self._noises[self.noise_type]
            self._noise = lambda: np.array([noise_function(**p) for p in self.noise_parameters])
    
    def update_noise_parameters(self, dt):
        scale = []
        for W in self.naturalFrequencies:
            if W != 0:
                scale.append(dt/np.abs(W**2))
            else:
                scale.append(dt)
        scale = np.array(scale)
        print('so-called scale is {}'.format(scale))

        if self.noise_type == 'uniform':
            self.noise_parameters = [{'low':-s, 'high': s} for s in scale]
        elif self.noise_type in self.noise_types:
            self.noise_parameters = [{'loc': 0, 'scale': s} for s in scale]
        else:
            pass
    
    # ----------------------------------------
    # MODEL INTEGRATOR (Runge-Kutta method)
    # ----------------------------------------
    def solve(self, t):
        dt = t[1] - t[0] # Take discrete timestep between first two entries of time axis
        if self.dt != dt and self.noise_type != 'custom': 
            self.dt = dt # Ensure the object has appropriate dt attribute
            self.update_noise_parameters(dt)
        
        #Fire up the kuramoto model as an integratable scipy.integrate.ode object
        kODEObject = ode(self.kuramotoODE)
        kODEObject.set_integrator('dopri5', nsteps=100000) # an integrator that uses the Runge-Kutta method
        #kODEObject.set_jac_params((self.naturalFrequencies, self.adjacencyMatrix))
        kODEObject.set_initial_value(self.initialPhases, t[0]) # setting initial phases wants the state variable and THEN time... this is how t = t0 makes it into the output so we'll exclude it from subsequent iterations
        kODEObject.set_f_params((self.naturalFrequencies, self.adjacencyMatrix)) # Give it the network parameters
        
        if self._noise != None:
            self.update_noise_parameters(dt)

        phaseOutput = np.empty((self.numberOfOscillators, len(t))) # Empty array with a row for each oscillator and a column for every time point

        #Loop through each column of the phase output (starting at t[1], recall t[0] is given by Y0)
        # enumerate turns every entry into a tuple where the first entry is an integer value for index... so we get two iteration variables for the price of one 
        for index, timePoint in enumerate(t[1:]): 
            # ***QUERY***: Why do these two lines occur in this order? Wouldn't t[1] have the same phase as at t[0]?
            phaseOutput[:, index] = kODEObject.y  # for all three rows (oscillators) simultaneously, store their respective phases at the current timepoint 
            kODEObject.integrate(timePoint) # integrate up to the present time step to get the current phase for all three oscillators... sets kODEObject.y = current phases
        phaseOutput[:, -1] = kODEObject.y # Get the last column in the phaseOutput that was missed in the for loop

        return phaseOutput
    # ----------------------------------
    # DEFINITION OF KURAMOTO EQUATION
    # ----------------------------------
    def kuramotoODE(self, t, y, arg): 
        # An image of the equation: http://go.owu.edu/~physics/StudentResearch/2005/BryanDaniels/kuramoto_equation.JPG
        # All class methods start with self parameter, t and y are required (in that order) to work as a scipy.integrate.ode object, ode objects support *f_args but we only need the one (a tuple with frequency and adjacency matrices)
        w, k = arg # see kODEObject.set_f_params() for why this order works
        # ***QUERY***: What does 'None' specify here? ***ANSWER***: see https://stackoverflow.com/questions/40574982/numpy-index-slice-with-none
        yt = y[:, None] # Create 2D column of the three current phases 
        # ***QUERY***: what's dy? why does this operation work? ***ANSWER***: dy is a square matrix with entries yt(ij) which are 0j-0i for each oscillator pair at time t.
        dy = y - yt # This is a 2D, 3x1 array being subtracted from a 1D array of three values. 
        # ***Query***: Why should frequencies be lower precision? Why does phase start out as the frequency (maybe its rate of change of phase)?
        phaseDot = w.astype(self.dtype) # Cast the array of frequencies as float32
        if self.noise != None:
            xi_t = self.noise().astype(self.dtype)
            phaseDot += xi_t
        # ***Query***: This for loop is majorly confusing... especially without knowing what yt or phaseDot are (not even sure if phaseDot is a correct name descriptively...)
        for index, rowInK in enumerate(k): # See scrapPaper.py for some notion of how this thing works
            nonlinearity = np.sin(dy - self.alpha) # Take the sin of the phase differences (0j - 0i)
            phaseDot += np.sum(rowInK*nonlinearity, axis=1) # Scale columns by the appropriate coupling weight, then sum up the ROWS. Add these to the rate of phase change!
        return phaseDot
    # def kuramotoODEJacobian(self, t, y, arg):
    #     w, k = arg
    #     yt = y[:, None]
    #     dy = y - yt
    #     phase = [m*k[m-1]*np.cos(m*dy) for m in range(1,1+self.mOrder)]
    #     phase = np.sum(phase, axis=0)

    #     for i in range(self.numberOfOscillators):
    #         phase[i,i] = -np.sum(phase[:,i])

    #     return phase

def formNetwork(k_11, k_12, k_13, k_21, k_22, k_23, k_31, k_32, k_33):
    # This function merely arranges the input into the appropriate matrix for desired network structure
    adjacencyMatrix = np.array([[k_11, k_12, k_13], 
                                [k_21, k_22, k_23], 
                                [k_31, k_32, k_33]])
    return adjacencyMatrix
def setParameters(t0, tEND, dt, initialPhaseArgsLIST, naturalFrequencyArgsLIST, customCouplingCoefficients):
    # Wraps initial values into a linearly spaced time vector and a parameter dictionary to generate Kuramoto object
    # Create an array with linearly spaced time points. Units must be reciprocal to frequency
    timeAxis = np.arange(t0, tEND, dt) 

    # Convert to a NumPy array
    initialPhases = np.array(initialPhaseArgsLIST)
    print("{} points to compute...".format(len(initialPhases)*((tEND - t0)/dt))) # provides some notion of how challenging this is to compute

    # Convert to a NumPy array
    naturalFrequencies = np.array(naturalFrequencyArgsLIST)

    adjacencyMatrix = formNetwork(customCouplingCoefficients[0], customCouplingCoefficients[1], customCouplingCoefficients[2], customCouplingCoefficients[3], customCouplingCoefficients[4], customCouplingCoefficients[5], customCouplingCoefficients[6], customCouplingCoefficients[7], customCouplingCoefficients[8])
    
    # Wrap parameters as a dictionary with order and variable names conducive to kuramoto object construction
    parameters = timeAxis, {'Y0': initialPhases, 'W': naturalFrequencies, 'K': adjacencyMatrix}

    return parameters
def assessSynchrony(phase_T, oscillatorsOfInterest, iterationNumber, workingDirectory, manimFriendly=False):

    # Phase offset series is taken as the phase difference between the 2 SSO's at each time point
    leftOscillator = phase_T[oscillatorsOfInterest[0]] % 2*np.pi
    rightOscillator = phase_T[oscillatorsOfInterest[1]] % 2*np.pi
    phaseOffsetList = np.subtract(leftOscillator, rightOscillator)
    vectorList = []

    # Compute PLV between oscillators of interest
    for offset in phaseOffsetList:
        polarVector = exp(offset*1j)
        vectorList.append(polarVector)
    totalVectorModulus = abs(sum(vectorList))
    PLV = totalVectorModulus/len(vectorList)

    # Wrap the difference on [-pi, 0) U (0, pi] for histogram (if you dont then the ends of the histogram both represent vanishing phase difference)
    for index, offset in enumerate(phaseOffsetList, start=0):
        if offset < -np.pi: # If the vector is in the upper hemisphere but NEGATIVE
            phaseOffsetList[index] = -(offset + 2*np.pi) # Add 2pi to make positive without moving, then reflect over real axis to the lower hemisphere
        elif offset > np.pi: # If the vector is in the lower hemisphere and POSITIVE
            phaseOffsetList[index] = -(offset - 2*np.pi) # Subtract 2pi to make negative without moving, then reflect over real axis to the upper hemisphere

    # Construct a histogram
    if not manimFriendly:
        plt.close()
        fig, ax = plt.subplots()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        count, bins, patches = ax.hist(phaseOffsetList, bins=20, color='k', range=(-np.pi, np.pi), density=True, alpha=0.7, rwidth=0.85) # Narrow bars, black color
        #plt.title('Kuramoto model phase offset probability distribution') # Title with abf file name for now
        ax.set_xlabel(r'$\theta_L - \theta_R$') # Symbology for phase difference
        ax.set_ylabel('Probability')

        xLabelList = [r" ", r"$-\pi$", r"$-\frac{2}{3}\pi$", r"$-\frac{1}{3}\pi$", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$"]
        ax.set_xticklabels(xLabelList)

        plt.savefig('{}\\modelFigures\\Histogram_iteration{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')

    else:
        manimFriendlyHistogram(phaseOffsetList, iterationNumber, workingDirectory)

    return PLV, phaseOffsetList
def manimFriendlyHistogram(phaseOffsetList, iterationNumber, workingDirectory):
    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    probabilities, bins, patches = ax.hist(phaseOffsetList, range=(-np.pi, np.pi), bins=20, color='w', density=True, rwidth=0.85) # Narrow bars, black color

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color='w', labelcolor='w')
    for spine in ax.spines.values():
        spine.set_edgecolor('w')
    
    xLabelList = [r" ", r"$-\pi$", r"$-\frac{2}{3}\pi$", r"$-\frac{1}{3}\pi$", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$"]
    ax.set_xticklabels(xLabelList)

    ax.set_xlabel(r'$\theta_L - \theta_R$', color='w') # Symbology for phase difference
    ax.set_ylabel('Probability', color='w')

    plt.savefig('{}\\modelFigures\\Histogram_iteration{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')
def manimFriendlyOffsetCurve(timeVector, phaseOffsetList, iterationNumber, workingDirectory):
    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.plot(timeVector, phaseOffsetList, linewidth=0.3, color='w')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color='w', labelcolor='w')
    for spine in ax.spines.values():
        spine.set_edgecolor('w')

    ax.set_xlabel('$Time$ $(s)$', color='w') # Symbology for phase difference
    ax.set_ylabel(r'$\theta_L - \theta_R$', color='w')

    plt.savefig('{}\\modelFigures\\offsetCurve_iteration{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')
def updateLog(iterationNumber, couplingConfiguration, beta, naturalFrequencyList, omega_3, timeAxisLength, analyzedTimeLength, PLV, alpha, betaStationarity=True, omegaStationarity=True, init_Entry=False):
    
    dateAndTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    outfile = open('modelLog.txt', 'a')

    if init_Entry:
        outfile.write('=======================\n{}\n=======================\nINITIALIZING:\nCoupling Configuration: {}\nNatural Frequencies: {}\nAlpha: {}\nSimulating {}s of data; analyzing last {}s\n'.format(dateAndTime, couplingConfiguration, naturalFrequencyList, alpha, timeAxisLength, analyzedTimeLength))
    else:
        if betaStationarity == False and omegaStationarity == True:
            outfile.write('ITERATION {}; Beta = {}: PLV = {}\n'.format(iterationNumber, beta, PLV))
        elif betaStationarity == True and omegaStationarity == False:
            outfile.write('ITERATION {}; Brainstem Omega = {}: PLV = {}\n'.format(iterationNumber, omega_3, PLV))
        elif betaStationarity == False and omegaStationarity == False:
            outfile.write('ITERATION {}; Beta = {}; Brainstem Omega = {}: PLV = {}\n'.format(iterationNumber, beta, omega_3, PLV))
        else:
            outfile.write('ITERATION {}: PLV = {}\n'.format(iterationNumber, PLV))

    outfile.close()
def unwrappedAnalysis(phase_T, timeVector, oscillatorsOfInterest, iterationNumber, workingDirectory, viewFigure=False, manimFriendly=False):
    
    leftOscillator = phase_T[oscillatorsOfInterest[0]]
    rightOscillator = phase_T[oscillatorsOfInterest[1]]
    phaseOffsetList = np.subtract(leftOscillator, rightOscillator)

    if not manimFriendly:
        plt.close()
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.plot(timeVector, phaseOffsetList, linewidth=0.3, color='deeppink')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel('$Time$ $(s)$') # Symbology for phase difference
        ax.set_ylabel(r'$\theta_L - \theta_R$', color='k')

        plt.savefig('{}\\modelFigures\\offsetPlot_iteration{}.png'.format(workingDirectory, iterationNumber), bbox_inches='tight')

        if viewFigure:
            plt.show()
    else:
        manimFriendlyOffsetCurve(timeVector, phaseOffsetList, iterationNumber, workingDirectory)
def notionInitialize(notion_token_v2, pageURL="None", collectionURL="None"):
    DEPTHCOUNTER = 0
    client = NotionClient(token_v2=notion_token_v2) # open the client using a token (find using Chrome developer console: Application --> Cookies)
    if pageURL:
        page = client.get_block(pageURL)
        DEPTHCOUNTER += 1
    else:
        print("notionInitialize() RETURNING: NOTION CLIENT")
        return client
    if collectionURL:
        cv = client.get_collection_view(collectionURL)
        DEPTHCOUNTER += 1
    else:
        print("notionInitialize() RETURNING: NOTION PAGE")
        return page
    print("notionInitialize() RETURNING: COLLECTION")
    return cv
def notionLog(notion_cv, iterationNumber, couplingConfiguration, beta, naturalFrequencyList, omega_3, timeAxisLength, analyzedTimeLength, PLV, alpha, simulationStartTime):

    networkConfig_dictionary = {"BL": [1, 2, 3, 5, 6, 7], "BT": [1, 3], "MT": [2, 5, 6, 7]}
    indexList = []
    for slot, connection in enumerate(couplingConfiguration, start=0):
        if connection > 0:
            indexList.append(slot)
    for key in networkConfig_dictionary.keys():
        if networkConfig_dictionary[key] == indexList:
            networkConfig = key
            break  
    
    newEntry = notion_cv.collection.add_row()
    newEntry.date_iterationnumber = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": {}".format(iterationNumber)
    newEntry.oscillator_configuration = networkConfig
    newEntry.plv = PLV
    newEntry.alpha = alpha
    newEntry.beta = beta
    newEntry.omega_1 = naturalFrequencyList[0]
    newEntry.omega_2 = naturalFrequencyList[1]
    newEntry.omega_3 = omega_3
    newEntry.timeseries_duration = timeAxisLength
    newEntry.window_of_analysis_duration = analyzedTimeLength

def main():
    # =========== INITIALIZE GLOBAL CONSTRUCTS =========== #
    SIMULATION_START_TIME = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    manimFriendly = True
    beta = -1
    PLV = -1
    STATIONARY_BETA = True
    STATIONARY_OMEGA_3 = True
    iterationNumber = 0
    viewPhaselocking = False
    WORKING_DIRECTORY = os.path.dirname(os.path.realpath("kuramotoModel_Research\\modelFigures")).encode('unicode-escape').decode()
    NOTION_TOKEN_V2 = "058471a79978470c42ae3a722a4006617d4bc91f1030ee49b42101513296a2019dacc910d911b2a211500c8b89ea8e48e3d836ab88a8101200f8651802721b1ead7fe8497102145d5f3df26001e9"
    NOTION_PAGE_URL = "https://www.notion.so/kuramoto3_Simulations-de870a1e94554a54b4e8b7cbc9052b2c"
    NOTION_COLLECTION_URL = "https://www.notion.so/70a85ab49e5347cb998a2f3212f4cdef?v=ebd043a6bdc3485a8f35ac00d40a24d6"
    # ==================================================== #
    # =========== CONTROL PANEL: HARDCODE PARAMETERS =========== #
                        #--- k_11, k_12, k_13, k_21, k_22, k_23, k_31, k_32, k_33 ---#
    couplingConfiguration = [ 0,    1,    0,    1,    0,    0,    0,    0,    0] # Values of 1 (integer dtype) will be scaled by beta linspace
        #--- Left   Right  Brainstem ---#
    omega = [0.871, 0.832, 0] # Natural frequencies of network SSO's; Brainstem values of 1 (integer dtype) will be scaled by omega3 linspace
    omega_3 = omega[2]
    # Indices of natural frequencies in omega corresponding to oscillators whose phase relationship is of interest
    OSCILLATORS_TO_COMPARE = [0, 1]
    NOISE_TYPE = None
    ALPHA = 1.5902 # PREFERRED OFFSET AS SEEN IN EXPERIMENT
    t0 = 0 # Start of time axis
    tEND = 1800 # End of time axis
    dt = 0.001 # Discrete time step
    initialPhases = [0, 0, 0] # list of initial phases
    len_asymptoticTimeVector = 600
    num_ITERATIONS = 25 # NOTE: WHEN THIS IS 1 THEN linspaceSTART_1/2 WILL BE USED, NOT linspaceEND_1/2

    # Linspace 1 determines beta parameter sampling
    linspaceSTART_1 = 0.75
    linspaceEND_1 = 1.0

    # Linspace 2 determines omega3 parameter sampling
    linspaceSTART_2 = 1.5902
    linspaceEND_2 = 1.5902
    # ========================================================== #

    # Initialize model log prior to iteration
    updateLog(iterationNumber, couplingConfiguration, beta, omega, omega_3, tEND, len_asymptoticTimeVector, PLV, ALPHA, init_Entry=True)

    # =========== CONTROL PANEL: INFER PARAMETERS =========== #
    vectorSpace_1 = np.around(np.linspace(linspaceSTART_1, linspaceEND_1, num=num_ITERATIONS), decimals=4).tolist()
    vectorSpace_2 = np.around(np.linspace(linspaceSTART_2, linspaceEND_2, num=num_ITERATIONS), decimals=4).tolist()
    timeAxis_Length =  int((1/dt)*(tEND - t0))
    timeAxis_Asymptotic_Length = int((1/dt)*len_asymptoticTimeVector)
    endDynamicsSTART = timeAxis_Length - timeAxis_Asymptotic_Length
    timeAxis_Asymptotic = np.arange(0, len_asymptoticTimeVector, dt) 
    # ======================================================= #
    # =========== ITERATE OVER PARAMETER SPACE =========== #
    for iteration, beta in enumerate(vectorSpace_1, start=1): # Iterate Kuramoto model n times, where n is the number of entries in the beta linspace
        startTime = time.time()
        for index, binary in enumerate(couplingConfiguration, start=0): # For each entry in the coupling configuration list
            if binary == 1: # If the connection is binary AND one...
                couplingConfiguration[index] = binary*beta # Scale the entry by beta
                if STATIONARY_BETA == True:
                    STATIONARY_BETA = False # If even one entry gets scaled, then beta is nonstationary over the space
        if omega[2] == 1: # If the brainstem oscillator has a frequency of 1 (integer dtype)...
            omega[2] = omega[2]*vectorSpace_2[iteration - 1] # Scale the frequency by omega3
            omega_3 = omega[2]
            if STATIONARY_OMEGA_3 == True:
                STATIONARY_OMEGA_3 = False # If brainstem oscillator frequency is scaled even once, then omega3 is nonstationary over the space
        if ALPHA == 1: # If preferered offset is set to be 1 (integer dtype)
            ALPHA = ALPHA*vectorSpace_2[iteration - 1] # Scale the offset by the current entry in the second vectorspace
        
        # Assign names to the major parameter variables
        timeAxis, parameterDictionary = setParameters(t0, tEND, dt, initialPhases, omega, customCouplingCoefficients=couplingConfiguration) # Wrap up parameters as dicitonary to pass into model class generator
        adjacencyMatrix = parameterDictionary.get('K') # Get the adjacency matrix as packaged in setParameters()

        # Print the parameters prior to model execution
        print("Initial Parameters:\nInitial phases: {}\nNatural frequencies: {}\nCoupling matrix: \n{}".format(parameterDictionary['Y0'], parameterDictionary['W'], parameterDictionary['K']))

        # Produce a kuramoto model object with scipy.integrate.ode; integration at time t is executed using the ode.integrate method (dopri5)
        kuramotoModel = Kuramoto(parameterDictionary, noise=NOISE_TYPE, alpha=ALPHA) # Create an instance of a Kuramoto system
        phase_T = kuramotoModel.solve(timeAxis) # phase_T = 0(t); solve it using the timeAxis created earlier

        # Compute LHS of Kuramoto ODE as a function of time
        phaseDot_T = np.diff(phase_T)/dt # phaseDot_t = d0/dt as a function of time as estimated by the finite difference
        print("===== {} m {} s to compute dynamics =====".format((time.time() - startTime)//60, ((time.time() - startTime)%60)//1))

        phase_T_Asymptotic = phase_T[:, endDynamicsSTART:] # Slice model output to only consider endpoint dynamics

        PLV, phaseOffsetList = assessSynchrony(phase_T_Asymptotic, OSCILLATORS_TO_COMPARE, iteration, WORKING_DIRECTORY, manimFriendly=manimFriendly) # Compute phase offset series for two SSO's and return PLV
        print('Iteration {} PLV: {}'.format(iteration, PLV)) # Quick printout for operator
        unwrappedAnalysis(phase_T_Asymptotic, timeAxis_Asymptotic, OSCILLATORS_TO_COMPARE, iteration, WORKING_DIRECTORY, viewFigure=viewPhaselocking, manimFriendly=manimFriendly)

        # Update model log after current iteration
        updateLog(iteration, couplingConfiguration, beta, omega, omega_3, tEND, len_asymptoticTimeVector, PLV, ALPHA, betaStationarity=STATIONARY_BETA, omegaStationarity=STATIONARY_OMEGA_3, init_Entry=False)
        notionCollection = notionInitialize(NOTION_TOKEN_V2, pageURL=NOTION_PAGE_URL, collectionURL=NOTION_COLLECTION_URL)
        notionLog(notionCollection, iteration, couplingConfiguration, beta, omega, omega_3, tEND, len_asymptoticTimeVector, PLV, ALPHA, SIMULATION_START_TIME)

        # Invert scaling operations s.t. next the next iteration will recognize which entries to scale
        for index, entry in enumerate(couplingConfiguration, start=0): # For each entry in the model's coupling configuration
            if entry == beta: # If that entry is beta, then ASSUME it used to be 1 (integer dtype)
                couplingConfiguration[index] = 1 # Make it 1 again so that the program recognizes it as a nonstationary parameter
        if omega[2] == vectorSpace_2[iteration - 1]: # If the brainstem oscillator omega is omega3, ASSUME it was scaled to be so
            omega[2] = 1 # Make it 1 again so that the program recognizes it as a nonstationary parameter
        if ALPHA == vectorSpace_2[iteration - 1]: # If ALPHA is the current nonstationary scalar the assume is was scaled to be so
            ALPHA = 1 # Make it 1 again so that the program recognizes it as a nonstationary parameter
    # ==================================================== #

if __name__ == '__main__':
    main()