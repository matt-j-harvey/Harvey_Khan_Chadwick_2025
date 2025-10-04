import numpy as np

class ridge_model():

    def __init__(self, Nvar, Nstim, Nt, Nbehav, Ntrials, interaction_ridge_penalty, stimulus_ridge_penalty, behaviour_ridge_penalty):

        # inputs:
        # Nvar - Number of pixels or neurons
        # Nstim - Number of stimuli
        # Nt - Number of timepoints in each Trial
        # Nbehav - Number of behavioural predictor variables
        # dF_F = array of dF/F tensors for each stimulus [time x trials x vars]
        # timewindow = indices of time samples to select from dF_F (a window about stim onset, e.g. -1 to +1 s)
        # behaviour = array of behavioural measurement tensors for each stimulus [time x trials x behav_vars]
        # reg_coeff_weights = regularisation coefficient for weight matrix
        # reg_coeff_inputs  = regularisation coefficient for stimulus inputs
        # reg_coeff_behav   = regularisation coefficient for behavioural weights

        #  Save Initialisation Variables
        self.Nvar = Nvar
        self.Nstim = Nstim
        self.Nt = Nt
        self.Nbehav = Nbehav
        self.Ntrials = Ntrials
        self.stimulus_ridge_penalty = stimulus_ridge_penalty
        self.behaviour_ridge_penalty = behaviour_ridge_penalty
        self.interaction_ridge_penalty = interaction_ridge_penalty
        self.MVAR_parameters = None

        # Create Regularisation Matrix (Tikhonov matrix)

        self.Tikhonov = np.zeros([Nvar + Nstim * Nt + Nbehav, Nvar + Nstim * Nt + Nbehav])
        self.Tikhonov[0:Nvar, 0:Nvar] = np.sqrt(interaction_ridge_penalty) * np.eye(Nvar)
        self.Tikhonov[Nvar:(Nvar + Nstim * Nt), Nvar:(Nvar + Nstim * Nt)] = np.sqrt(stimulus_ridge_penalty) * np.eye(Nstim * Nt)
        self.Tikhonov[-Nbehav:, -Nbehav:] = np.sqrt(behaviour_ridge_penalty) * np.eye(np.sum(Nbehav))

    def fit(self, design_matrix, delta_f_matrix):
        delta_f_matrix = np.transpose(delta_f_matrix)

        ## Perform least squares fit with L2 penalty
        self.MVAR_parameters = np.linalg.solve(design_matrix.T @ design_matrix + self.Tikhonov.T @ self.Tikhonov, design_matrix.T @ delta_f_matrix.T)  # Tikhonov regularisation
        self.MVAR_parameters = self.MVAR_parameters.T


    def predict(self, design_matrix):
        self.prediction = np.matmul(self.MVAR_parameters, design_matrix.T)
        self.prediction = np.transpose(self.prediction)
        return self.prediction

