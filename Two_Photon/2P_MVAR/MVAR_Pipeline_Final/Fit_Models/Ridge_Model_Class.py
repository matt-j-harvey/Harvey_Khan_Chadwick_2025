import numpy as np

class ridge_model():

    def __init__(self, Nvar, Nstim, Nt, Nbehav, Ntrials, model_type, ridge_penalties):

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
        self.MVAR_parameters = None
        self.model_type = model_type
        self.stimulus_ridge_penalty = ridge_penalties[0]
        self.behaviour_ridge_penalty = ridge_penalties[1]
        if model_type != "No_Recurrent":
            self.interaction_ridge_penalty = ridge_penalties[2]

        # Create Regularisation Matrix (Tikhonov matrix)
        self.create_regularisation_matrix()


    def create_regularisation_matrix(self):

        if self.model_type == "Standard":
            self.Tikhonov = np.zeros([self.Nvar + self.Nstim * self.Nt + self.Nbehav, self.Nvar + self.Nstim * self.Nt + self.Nbehav])
            self.Tikhonov[0:self.Nvar, 0:self.Nvar] = np.sqrt(self.interaction_ridge_penalty) * np.eye(self.Nvar)
            self.Tikhonov[self.Nvar:(self.Nvar + self.Nstim * self.Nt), self.Nvar:(self.Nvar + self.Nstim * self.Nt)] = np.sqrt(self.stimulus_ridge_penalty) * np.eye(self.Nstim * self.Nt)
            self.Tikhonov[-self.Nbehav:, -self.Nbehav:] = np.sqrt(self.behaviour_ridge_penalty) * np.eye(np.sum(self.Nbehav))

        elif self.model_type == "No_Recurrent":
            self.Tikhonov = np.zeros([self.Nstim * self.Nt + self.Nbehav, self.Nstim * self.Nt + self.Nbehav])
            self.Tikhonov[0:(self.Nstim * self.Nt), 0:(self.Nstim * self.Nt)] = np.sqrt(self.stimulus_ridge_penalty) * np.eye(self.Nstim * self.Nt)
            self.Tikhonov[-self.Nbehav:, -self.Nbehav:] = np.sqrt(self.behaviour_ridge_penalty) * np.eye(np.sum(self.Nbehav))

        elif self.model_type == "Seperate_Contexts":
            self.Tikhonov = np.zeros([self.Nvar*2 + self.Nstim * self.Nt + self.Nbehav, self.Nvar*2 + self.Nstim * self.Nt + self.Nbehav])
            self.Tikhonov[0:self.Nvar*2, 0:self.Nvar*2] = np.sqrt(self.interaction_ridge_penalty) * np.eye(self.Nvar*2)
            self.Tikhonov[self.Nvar*2:(self.Nvar*2 + self.Nstim * self.Nt), self.Nvar*2:(self.Nvar*2 + self.Nstim * self.Nt)] = np.sqrt(self.stimulus_ridge_penalty) * np.eye(self.Nstim * self.Nt)
            self.Tikhonov[-self.Nbehav:, -self.Nbehav:] = np.sqrt(self.behaviour_ridge_penalty) * np.eye(np.sum(self.Nbehav))


    def fit(self, design_matrix, delta_f_matrix):
        delta_f_matrix = np.transpose(delta_f_matrix)

        ## Perform least squares fit with L2 penalty
        self.MVAR_parameters = np.linalg.solve(design_matrix.T @ design_matrix + self.Tikhonov.T @ self.Tikhonov, design_matrix.T @ delta_f_matrix.T)  # Tikhonov regularisation
        self.MVAR_parameters = self.MVAR_parameters.T


    def predict(self, design_matrix):
        self.prediction = np.matmul(self.MVAR_parameters, design_matrix.T)
        self.prediction = np.transpose(self.prediction)
        return self.prediction

