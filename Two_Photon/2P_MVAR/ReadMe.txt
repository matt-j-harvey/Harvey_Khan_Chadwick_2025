MVAR Pipeline Documentation

Data is kept separate from MVAR results and intermediate files
have separate data_root directory, and mvar_output_root directory


1.) Preprocessing steps 
	Create Behaviour Matrix
	Downsample AI Matrix Framewise

2.) Create Regression Matricies (Getting data into format for MVAR)
	Create Activity Tensors - Tensor of shape (N_Trials, Trial_Length, N_Neurons), one for each trial type
	Create Behaviour Tensors - Tensor of shape (N_Trials, Trial_Length, N_Behaviour_Regressors), one for each trial type
	Combine Into Regression Matricies

3.) N_Fold Cross validation
	Perform Ridge penalty parameter search

4.) Fit Full Model

5.) Visualise Model Outputs