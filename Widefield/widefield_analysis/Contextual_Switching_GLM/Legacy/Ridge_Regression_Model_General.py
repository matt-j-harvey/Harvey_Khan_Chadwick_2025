import os
import numpy as np
from sklearn.linear_model import Ridge

from Ridge_Regression_Model import Get_Cross_Validated_Ridge_Penalties


def fit_ridge_model(delta_f_matrix, design_matrix, save_directory):

    # Get Cross Validated Ridge Penalties
    ridge_penalty_list = Get_Cross_Validated_Ridge_Penalties.get_cross_validated_ridge_penalties(design_matrix, delta_f_matrix)

    # Create Model
    model = Ridge(solver='auto', alpha=ridge_penalty_list)

    # Fit Model
    model.fit(y=delta_f_matrix, X=design_matrix)

    # Get Coefs
    regression_coefs_list = model.coef_
    regression_intercepts_list = model.intercept_

    # Save These

    # Create Regression Dictionary
    regression_dict = {
        "Coefs": regression_coefs_list,
        "Intercepts": regression_intercepts_list,
        "Ridge_Penalties": ridge_penalty_list,
    }

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Regression_Dictionary_Simple.npy"), regression_dict)



