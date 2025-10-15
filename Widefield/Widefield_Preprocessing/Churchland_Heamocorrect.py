import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import h5py
import os
import pickle
from scipy import ndimage
from tqdm import tqdm


def lowpass(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=10000)

def highpass(X, w = 0.0033, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=10000)




def hemodynamic_correction(U, SVT_470, SVT_405,
                           fs=30.,
                           freq_lowpass=14.,
                           freq_highpass=0.1,
                           nchunks=1024,
                           run_parallel=False):

    # split channels and subtract the mean to each
    SVTa = SVT_470  # [:,0::2]
    SVTb = SVT_405  # [:,1::2]

    # reshape U
    dims = U.shape
    U = U.reshape([-1, dims[-1]])
    print("flat u", np.shape(U))

    # Single channel sampling rate
    fs = fs

    # Highpass filter
    if not freq_highpass is None:
        SVTa = highpass(SVTa, w=freq_highpass, fs=fs)
        SVTb = highpass(SVTb, w=freq_highpass, fs=fs)

    if not freq_lowpass is None:
        if freq_lowpass < fs / 2:
            SVTa = lowpass(SVTa, freq_lowpass, fs=fs)
            SVTb = lowpass(SVTb, freq_lowpass, fs=fs)
        else:
            print('Skipping lowpass on the violet channel.')

    # subtract the mean
    SVTa = (SVTa.T - np.nanmean(SVTa, axis=1)).T.astype('float32')
    SVTb = (SVTb.T - np.nanmean(SVTb, axis=1)).T.astype('float32')

    npix = U.shape[0]
    print("n pix", npix)
    idx = np.array_split(np.arange(0, npix), nchunks)

    # find the coefficients
    rcoeffs = np.zeros((npix))
    for i, ind in enumerate(idx):

        # rcoeffs[ind] = _hemodynamic_find_coeffs(U[ind,:],SVTa,SVTb)
        a = np.dot(U[ind, :], SVTa)
        b = np.dot(U[ind, :], SVTb)
        rcoeffs[ind] = np.sum(a * b, axis=1) / np.sum(b * b, axis=1)

    # drop nan
    rcoeffs[np.isnan(rcoeffs)] = 1.e-10

    # find the transformation
    print("r coefs shape", np.shape(rcoeffs))
    print("u shape", np.shape(U))
    T = np.dot(np.linalg.pinv(U), (U.T * rcoeffs).T)

    # apply correction
    SVTcorr = SVTa - np.dot(T, SVTb)

    # return a zero mean SVT
    SVTcorr = (SVTcorr.T - np.nanmean(SVTcorr, axis=1)).T.astype('float32')

    # put U dims back in case its used sequentially
    U = U.reshape(dims)

    return SVTcorr.astype('float32'), rcoeffs.astype('float32').reshape(dims[:2]), T.astype('float32')

