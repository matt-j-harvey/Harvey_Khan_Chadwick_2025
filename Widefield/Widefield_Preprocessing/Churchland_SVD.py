import Load_Binary_File
import matplotlib.pyplot as plt
import numpy as np
import Visualise_Raw_Data
from sklearn.preprocessing import normalize
from tqdm import tqdm
from numpy.linalg import svd
import os
import h5py
from scipy import ndimage

def reconstruct_data(base_directory, data):

    # Load Mask
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]

    reconstructed_data = []
    for frame in data:
        template = np.zeros(image_width * image_height)
        template[indicies] = frame
        template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)
        reconstructed_data.append(template)

    reconstructed_data = np.array(reconstructed_data)
    return reconstructed_data


def get_trial_baseline(idx,frames_average,onsets):
    if len(frames_average.shape) <= 3:
        return frames_average
    else:
        if onsets is None:
            print(' Trial onsets not defined, using the first trial')
            return frames_average[0]
        return frames_average[np.where(onsets<=idx)[0][-1]]


def approximate_svd(dat, frames_average,
                    onsets=None,
                    k=200,
                    nframes_per_bin=30,
                    nbinned_frames=5000,
                    nframes_per_chunk=500,
                    divide_by_average=True):

    ## No documentation on this, but looks like frames average should be, either a single average for each frame, or a list of averages for each trial

    '''
    Approximate single value decomposition by estimating U from the average movie and using it to compute S.VT.
    This is similar to what described in Steinmetz et al. 2017
    Joao Couto - March 2020

    This computes the mean centered SVD of the dataset, it does not compute F-F0/F0 a.k.a. df/f.
    Compute it after using the SVD components.
    Inputs:
        dat (array)             : (NFRAMES, NCHANNEL, H, W)
        k (int)                 : number of components to estimate (200)
        nframes_per_bin (int)   : number of frames to estimate the initial U components
        nbinned_frames (int)    : maximum number frames to estimate tje initial U components
        nframes_per_chunk (int) : window size to load to memory each time.
    Returns:
        U   (array)             :
        SVT (array)             :
    '''


    dims = dat.shape[1:]

    # the number of bins needs to be larger than k because of the number of components.
    if nbinned_frames < k:
        nframes_per_bin = np.clip(int(np.floor(len(dat) / (k))), 1, nframes_per_bin)

    nbinned_frames = np.min([nbinned_frames, int(np.floor(len(dat) / nframes_per_bin))])

    idx = np.arange(0, nbinned_frames * nframes_per_bin, nframes_per_bin, dtype='int')
    if not idx[-1] == nbinned_frames * nframes_per_bin:
        idx = np.hstack([idx, nbinned_frames * nframes_per_bin - 1])

    binned = np.zeros([len(idx) - 1, *dat.shape[1:]], dtype='float32')

    for i in range(len(idx) - 1):

        blk = dat[idx[i]:idx[i + 1]]  # work when data are loaded to memory
        avg = get_trial_baseline(idx[i], frames_average, onsets)

        if divide_by_average:
            binned[i] = np.mean((blk - (avg + np.float32(1e-5)))/ (avg + np.float32(1e-5)), axis=0)
        else:
            binned[i] = np.mean(blk - (avg + np.float32(1e-5)), axis=0)

    binned = binned.reshape((-1, np.multiply(*dims[-2:])))
    print("BInned T Shape", np.shape(binned))


    # Get U from the single value decomposition
    cov = np.dot(binned, binned.T) / binned.shape[1]
    cov = cov.astype('float32')

    u, s, v = svd(cov)
    U = normalize(np.dot(u[:, :k].T, binned), norm='l2', axis=1)
    k = U.shape[0]  # in case the k was smaller (low var)
    # if trials are defined, then use them to chunk data so that the baseline is correct
    if onsets is None:
        idx = np.arange(0, len(dat), nframes_per_chunk, dtype='int')
    else:
        idx = onsets
    if not idx[-1] == len(dat):
        idx = np.hstack([idx, len(dat) - 1])
    V = np.zeros((k, *dat.shape[:2]), dtype='float32')

    # Compute SVT
    for i in range(len(idx) - 1):

        blk = dat[idx[i]:idx[i + 1]]  # work when data are loaded to memory
        avg = get_trial_baseline(idx[i], frames_average, onsets).astype('float32')
        blk = blk - (avg + np.float32(1e-5))
        if divide_by_average:
            blk /= avg + np.float32(1e-5)

        V[:, idx[i]:idx[i + 1], :] = np.dot(U, blk.reshape([-1, np.multiply(*dims[1:])]).T).reshape((k, -1, dat.shape[1]))

    SVT = V.reshape((k, -1))
    U = U.T.reshape([*dims[-2:], -1])
    return U, SVT



def svd_blockwise(dat,frames_average,
                  k=200, block_k=20,
                  blocksize=120, overlap=8,
                  divide_by_average=True,
                  random_state=42):
    '''
    Computes the blockwise single value decomposition for a matrix that does not fit in memory.
    U,SVT,S,(block_U,block_SVT,blocks) = svd_blockwise(dat,
                                                   frames_average,
                                                   k = 200,
                                                   block_k = 20,
                                                   blocksize=120,
                                                   overlap=8)
    dat is a [nframes X nchannels X width X height] array
    frames_average is a [nchannels X width X height] array; the average to be subtracted before computing the SVD
    k is the number of components to be extracted (randomized SVD)
    The blockwise implementation works by first running the SVD on overlapping chunks of the movie. Secondly,  SVD is ran on the extracted temporal components and the spatial components are scaled to match the actual frame size.
The chunks have all samples in time but only a fraction of pixels.
    This is adapted from matlab code by Simon Musall.
    A similar approach is described in Stringer et al. Science 2019.
    Joao Couto - March 2020
    '''
    from sklearn.utils.extmath import randomized_svd
    from sklearn.preprocessing import normalize

    nframes,nchannels,w,h = dat.shape
    n = nframes*nchannels
    # Create the chunks where the SVD is ran initially,
    #these have all samples in time but only a few in space
    #chunks contain pixels that are nearby in space
    blocks = make_overlapping_blocks((w,h),blocksize=blocksize,overlap=overlap)
    nblocks = len(blocks)

    # M = U.S.VT
    # U are the spatial components in this case
    block_U = np.zeros((nblocks,blocksize,blocksize,block_k),dtype=np.float32)
    block_U[:] = np.nan
    # V are the temporal components
    block_SVT = np.zeros((nblocks,block_k,n),dtype=np.float32)
    block_U[:] = np.nan
    # randomized svd is ran on each chunk
    for iblock,(i,j) in tqdm(enumerate(blocks), total= len(blocks), desc= 'Computing SVD on data chunks:', leave=True):

        # subtract the average (this should be made the baseline instead)
        arr = np.array(dat[:,:,i[0]:i[1],j[0]:j[1]],dtype='float32')
        arr -= frames_average[:,i[0]:i[1],j[0]:j[1]]
        if divide_by_average:
            arr /= frames_average[:,i[0]:i[1],j[0]:j[1]]
        bw,bh = arr.shape[-2:]
        arr = arr.reshape([-1,np.multiply(*arr.shape[-2:])])

        arr = np.nan_to_num(arr)

        u, s, vt = randomized_svd(arr.T,
                                  n_components=block_k,
                                  n_iter=5,
                                  power_iteration_normalizer ='LQ',
                                  random_state=random_state)
        block_U[iblock,:bw,:bh,:] = u.reshape([bw,bh,-1])
        block_SVT[iblock] = np.dot(np.diag(s),vt)

    U,SVT,S = _complete_svd_from_blocks(block_U,block_SVT,blocks,k,(w,h))
    return U,SVT,S,(block_U,block_SVT,blocks)


def _complete_svd_from_blocks(block_U,block_SVT,blocks,k,dims,
                              n_iter=15,random_state=42):
    # Compute the svd of the temporal components from all blocks
    from sklearn.utils.extmath import randomized_svd
    u, s, vt = randomized_svd(
        block_SVT.reshape([np.multiply(*block_SVT.shape[:2]),-1]),
        n_components=k,
        n_iter=n_iter,
        power_iteration_normalizer ='QR',
        random_state=random_state)
    S = s
    SVT = np.dot(np.diag(S),vt)
    # Map the blockwise spatial components components to the second SVD
    U = np.dot(assemble_blockwise_spatial(block_U,blocks,dims),u)
    return U,SVT,S


def assemble_blockwise_spatial(block_U,blocks,dims):
    w,h = dims
    U = np.zeros([block_U.shape[0],block_U.shape[-1],w,h],dtype = 'float32')
    weights = np.zeros((w,h),dtype='float32')
    for iblock,(i,j) in enumerate(blocks):
        lw,lh = (i[1]-i[0],j[1]-j[0])
        U[iblock,:,i[0]:i[1],j[0]:j[1]] = block_U[iblock,:lw,:lh,:].transpose(-1,0,1)
        weights[i[0]:i[1],j[0]:j[1]] += 1
    U = (U/weights).reshape((np.multiply(*U.shape[:2]),-1))
    return U.T


def make_overlapping_blocks(dims, blocksize=128, overlap=16):
    '''
    Creates overlapping block indices to span an image
    '''

    w, h = dims
    blocks = []
    for i, a in enumerate(range(0, w, blocksize - overlap)):
        for j, b in enumerate(range(0, h, blocksize - overlap)):
            blocks.append([(a, np.clip(a + blocksize, 0, w)), (b, np.clip(b + blocksize, 0, h))])
    return blocks
