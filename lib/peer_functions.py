from scipy.io import loadmat, savemat
from scipy.linalg import orth
import numpy as np
# import faiss
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import gc
from randn2 import randn2 


hrr_flag = False


def findAttractorsAndSplitIdx(p, attractorP, nDims):

    assert p>=0 ;
    assert p<=1;
    assert not np.any(attractorP[:]<0) ;
    assert not np.any(attractorP[:]>1);

#     % find x attractor below (the largest pos that is smaller than p)
    idx1 = max(np.argwhere(attractorP<=p))[0];
        
#     % find x attractor above (the smallest pos that is larger than p)
    idx2 = min(np.argwhere(attractorP>=p))[0];
        
#     % find weighting
    d1 = abs(attractorP[idx1]-p);
    d2 = abs(attractorP[idx2]-p);
    w = d2/(d1+d2); 
    
    if np.isnan(w):
        w = 0;
    
#     % compute index from weighting 
    splitIdx = round(w*nDims);
    return idx1, idx2, splitIdx
    
def encodePosesHDCconcatMultiAttractor(P, X, Y, params):

#     %% prepare basis vectors
    nBaseX = params["nX"]+1
    posX = np.append(np.arange(0, 1,1/(nBaseX-1)),1)
        
    nBaseY = params["nY"]+1
    posY = np.append(np.arange(0, 1, 1/(nBaseY-1)),1)
    
#     %% check relative poses
    xr = P[:,1]
    yr = P[:,0]       
    assert np.all(xr>=0) and np.all(xr<=1)
    assert np.all(yr>=0) and np.all(yr<=1)
        
#     %% encode
    encodedP = np.zeros((P.shape[0], params["nDims"]))
    for i in range(P.shape[0]):
    
#         % find attractors and split index
        Yidx1, Yidx2, YsplitIdx =  findAttractorsAndSplitIdx(yr[i], posY, params["nDims"])
        Xidx1, Xidx2, XsplitIdx =  findAttractorsAndSplitIdx(xr[i], posX, params["nDims"])
        
#         % apply
        xVec = np.concatenate([X[Xidx1, :XsplitIdx],X[Xidx2, XsplitIdx:]])
        yVec = np.concatenate([Y[Yidx1, :YsplitIdx],Y[Yidx2, YsplitIdx:]])
        
#         % combine
        if hrr_flag:
            encodedP[i, :] = circular_convolution(xVec, yVec)
        else:
            encodedP[i,:] = np.multiply(xVec, yVec)
    return encodedP
    
def bundleLocalDescriptorsWithPoses(D, A, X, Y, P, params):

    projD = np.matmul(D,A);

#     % standardize descriptors per image        
    mu = np.mean(projD,axis=0)
    sigma = np.std(projD,axis=0)
    stdProjD = np.divide(np.subtract(projD,mu),sigma)
    stdProjD[np.isnan(stdProjD)] = 0
    
#     %% encode poses
    encodedP = encodePosesHDCconcatMultiAttractor(P, X, Y, params)

#     %% bind each descriptor to its pose and bundle
    if hrr_flag:
        hdcD = np.sum(circular_convolution(stdProjD, encodedP), axis=0)
    else:
        hdcD = np.sum(np.multiply(stdProjD,encodedP), axis=0)
    return hdcD
    
def encodeImagesHDC(Y,params):
    
    nBaseX = params["nX"]+1
    
    nBaseY = params["nY"]+1

    seed_from_matlab = np.sum(np.frombuffer(b'projection', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    A = randn2(params["sDims"], params["nDims"])  ## (1024,4096)

    seed_from_matlab = np.sum(np.frombuffer(b'poseX', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    X_rnd = 1-2*(randn2(nBaseX, params["nDims"])>0);
    
    seed_from_matlab = np.sum(np.frombuffer(b'poseY', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    Y_rnd = 1-2*(randn2(nBaseY, params["nDims"])>0);


    nIm = Y.shape[0]   # 200 (number of images)
    hdcD = np.zeros((nIm, params["nDims"]))  # (200,4096)
    for i in range(nIm):  # for each image
        
#         % we use params.nFeat local feature descriptors D
        nFeat = min(params["nFeat"], Y[i]["descriptors"][0].shape[0])
        D = Y[i]["descriptors"][0]
        
#         % we need poses P to be in range [0,1]
        try:
            P = Y[i]["keypoints"][0]  / [Y[i]["imsize"][0][0][1], Y[i]["imsize"][0][0][0]]
        except: 
            P = Y[i]["keypoints"][0]  / [Y[i]["imheight"], Y[i]["imwidth"]]
        
        P[np.isnan(P)]=0;
        assert np.all(P>=0)

        # print(P.shape)
        # assert(False)

        # print(Y[i]["imsize"][0][0][1], Y[i]["imsize"][0][0][0])
        
        hdcD[i,:] = bundleLocalDescriptorsWithPoses(D, A, X_rnd, Y_rnd, P, params)
    return hdcD

def encodeImagesHDCOther(Y,params):
    
    nBaseX = params["nX"]+1
    
    nBaseY = params["nY"]+1

    seed_from_matlab = np.sum(np.frombuffer(b'projection', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    A = randn2(params["sDims"], params["nDims"])  ## (1024,4096)

    seed_from_matlab = np.sum(np.frombuffer(b'poseX', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    X_rnd = 1-2*(randn2(nBaseX, params["nDims"])>0);
    
    seed_from_matlab = np.sum(np.frombuffer(b'poseY', dtype=np.uint8))
    np.random.seed(seed_from_matlab)
    
    Y_rnd = 1-2*(randn2(nBaseY, params["nDims"])>0);


    nIm = Y.shape[0]   # 200 (number of images)
    hdcD = np.zeros((nIm, params["nDims"]))  # (200,4096)
    for i in range(nIm):  # for each image
        
#         % we use params.nFeat local feature descriptors D
        nFeat = min(params["nFeat"], Y[i]["descriptors"][0].shape[0])
        D = Y[i]["descriptors"][0]
        
#         % we need poses P to be in range [0,1]
        try:
            P = Y[i]["keypoints"][0]  / [Y[i]["imsize"][0][0][1], Y[i]["imsize"][0][0][0]]
        except: 
            P = Y[i]["keypoints"][0]  / [Y[i]["imheight"], Y[i]["imwidth"]]
        
        P[np.isnan(P)]=0;
        assert np.all(P>=0)

        # print(P.shape)
        # assert(False)

        # print(Y[i]["imsize"][0][0][1], Y[i]["imsize"][0][0][0])
        
        hdcD[i,:] = bundleLocalDescriptorsWithPoses(D, A, X_rnd, Y_rnd, P, params)
    return hdcD

def directCandSel(DD, n=100):

    nDB, nQ = DD.shape
    Idx = np.zeros((n, nQ))

    for j in range(nQ):
        idx = np.argsort(DD[:, j]);
        Idx[:, j] = idx[:n];
    return Idx



def getRecallAtKVector(Idx, GT):

    k, nQ = Idx.shape

    r = np.zeros(k);
    nMustFind = 0;
    for j in range(nQ):

    #only evaluate if there is a GThard matching for this query
        if np.any(GT["GThard"][0][0][: ,j]):

            nMustFind = nMustFind + 1;

            for i in range(k):
                # if np.any(GT["GTsoft"][0][0][Idx[:i+1, j], j] == 1):
                if np.any(np.take(GT["GTsoft"][0][0][:, j], Idx[:i + 1, j].astype(int))==1):
                    r[i] = r[i] + 1;

    r = r / nMustFind;
    return r

def circular_convolution(x, y):
    """A fast version of the circular convolution."""
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    if np.ndim(z) == 1:
        z = z[None, :]
    return z


def createPRNew(S_in, GThard, GTsoft=None, matching='multi', n_thresh=100):
    """
    Calculates the precision and recall at n_thresh equally spaced threshold values
    for a given similarity matrix S_in and ground truth matrices GThard and GTsoft for
    single-best-match VPR or multi-match VPR.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The string matching should be set to either "single" or "multi" for single-best-
    match VPR or multi-match VPR.
    The integer n_tresh controls the number of threshold values and should be >1.
    """

    # print(f'S_in.shape= {S_in.shape}\nGThard.shape= {GThard.shape}\nGTsoft.shape= {GTsoft.shape}')

    # Convert sparse matrices to dense arrays if needed
    if hasattr(GThard, 'toarray'):
        GThard = GThard.toarray()
    if hasattr(GTsoft, 'toarray'):
        GTsoft = GTsoft.toarray()

    assert (S_in.shape == GThard.shape),"S_in and GThard must have the same shape"
    if GTsoft is not None:
        assert (S_in.shape == GTsoft.shape),"S_in and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (matching in ['single', 'multi']),"matching should contain one of the following strings: [single, multi]"
    assert (n_thresh > 1),"n_thresh must be >1"

    if GTsoft is not None and matching == 'single':
        raise ValueError(
            "GTSoft with single matching is not supported. "
            "Please use dilated hard ground truth directly. "
            "For more details, visit: https://github.com/stschubert/VPR_Tutorial"
        )

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype('bool')
    if GTsoft is not None:
        GTsoft = GTsoft.astype('bool')

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()

    # single-best-match or multi-match VPR
    if matching == 'single':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT.any(0))

        # GT-values for best match per query (i.e., per column)
        GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]

        # similarities for best match per query (i.e., per column)
        S = np.max(S, axis=0)

    elif matching == 'multi':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT) # ground truth positives

    # init precision and recall vectors
    R = [0, ]
    P = [1, ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold

    # iterate over different thresholds
    for i in np.linspace(startV, endV, n_thresh):
        B = S >= i  # apply threshold

        TP = np.count_nonzero(GT & B)  # true positives
        FP = np.count_nonzero((~GT) & B)  # false positives

        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall

    return R, P

def createPR(S, GThard, GTsoft):

    # Convert sparse matrices to dense arrays if needed
    if hasattr(GThard, 'toarray'):
        GThard = GThard.toarray()
    if hasattr(GTsoft, 'toarray'):
        GTsoft = GTsoft.toarray()
    
    # print(GThard.shape, GThard.shape)
    #% remove soft-but-not-hard-entries
    S[np.where(GTsoft &  ~GThard)] = np.min(S[:]);

    
    GT = GThard #; % ensure logical-datatype
    
#     % init precision and recall vectors
    R = [0];
    P = [1];
    
#     % select start and end treshold
    startV = np.max(S[:]); #% start-value for treshold
    endV = np.min(S[:]); #% end-value for treshold
    
#     % iterate over different thresholds
    for i in np.linspace(startV, endV, 100):
        B = S>=i; #% apply threshold
        
        TP = np.count_nonzero( GT & B ); #% true positives
        FN = np.count_nonzero( GT & (~ B) ); #% false negatives
        FP = np.count_nonzero( (~ GT) & B ); #% false positives
        
        P.append(TP/(TP + FP)); #% precision
        R.append(TP/(TP + FN)); #% recall
    return R,P