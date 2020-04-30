#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank

def tformfwd(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array, transform matrix
        @uv: Kx2 np.array, each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array, each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy):

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    if rank(X) >= 4:  # fixed 4
        r, res, rank_X, sv = lstsq(X, U, rcond=None)  # X * r = U
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    Tinv = np.array([[r[0], -r[1], 0],
                     [r[1],  r[0], 0],
                     [r[2],  r[3], 1]])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy):

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy)

    # Solve for trans2
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]  # reflect the xy data across the Y-axis
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    norm1 = norm(tformfwd(trans1, uv) - xy)
    norm2 = norm(tformfwd(trans2, uv) - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def cal_transmat(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u, v = src_pts[:, 0], src_pts[:, 1]
            x, y = dst_pts[:, 0], dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @src_pts: Kx2 np.array[each row is a pair of coordinates (x, y)]
        @dst_pts: Kx2 np.array[each row is a pair of transformed coordinates (x, y)]
        reflective: True or False
            if True: use reflective similarity transform
            else: use non-reflective similarity transform
    Returns:
    ----------
        @cv2_trans: 2x3 np.array | for cv2.warpAffine()
    """
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans[:, 0:2].T
