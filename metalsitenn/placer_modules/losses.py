# metalsitenn/placer_modules/losses.py
'''
This is a reproduction of chemnet's losses:
https://github.com/baker-laboratory/PLACER/blob/main/modules/losses.py

The following changes were made:
1. [TODO] Added losses for atom and bond order demasking 
'''
import sys, os
import numpy as np
import scipy
import scipy.optimize
import torch
import torch.distributions as Dist

from typing import Optional,List,Dict
try:
    from opt_einsum import contract as einsum
except:
    from torch import einsum

sys.path.append(os.path.dirname(__file__))
import geometry


# ============================================================
def bondLoss(X    : torch.Tensor,
             ij   : torch.Tensor,
             mean : bool = True,
             Y    : Optional[torch.Tensor] = None,
             b0   : Optional[torch.Tensor] = None,
             sel  : Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE loss on bonds

    Args:
        X:    atom coordinates in the model, [L,3]
        ij:   pairs of atoms forming bonds, [N,2]
        mean: boolean flag controlling the reduction to apply to the output (True=='mean', False=='sum')
        Y:    atom coordinates in the reference structure, [L,3]
        b0:   reference bond lengths (will be used if Y is not provided), [N]
        sel:  [L], boolean tensor to specify which atoms to use for bond loss calculation

    Returns:
        MSE loss on bond lengths derived from the model and the reference,
        summation is used instead of mean if mean=False
    """

    assert (Y==None)^(b0==None) # either Y or b0 must be provided

    if mean==True:
        # MSE = torch.nn.MSELoss(reduction='mean')
        MSE = torch.nn.L1Loss(reduction='mean')
    else:
        # MSE = torch.nn.MSELoss(reduction='sum')
        MSE = torch.nn.L1Loss(reduction='sum')

    # select everything if selection is not provided
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)
    
    # pick bonds within selection
    mask = sel[ij].all(-1)
    i,j = ij[mask].T
    
    # get bonds from X
    bx = torch.norm(X[i]-X[j], dim=-1)


    if Y is not None:
        # get bond distances from Y
        b0 = torch.norm(Y[i]-Y[j], dim=-1)
        loss = MSE(bx,b0)
    else:
        # use provided bonds as reference
        loss = MSE(bx,b0[mask])

    return loss


# ============================================================
def angleLoss(X    : torch.Tensor,
              ijk  : torch.Tensor,
              mean : bool = True,
              Y    : Optional[torch.Tensor] = None,
              a0   : Optional[torch.Tensor] = None,
              sel  : Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE loss on bonded angles

    Args:
        X:    atom coordinates in the model, [L,3]
        ijk:  triples of atoms forming bonded angles, [N,3]
        mean: boolean flag controlling the reduction to apply to the output (True=='mean', False=='sum')
        Y:    atom coordinates in the reference structure, [L,3]
        a0:   reference values of bonded angles (will be used if Y is not provided), [N]
        sel:  [L], boolean tensor to specify which atoms to use for angle loss calculation

    Returns:
        MSE loss on bonded angles derived from the model and the reference,
        summation is used instead of mean if mean=False
    """

    assert (Y==None)^(a0==None) # either Y or a0 must be provided

    if mean==True:
        #MSE = torch.nn.MSELoss(reduction='mean')
        MSE = torch.nn.L1Loss(reduction='mean')
    else:
        #MSE = torch.nn.MSELoss(reduction='sum')
        MSE = torch.nn.L1Loss(reduction='sum')

    # select everything if selection is not provided
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)
    
    # pick angles within selection
    mask = sel[ijk].all(-1)
    i,j,k = ijk[mask].T
    
    # get angles from X
    ax = geometry.get_ang(X[i],X[j],X[k])

    if Y is not None:
        # get angles from Y
        a0 = geometry.get_ang(Y[i],Y[j],Y[k])
        loss = MSE(ax,a0)
    else:
        # use provided angles as reference
        loss = MSE(ax,a0[mask])

    return loss


# ============================================================
def oopLoss(X    : torch.Tensor,
            ijkl : torch.Tensor,
            mean : bool = True,
            Y    : Optional[torch.Tensor] = None,
            a0   : Optional[torch.Tensor] = None,
            sel  : Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' '''

    assert (Y==None)^(a0==None) # either Y or a0 must be provided

    if mean==True:
        #MSE = torch.nn.MSELoss(reduction='mean')
        MSE = torch.nn.L1Loss(reduction='mean')
    else:
        #MSE = torch.nn.MSELoss(reduction='sum')
        MSE = torch.nn.L1Loss(reduction='sum')

    # select everything if selection is not provided
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    # pick angles within selection
    mask = sel[ijkl].all(-1)
    i,j,k,l = ijkl[mask].T

    # get angles from X
    nx = torch.cross(X[j]-X[i],X[k]-X[i],dim=-1)
    vx = X[l]-X[i]
    ox = torch.zeros_like(nx)
    ax = geometry.get_ang(nx,ox,vx)
    
    if Y is not None:
        # get angles from Y
        ny = torch.cross(Y[j]-Y[i],Y[k]-Y[i],dim=-1)
        vy = Y[l]-Y[i]
        oy = torch.zeros_like(ny)
        a0 = geometry.get_ang(ny,oy,vy)
        loss = MSE(ax,a0)
    else:
        # use provided angles as reference
        loss = MSE(ax,a0[mask])

    return loss

# ============================================================
def dMAE(X   : torch.Tensor,
         Y   : torch.Tensor,
         ij  : torch.Tensor,
         sel : Optional[torch.Tensor] = None) -> torch.Tensor:

    margin = 1.0

    if sel is None:
        i,j = ij.T
    else:
        mask = sel[ij].all(-1)
        i,j = ij[mask].T

    dX = (X[i]-X[j]).norm(dim=-1)
    dY = (Y[i]-Y[j]).norm(dim=-1)
    loss = torch.clip((dX-dY).abs() - margin, min=0.0)

    return loss.mean()


# ============================================================
def localDistLoss(X   : torch.Tensor,
                  Y   : torch.Tensor,
                  sep : torch.Tensor,
                  cut : int = 2,
                  sel : Optional[torch.Tensor] = None) -> torch.Tensor:
    '''distance loss for atoms which are close in the chemical graph'''
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    i,j = torch.where((sep>0) & (sep<=cut))
    mask = (i<j) & sel[i] & sel[j]
    i = i[mask]
    j = j[mask]

    if i.shape[0]>1:
        dX = (X[i]-X[j]).norm(dim=-1)
        dY = (Y[i]-Y[j]).norm(dim=-1)

        #MSE = torch.nn.MSELoss(reduction='mean')
        MSE = torch.nn.L1Loss(reduction='mean')
        loss = MSE(dX,dY)

    else:
        loss = torch.tensor(0.0,device=X.device)

    return loss


# ============================================================
def torsionLoss(X    : torch.Tensor,
                ijkl : torch.Tensor,
                mean : bool = True,
                Y    : Optional[torch.Tensor] = None,
                t0   : Optional[torch.Tensor] = None,
                sel  : Optional[torch.Tensor] = None) -> torch.Tensor:
    '''loss on torsion angles'''

    assert (Y==None)^(t0==None) # either Y or t0 must be provided

    if mean==True:
        MSE = torch.nn.MSELoss(reduction='mean')
    else:
        MSE = torch.nn.MSELoss(reduction='sum')

    # select everything if selection is not provided
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)
    
    # pick torsions within selection
    mask = sel[ijkl].all(-1)
    i,j,k,l = ijkl[mask].T

    # get torsions from X
    tx = geometry.get_dih(X[i],X[j],X[k],X[l])

    if Y is not None:
        # get torsions from Y
        t0 = geometry.get_dih(Y[i],Y[j],Y[k],Y[l])
        loss = MSE(tx.sin(),t0.sin()) + MSE(tx.cos(),t0.cos())
    else:
        # use provided torsions as reference
        loss = MSE(tx.sin(),t0[sel].sin()) + MSE(tx.cos(),t0[sel].cos())

    return loss


# ============================================================
def MSD(X: torch.Tensor,
        Y: torch.Tensor) -> torch.Tensor:
    '''mean squared distance between predicted (X) and 
    reference (Y) coordinates (w/o superimposition)'''

    msd = (X-Y).pow(2).sum(-1)
    return msd.mean()


# ============================================================
def Kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    '''Kabsch algorthm'''

    if P.shape[0]<=3:
        return torch.tensor(0.0,device=P.device)

    def rmsd(V, W):
        return torch.sqrt( torch.sum( (V-W)*(V-W) ) / len(V) )
    def centroid(X):
        return X.mean(axis=0)

    cP = centroid(P)
    cQ = centroid(Q)
    P = P - cP
    Q = Q - cQ

    # Computation of the covariance matrix
    C = torch.mm(P.T, Q)

    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3],device=P.device)
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))

    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    # Rotate P
    rP = torch.mm(P, U)

    # get RMS
    rms = rmsd(rP, Q)

    return rms #, rP


# ============================================================
def dRMSD(X   : torch.Tensor,
          Y   : torch.Tensor,
          sel : Optional[torch.Tensor] = None) -> torch.Tensor:
    '''distance RMSD loss'''

    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    dX = torch.cdist(X[sel],X[sel])
    dY = torch.cdist(Y[sel],Y[sel])

    MSE = torch.nn.MSELoss(reduction='mean')
    loss = MSE(dX,dY)

    return loss

# ============================================================
def pDE(logits : torch.Tensor,
        X      : torch.Tensor,
        Y      : torch.Tensor,
        sel    : Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' '''

    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    dX = torch.cdist(X[sel],X[sel])
    dY = torch.cdist(Y[sel],Y[sel])
    dd = dY-dX
    
    bins = torch.linspace(-5.0,5.0,101, device=X.device)
    target = torch.bucketize(dd,bins)

    CCE = torch.nn.CrossEntropyLoss()(logits[sel][:,sel].permute(2,0,1)[None],target[None].detach())

    return CCE


# ============================================================
def getGroundTruthProt(Xs : List[torch.Tensor],
                       xyz : torch.Tensor,
                       observed : torch.Tensor,
                       ra : torch.Tensor,
                       frames : torch.Tensor,
                       huber : float,
                       cut : float) -> torch.Tensor:
    '''get ground truth atom coordinates for a protein accounting for 
    alternative atom assigments in symmetric side chains

    Args:
        Xs:       [[N,3]], list of atom coordinates along the trajectory
        xyz:      [2,L,14,3], reference atomic coordinates w/o and w/ flips
                  of symmetric side chains
        observed: [N], bool mask to indicate which atoms are physically present in the reference
        ra:       [N,2], residue and atom indices to map observed atoms in xyz to X
                  X[observed]=xyz[r,a], where r,a=ra.T
        frames:   [N,3], triplets of atoms defining frames
        huber:    residual in the Huber loss
        cut:      distances beyond this value are trimmed

    Returns:
        reference atom coordinates Y.shape=[N,3]
    '''

    r,a = ra.T
    
    Yref = torch.zeros_like(Xs[0])
    Yalt = torch.zeros_like(Xs[0])

    Yref[observed] = xyz[0,r,a]
    Yalt[observed] = xyz[1,r,a]

    fape_ref = torch.zeros_like(xyz[0,:,:,0])
    fape_alt = torch.zeros_like(xyz[0,:,:,0])

    fape_ref[r,a] = torch.stack([FAPE(X,Yref,frames=frames,sel=observed, reduce=False) for X in Xs],dim=0).mean(0)
    fape_alt[r,a] = torch.stack([FAPE(X,Yalt,frames=frames,sel=observed, reduce=False) for X in Xs],dim=0).mean(0)

    xyz_ = torch.where((fape_ref.sum(-1)<=fape_alt.sum(-1))[:,None,None],xyz[0],xyz[1])
    Yref[observed] = xyz_[r,a]

    return Yref.detach()


# ============================================================
def getGroundTruthProt2(Xs : List[torch.Tensor],
                        Y : torch.Tensor,
                        sel : torch.Tensor,
                        flips : torch.Tensor,
                        frames : torch.Tensor) -> torch.Tensor:
    '''get ground truth atom coordinates for a protein accounting for 
    alternative atom assigments in symmetric side chains

    Args:
        Xs:       [[N,3]], list of atom coordinates along the trajectory
        Y:        [N,3], reference atomic coordinates
        sel:      [N], bool mask to indicate which atoms are physically present in the reference
        flips:    [N,2], pairs of flippable atoms
        frames:   [N,3], triplets of atoms defining frames

    Returns:
        reference atom coordinates Y.shape=[N,3]
    '''

    if flips.shape[0]<1:
        return Y.detach()
    
    a,b = flips.T

    # alternative placement of flippable atoms
    Y_ref = Y.clone()
    Y_alt = Y.clone()
    Y_alt[a] = Y_ref[b]
    Y_alt[b] = Y_ref[a]

    # tensors to save atom-wise FAPEs
    f1 = torch.zeros_like(Y[:,0])
    f2 = torch.zeros_like(Y[:,0])

    # collect FAPEs along the trajectory
    for X in Xs:
        f1[sel] += FAPE(X=X,Y=Y_ref,frames=frames,sel=sel, reduce=False)
        f2[sel] += FAPE(X=X,Y=Y_alt,frames=frames,sel=sel, reduce=False)

    # find pairs where flipped atoms give better FAPE    
    flag = f1[a]+f2[b]>f1[b]+f2[a]
    
    # flip coordinates
    Y_ref[a[flag]] = Y_alt[b[flag]]
    Y_ref[b[flag]] = Y_alt[a[flag]]
    
    return Y_ref.detach()


# ============================================================
def getGroundTruthLig(Xs : List[torch.Tensor],
                      Y : torch.Tensor,
                      idx : torch.Tensor,
                      frames : List[torch.Tensor],
                      huber : float,
                      cut : float) -> torch.Tensor:
    '''pick reference conformation with the lowest FAPE to a set of models

    Args:
        Xs:     [[L,3]], list of atom coordinates along the trajectory
        Y:      [B,L,3], atom coordinates of the reference conformations
        idx:    [L], indices showing which molecule each of atoms belong to
        frames: [N,3], ijk triples of atoms forming frames
        huber:  residual in the Huber loss
        cut:    distances beyond this value are trimmed

    Returns:
        reference atom coordinates Y.shape=[N,3]
    '''

    # shortcut for FAPE
    _FAPE_ = lambda x : FAPE(*x, huber=huber,cut=cut,reduce=True)
    
    # number of atoms in each molecule
    chunks = torch.unique(idx,return_counts=True)[1].tolist()
    
    # split reference coordinates into molecules
    Ys = torch.split(Y,chunks,dim=1)

    # get FAPE for all snapshots and all molecules
    fapes = torch.stack([torch.stack([_FAPE_(x) for x in zip(torch.split(X,chunks,dim=0), Ys, frames)]) for X in Xs])
    
    # average along trajectory
    fapes = fapes.mean(dim=0)
    
    # collect reference coordinates into one tensor
    ref = fapes.argmin(dim=-1).tolist()
    Yref = torch.cat([Yi[i] for Yi,i in zip(Ys,ref)], dim=0)
    
    return Yref.detach()


# ============================================================
def getGroundTruthLig2(Xs : List[torch.Tensor],
                       Y : torch.Tensor,
                       rmask : torch.Tensor,
                       frames : torch.Tensor,
                       huber : float,
                       cut : float) -> torch.Tensor:
    '''pick reference conformation with the lowest FAPE to a set of models

    Args:
        Xs:     [[N x L,3]], list of atom coordinates along the trajectory
        Y:      [N,B,L,3], atom coordinates of the reference conformations
        rmask:  [N], boolean flag indicating whether a residue is fixed
        frames: [N x M,3], ijk triples of atoms forming frames
        huber:  residual in the Huber loss
        cut:    distances beyond this value are trimmed

    Returns:
        reference atom coordinates Y.shape=[N,3]
    '''

    N,B,L,_ = Y.shape
    nfixed = rmask.sum()
    nmasked = N-nfixed
    
    # frames for one molecule
    frames0 = frames[:int(frames.shape[0]//N)]

    # frames for all fixed molecules
    frames_fixed = frames[:int(frames0.shape[0]*nfixed)]
    
     
    #
    # process fixed molecules
    #
    
    # centers of masses for fixed molecules
    #Ycom = Y[rmask].mean(dim=(1,2))
    Ycom = Y.mean(dim=(1,2))
    
    # process fixed molecules
    fapes = [] # [nXs, nfixed, npermuts]
    for X in Xs:
        #Xcom = X.reshape(N,L,3)[rmask].mean(dim=1) # [nfixed,3]
        Xcom = X.reshape(N,L,3).mean(dim=1) # [nfixed,3]
        for Xi,Yi in zip(X.reshape(N,L,3),Y):
            fapes.append(
                FAPE(X=Xi,Y=Yi, frames=frames0,huber=huber,cut=cut,reduce=True) + \
                FAPE_query(X=Xi,Y=Yi, frames=frames0, Xq=Xcom,Yq=Ycom,huber=huber,cut=cut,reduce=True)
            )
    #fapes = torch.stack(fapes).reshape(-1,nfixed,B).mean(dim=0) # [nfixed, npermuts]
    #Yfixed = torch.cat([Yi[fi.argmin()] for fi,Yi in zip(fapes,Y[rmask])], dim=0)
    
    fapes = torch.stack(fapes).reshape(-1,N,B).mean(dim=0)
    Yfixed = torch.cat([Yi[fi.argmin()] for fi,Yi in zip(fapes,Y)], dim=0)

    '''
    # process masked molecules
    fapes = [] # [nXs, nmasked, nmasked, npermuts]
    for X in Xs:
        Xfixed = X.reshape(N,L,3)[rmask].reshape(-1,3)
        for Xi,Yj in itertools.product(X.reshape(N,L,3)[~rmask],Y[~rmask]):
            fapes.append(
                FAPE(X=Xi,Y=Yj, frames=frames0,huber=huber,cut=cut,reduce=True) + \
                FAPE_query(X=Xi,Y=Yj, frames=frames0, Xq=Xfixed,Yq=Yfixed,huber=huber,cut=cut,reduce=True) + \
                FAPE_query(X=Xfixed,Y=Yfixed, frames=frames_fixed, Xq=Xi,Yq=Yj,huber=huber,cut=cut,reduce=True)
            )
    fapes = torch.stack(fapes).reshape(-1,nmasked,nmasked,B).mean(dim=0) # [nmasked, nmasked, npermuts]
    
    # 
    cost,indices = fapes.min(dim=-1)
    i,j = scipy.optimize.linear_sum_assignment(cost.cpu().detach())
    Ymasked = Y[~rmask][j,indices[i,j]].reshape(-1,3)

    
    Yref = torch.cat([Yfixed,Ymasked],dim=0)

    return Yref.detach()
    '''

    return Yfixed.detach()



# ============================================================
def FAPE(X : torch.Tensor,
         Y : torch.Tensor,
         frames : torch.Tensor,
         sel : Optional[torch.Tensor] = None,
         huber : float = 1.0,
         cut : float = 10.0,
         reduce : bool = True) -> torch.Tensor:
    '''all-tom FAPE loss

    Args:
        X:      [L,3], atom coordinates in the model
        Y:      [...,L,3], atom coordinates in the reference structure;
                batch dimension(s) can be used to account for multiple conformations
        frames: [N,3], ijk triples of atoms forming frames
        sel:    [N], boolean tensor to specify which atoms to use for FAPE calculation
        huber:  residual in the Huber loss
        cut:    distances beyond this value are trimmed
        reduce: whether to reduce the output along atom dimension L

    Returns:
        Huber loss on atom-atom distances projected onto local frames;
        the shape of the returned tensor is equal to the shapae of the
        leading dimension(s) of Y, e.g. [N] if Y.shape=[N,L,3]
        (or [N,L] if reduce==False)
    '''

    if frames.shape[0]<1:
        return torch.tensor(0.0,device=X.device)

    if sel is None:
        i,j,k = frames.T
        sel = torch.full_like(X[:,0], True, dtype=bool)
    else:
        mask = sel[frames].all(-1)
        i,j,k = frames[mask].T
    
    if i.shape[0]<1:
        if reduce==True:
            return torch.tensor(0.0,device=X.device)
        else:
            return torch.zeros_like(X[:,0])
    
    uX = geometry.get_frames(X[i], X[j], X[k])
    uY = geometry.get_frames(Y[...,i,:], Y[...,j,:], Y[...,k,:])

    tX = X[j]
    tY = Y[...,j,:]

    rX = einsum('fij,fai->afj',uX,X[None,sel]-tX[:,None])
    rY = einsum('...fij,...fai->...afj',uY,Y[...,None,sel,:]-tY[...,:,None,:])

    d2 = ((rX-rY)**2).sum(-1)+1e-4
    d = d2**0.5
    loss = torch.where(d<huber, 0.5*d2, huber*(d-0.5*huber))/huber
    loss = torch.clip(loss,min=0.0,max=cut)
    
    # reduce along frame dimension
    loss = loss.mean(dim=(-1))
    
    # reduce along atom dimension
    if reduce==True:
        loss = loss.mean(dim=(-1))

    return loss


# ============================================================
def FAPE_query(X : torch.Tensor,
               Y : torch.Tensor,
               frames : torch.Tensor,             
               Xq : torch.Tensor,
               Yq : torch.Tensor,
               mask : torch.Tensor = None,
               huber : float = 1.0,
               cut : float = 10.0,
               reduce : bool = True) -> torch.Tensor:
    '''all-tom FAPE loss

    Args:
        X:      [L,3], atom coordinates in the model
        Y:      [...,L,3], atom coordinates in the reference structure;
                batch dimension(s) can be used to account for multiple conformations
        frames: [N,3], ijk triples of atoms forming frames
        huber:  residual in the Huber loss
        cut:    distances beyond this value are trimmed
        reduce: whether to reduce the output along atom dimension L

    Returns:
        Huber loss on atom-atom distances projected onto local frames;
        the shape of the returned tensor is equal to the shapae of the
        leading dimension(s) of Y, e.g. [N] if Y.shape=[N,L,3]
        (or [N,L] if reduce==False)
    '''

    i,j,k = frames.T
    
    uX = geometry.get_frames(X[i], X[j], X[k])
    uY = geometry.get_frames(Y[...,i,:], Y[...,j,:], Y[...,k,:])

    tX = X[j]
    tY = Y[...,j,:]

    if mask is None:
        rX = einsum('fij,fai->afj',uX,Xq[None]-tX[:,None])
        rY = einsum('...fij,...fai->...afj',uY,Yq[...,None,:,:]-tY[...,:,None,:])
    else:
        rX = einsum('fa,fij,fai->afj',mask,uX,Xq[None]-tX[:,None])
        rY = einsum('fa,...fij,...fai->...afj',mask,uY,Yq[...,None,:,:]-tY[...,:,None,:])

    d2 = ((rX-rY)**2).sum(-1)+1e-4
    d = d2**0.5
    loss = torch.where(d<huber, 0.5*d2, huber*(d-0.5*huber))/huber
    loss = torch.clip(loss,min=0.0,max=cut)
    
    # reduce along frame dimension
    if mask is None:
        loss = loss.mean(dim=(-1))
    else:
        loss = loss*mask.T
        loss = loss.sum(dim=-1)/mask.sum(dim=0)
    
    # reduce along atom dimension
    if reduce==True:
        loss = loss.mean(dim=(-1))

    return loss


# ============================================================
def lDDT(X      : torch.Tensor,
         Y      : torch.Tensor,
         sel    : Optional[torch.Tensor] = None,
         s      : float = 1e-6) -> torch.Tensor:
    '''smooth lDDT loss
    
    Args:
        X:   [L,3], atom coordinates in the model
        Y:   [L,3], atom coordinates in the reference structure
        sel: [L], boolean tensor to specify which atoms to use for lDDT calculation
        s:   smoothing factor; s=0.35 - good for training (smooth)

    Returns:
        lDDT loss
    '''
    
    eps = 1e-4
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    dY = torch.cdist(Y[sel],Y[sel]).fill_diagonal_(999.9) # exclude diagonal
    i,j = torch.where(dY<15.0)
    dX = torch.cdist(X[sel],X[sel])
    dd = torch.abs(dX[i,j]-dY[i,j])+eps
    def f(x,m,s):
        return 0.5*torch.erf((torch.log(dd)-np.log(m))/(s*2**0.5))+0.5
    lddt = torch.stack([f(dd,m,s) for m in [0.5,1.0,2.0,4.0]],dim=-1)

    return 1.0-lddt.mean()


# ============================================================
def plDDT(logits : torch.Tensor,
          X      : torch.Tensor,
          Y      : torch.Tensor,
          sel    : Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' '''
    
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    # differences in pariwise atom distances in the model
    # and in the reference structure
    eps = 1e-4
    dX = torch.cdist(X[sel],X[sel])
    dY = torch.cdist(Y[sel],Y[sel])
    dd = torch.abs(dX-dY)+eps

    # number of atom neighbors within dmax in the reference structure (per-atom)
    dmax = 15.0
    ncont_ref = (dY<dmax).float().sum(dim=-1)

    # number of preserved contacts in the model under difference thresholds
    ncont_mod = torch.stack([((dd<cutoff) & (dY<dmax)).to(float).sum(dim=-1)
                             for cutoff in [0.5,1.0,2.0,4.0]],dim=0)
    lddt = ncont_mod.mean(0)/ncont_ref*100.0

    bins = torch.linspace(1,99,50, device=X.device)
    target = torch.bucketize(lddt,bins)

    CCE = torch.nn.CrossEntropyLoss()(logits[sel],target.detach())

    return CCE


# ============================================================
def pDEV(sigmas : torch.Tensor,
         X      : torch.Tensor,
         Y      : torch.Tensor,
         sel    : Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' '''
    
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    eps = 1e-3
    loc = torch.zeros_like(sigmas)
    scale = sigmas+eps
    scale = torch.nan_to_num(scale, nan=eps, posinf=20.0)
    N = Dist.Normal(loc,scale)
    
    d = (X-Y).norm(dim=-1)

    loss = -N.log_prob(d)[sel].mean()
    
    return loss


# ============================================================
def pFAPE(sigmas : torch.Tensor,
          X      : torch.Tensor,
          Y      : torch.Tensor,
          frames : torch.Tensor,
          sel    : Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' '''
    
    if sel is None:
        sel = torch.full_like(X[:,0], True, dtype=bool)

    # TODO: the use of sel may be buggy

    devs = FAPE(X,Y,frames,sel, huber=1e-6,cut=10.0,reduce=False)

    eps = 1e-3
    loc = torch.zeros_like(sigmas)
    scale = sigmas+eps
    N = Dist.Normal(loc,scale)
    loss = -N.log_prob(devs)[sel].mean()
    
    return loss


# ============================================================
def dLogNorm(lognorm : torch.Tensor,
             Y       : torch.Tensor,
             mean    : bool = True,
             sel     : Optional[torch.Tensor] = None,
             ij      : Optional[torch.Tensor] = None):

    L = Y.shape[0]
    
    if sel is None:
        sel = torch.full_like(Y[:,0], True, dtype=bool)

    if ij is None:
        ij = torch.combinations(torch.arange(L,device=Y.device), r=2)

    # pick pairs within selection
    mask = sel[ij].all(-1)
    i,j = ij[mask].T

    Dref = torch.norm(Y[i]-Y[j], dim=-1)
    Dref = torch.clamp(Dref, min=1e-3)
        
    logits,loc,scale = lognorm[:,i,j]
    mixture = Dist.MixtureSameFamily(
        mixture_distribution = Dist.Categorical(logits=logits),
        component_distribution = Dist.LogNormal(loc=loc,scale=scale)
    )
    
    nll = -mixture.log_prob(Dref)
    
    if mean==True:
        nll = nll.mean()
    else:
        nll = nll.sum()
    
    return nll



# ============================================================
class StructureLossesCSD:
    def __init__(self, 
                 terms : List[str],
                 huber : float = 1.0,
                 fapecut : float = 10.0):

        self.terms = [t.lower() for t in terms]
        self.huber = huber
        self.fapecut = fapecut


    def get_print_str(self, losses, last=None):
        
        if last is None:
            last = len(losses)
        values = np.array(losses)[-last:].mean(axis=0)
        out = "".join([" %s= %.5f "%(t,v) for t,v in zip(self.terms,values)])
        return out
        
        
    def get_losses(self,
                   Xs : List[torch.Tensor],
                   Ds : List[torch.Tensor],
                   Y : torch.Tensor,
                   topology : Dict[str,torch.Tensor],
                   plDDTs : List[torch.Tensor] = None,
                   pDEVs : List[torch.Tensor] = None) -> torch.Tensor:
        ''' '''

        device = Y.device
        
        # limit loss calculation to selected atoms only
        if 'observed' in topology.keys():
            sel = topology['observed']
        else:
            sel = torch.full_like(Y[0,:,0], True, dtype=bool)

        # get reference ground truth
        with torch.no_grad():
            Y = getGroundTruthLig(Xs=Xs,
                                  Y=topology['Y'],
                                  idx=topology['idx'],
                                  frames=topology['frames'],
                                  huber=self.huber,
                                  cut=self.fapecut)

        # split Y into individual molecules
        idx = topology['idx']
        chunks = torch.unique(idx,return_counts=True)[1].tolist()
        Y_ = torch.split(Y,chunks)
        sel_ = torch.split(sel,chunks)
        Xs_ = [torch.split(X,chunks) for X in Xs]
        
        losses = []
        for term in self.terms:

            if term=='fape':
                frames = topology['frames']
                _FAPE_ = lambda x : FAPE(*x, huber=self.huber,cut=self.fapecut)
                fapes = [torch.stack([_FAPE_(x) if len(x[-1])>0 else torch.tensor(0.0,device=device)
                                      for x in zip(torch.split(X,chunks),Y_,frames,sel_)]).mean()
                         for X in Xs]
                losses.append(torch.stack(fapes).mean())

            elif term=='kabsch':
                rmsds = [torch.stack([Kabsch(x[s].cpu(),y[s].cpu()) 
                                      for x,y,s in zip(X_,Y_,sel_)]).mean()
                         for X_ in Xs_]
                losses.append(torch.stack(rmsds).mean().to(device))

            elif term=='drmsd':
                drmsds = [dRMSD(x,y,s) for X_ in Xs_ for x,y,s in zip(X_,Y_,sel_)]
                losses.append(torch.stack(drmsds).mean())

            elif term=='bond':
                bonds = topology['bonds'][:,:2].long()
                losses.append(torch.stack([bondLoss(X,bonds, Y=Y, sel=sel) for X in Xs]).mean())

            elif term=='angle':
                angles = topology['angles'][:,:3].long()
                losses.append(torch.stack([angleLoss(X,angles, Y=Y, sel=sel) for X in Xs]).mean())

            elif term=='torsion':
                torsions = topology['dihedrals']
                if torsions.shape[0]>0:
                    losses.append(torch.stack([torsionLoss(X,torsions, Y=Y, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))

            elif term=='chiral':
                chirals = topology['chirals']
                if chirals.shape[0]>0:
                    losses.append(torch.stack([oopLoss(X, chirals[:,:4].long(), Y=Y, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))

            elif term=='planar':
                planars = topology['planars']
                if planars.shape[0]>0:
                    losses.append(torch.stack([oopLoss(X, planars, Y=Y, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))
            
            elif term=='pde':
                Ds_ = [[torch.split(Di,chunks,dim=1)[i] 
                        for i,Di in enumerate(torch.split(D,chunks))]
                        for D in Ds]
                pdes = [pDE(d,x.detach(),y,s) 
                        for X_,D_ in zip(Xs_,Ds_)
                        for d,x,y,s in zip(D_,X_,Y_,sel_)]
                losses.append(torch.stack(pdes).mean())

            elif term=='plddt':
                plDDTs_ = [torch.split(P,chunks) for P in plDDTs]
                plddts = [plDDT(logits=p,X=x.detach(),Y=y,sel=s) 
                          for X_,P_ in zip(Xs_,plDDTs_) 
                          for p,x,y,s in zip(P_,X_,Y_,sel_)]
                losses.append(torch.stack(plddts).mean())

            elif term=='pfape':
                pDEVs_ = [torch.split(P,chunks) for P in pDEVs]
                frames = topology['frames']
                devs = [pFAPE(sigmas=p,X=x.detach(),Y=y,frames=f,sel=s) 
                        for X_,P_ in zip(Xs_,pDEVs_)
                        for p,x,y,f,s in zip(P_,X_,Y_,frames,sel_)]
                losses.append(torch.stack(devs).mean())

            elif term=='dev':
                pDEVs_ = [torch.split(P,chunks) for P in pDEVs]
                devs = [pDEV(sigmas=p,X=x,Y=y,sel=s) 
                        for X_,P_ in zip(Xs_,pDEVs_)
                        for p,x,y,s in zip(P_,X_,Y_,sel_)]
                losses.append(torch.stack(devs).mean())

            elif term=='ldist':
                sep = topology['separation']
                ldist = torch.stack([localDistLoss(X,Y,sel=sel,sep=sep) for X in Xs]).mean()
                losses.append(ldist)

            elif term=='lddt':
                lddts = [lDDT(x,y,sel=s) 
                         for X_ in Xs_
                         for x,y,s in zip(X_,Y_,sel_)]
                losses.append(torch.stack(lddts).mean())

            else:
                sys.exit(f"Error: wrong structure loss type '{term}'")

        losses = torch.stack(losses)
        
        return losses



# ============================================================
class StructureLossesPDB:
    def __init__(self, 
                 terms : List[str],
                 huber : float = 1.0,
                 fapecut : float = 10.0):

        self.terms = [t.lower() for t in terms]
        self.huber = huber
        self.fapecut = fapecut


    def get_print_str(self, losses, last=None):
        
        if last is None:
            last = len(losses)
        values = np.array(losses)[-last:].mean(axis=0)
        out = "".join([" %s= %.5f "%(t,v) for t,v in zip(self.terms,values)])
        return out

    def get_reference(self,
                      Xs : List[torch.Tensor],
                      Y : torch.Tensor,
                      topology : Dict[str,torch.Tensor]):
        
        '''resove alternative atom assigments'''
        Yref = torch.clone(Y).detach()
        frames = topology['frames']
        with torch.no_grad():
            for p in topology['permuts']:
                fapes = [FAPE_query(X,Y,frames,X[p[0]],Y[pi]) for X in Xs[-1:] for pi in p]
                fapes = torch.stack(fapes).reshape(-1,p.shape[0])
                idx = fapes.sum(0).argmin()
                Yref[p[0]] = Yref[p[idx]]

        return Yref

    def get_losses(self,
                   Xs : List[torch.Tensor],
                   Ds : List[torch.Tensor],
                   Y : torch.Tensor,
                   topology : Dict[str,torch.Tensor],
                   plDDTs : List[torch.Tensor] = None,
                   pDEVs : List[torch.Tensor] = None) -> torch.Tensor:
        ''' '''

        device = Y.device
        
        # limit loss calculation to selected atoms only
        if 'observed' in topology.keys():
            sel = topology['observed']
        else:
            sel = torch.full_like(Y[:,0], True, dtype=bool)

        # resove alternative atom assigments
        Yref = torch.clone(Y).detach()
        frames = topology['frames']
        with torch.no_grad():
            for p in topology['permuts']:
                fapes = [FAPE_query(X,Y,frames,X[p[0]],Y[pi]) for X in Xs[-1:] for pi in p]
                fapes = torch.stack(fapes).reshape(-1,p.shape[0])
                idx = fapes.sum(0).argmin()
                Yref[p[0]] = Yref[p[idx]]
        
        # protein and ligand frames
        frames = topology['angles']
        idx = topology['corrupted']
        mask = torch.zeros_like(sel)
        if idx.shape[0]==0:  # in case no ligand is corrupted
            idx = torch.arange(Y.shape[0],device=device).long()

        mask[idx] = True
        maskL = mask&sel
        maskP = (~mask)&sel
        framesL = frames[maskL[frames].all(-1)]
        framesP = frames[maskP[frames].all(-1)]
        
        losses = []
        for term in self.terms:
            
            if term=='fape':
                frames = topology['angles']
                fapes = [FAPE(X,Yref,frames,sel=sel,huber=self.huber,cut=self.fapecut) for X in Xs]
                losses.append(torch.stack(fapes).mean())

            elif term=='fape_l':
                '''
                frames = topology['angles']
                idx = topology['corrupted']
                mask = torch.zeros_like(sel)
                mask[idx] = True
                mask = mask&sel
                '''
                fapes = [FAPE(X,Yref,framesL,sel=maskL,huber=self.huber,cut=self.fapecut) for X in Xs]
                losses.append(torch.stack(fapes).mean())

            elif term=='fape_lp_v1':
                idx = topology['corrupted']
                #frames = topology['frames']
                frames = topology['angles']
                frames = frames[sel[frames].all(-1)]
                fapes = [FAPE_query(X,Yref,frames,X[idx],Yref[idx],huber=self.huber,cut=self.fapecut) for X in Xs]
                losses.append(torch.stack(fapes).mean())

            elif term=='fape_lp_v2':
                idx = topology['corrupted']
                frames = topology['angles']
                fapes = [FAPE_query(X,Yref,framesP,X[maskL],Yref[maskL],huber=self.huber,cut=self.fapecut) for X in Xs]
                if framesL.shape[0]>0:
                    fapes += [FAPE_query(X,Yref,framesL,X[maskP],Yref[maskP],huber=self.huber,cut=self.fapecut) for X in Xs]
                losses.append(torch.stack(fapes).mean())

            elif term=='kabsch':
                idx = topology['corrupted']
                if idx.shape[0]>=3:
                    losses.append(torch.stack([Kabsch(X[idx].cpu(),Yref[idx].cpu()).to(device) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))

            elif term=='rmsd':
                idx = topology['corrupted']
                if idx.shape[0]==0:  # in case no ligand is corrupted:
                    losses.append(torch.tensor(0.0,device=device))
                else:
                    losses.append(torch.stack([((X[idx]-Yref[idx])**2).sum(-1).mean()**0.5 for X in Xs]).mean())

            elif term=='lddt':
                losses.append(torch.stack([lDDT(X,Yref,sel=sel,s=1e-3) for X in Xs]).mean())

            elif term=='dev':
                if len(Xs)==len(pDEVs):
                    losses.append(torch.stack([pDEV(sigmas=s,X=X,Y=Yref,sel=sel) 
                                               for X,s in zip(Xs,pDEVs)]).mean())
                else:
                    losses.append(torch.stack([pDEV(sigmas=s,X=X,Y=Yref,sel=sel) 
                                               for X,s in zip(Xs[1:],pDEVs)]).mean())

            elif term=='pde':
                losses.append(torch.stack([pDE(D,X,Yref,sel) for X,D in zip(Xs,Ds)]).mean())

            elif term=='pde_lp':
                
                def pDE_lp(logits,X):
                    dX = torch.cdist(X,X)
                    dY = torch.cdist(Yref,Yref)
                    dd = dY-dX
                    bins = torch.linspace(-5.0,5.0,101, device=X.device)
                    target = torch.bucketize(dd,bins)[sel][:,maskL]
                    CCE = torch.nn.CrossEntropyLoss()(logits[sel][:,maskL].permute(2,0,1)[None],target[None].detach())
                    return CCE
                
                losses.append(torch.stack([pDE_lp(D,X) for X,D in zip(Xs,Ds)]).mean())

            elif term=='plddt':
                if len(Xs)==len(plDDTs):
                    losses.append(torch.stack([plDDT(logits=p,X=X,Y=Yref,sel=sel) 
                                               for X,p in zip(Xs,plDDTs)]).mean())
                else:
                    losses.append(torch.stack([plDDT(logits=p,X=X,Y=Yref,sel=sel) 
                                               for X,p in zip(Xs[1:],plDDTs)]).mean())

            elif term=='bond':
                bonds = topology['bonds'][:,:2].long()
                losses.append(torch.stack([bondLoss(X,bonds, Y=Yref, sel=sel) for X in Xs]).mean())

            elif term=='angle':
                angles = topology['angles'][:,:3].long()
                losses.append(torch.stack([angleLoss(X,angles, Y=Yref, sel=sel) for X in Xs]).mean())

            elif term=='torsion':
                torsions = topology['torsions']
                if torsions.shape[0]>0:
                    losses.append(torch.stack([torsionLoss(X,torsions, Y=Yref, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))
            
            elif term=='chiral':
                chirals = topology['chirals']
                if chirals.shape[0]>0:
                    losses.append(torch.stack([oopLoss(X, chirals, Y=Yref, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))

            elif term=='planar':
                planars = topology['planars']
                if planars.shape[0]>0:
                    losses.append(torch.stack([oopLoss(X, planars, Y=Yref, sel=sel) for X in Xs]).mean())
                else:
                    losses.append(torch.tensor(0.0,device=device))

            elif term=='ldist':
                mask = torch.zeros_like(sel)
                mask[idx] = True
                mask = mask&sel
                sep = topology['separation']
                ldist = torch.stack([localDistLoss(X,Y,sel=mask,sep=sep) for X in Xs]).mean()
                losses.append(ldist)

            elif term=='ldev':
                mask = torch.zeros_like(sel)
                mask[idx] = True
                mask = mask&sel
                if mask.sum()>0:
                    if len(Xs)==len(pDEVs):
                        devs = torch.stack([pDEV(sigmas=s,X=X,Y=Yref,sel=mask)
                                            for X,s in zip(Xs,pDEVs)]).mean()
                    else:
                        devs = torch.stack([pDEV(sigmas=s,X=X,Y=Yref,sel=mask)
                                            for X,s in zip(Xs[1:],pDEVs)]).mean()
                else:
                    devs = torch.tensor(0.0,device=device)
                losses.append(devs)
            else:
                sys.exit(f"Error: wrong structure loss type '{term}'")

        losses = torch.stack(losses)

        return losses