"""Differentiable XYZ-only corrections of PiPER joint action chunks.

The actor supplies metres in the robot base frame. IK targets retain the base
chunk's TCP orientation and gripper command. The JSON specification contains the
URDF kinematic chain, so a saved policy does not depend on a later asset edit.
Accepted solutions use a fixed, unrolled differentiable damped-Newton solve;
there is no straight-through replacement or detached IK approximation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .kinematics import PiperKinematics

TYPE = 'piper_joint7_cartesian_xyz'
VERSION = 1
DEFAULTS = dict(iterations=6, damping=1e-3, rotation_weight=.15,
                max_joint_step_rad=.15, max_joint_delta_rad=.35,
                position_tolerance_m=5e-5, rotation_tolerance_rad=1e-3,
                backoff_scales=[1., .5, .25])


def build_piper_xyz_spec(urdf_path, *, tip_offset_m=.1358, **solver):
    """Create a portable immutable-in-use mapping from the actual robot asset."""
    path = Path(urdf_path)
    kin = PiperKinematics(path, tip_offset=tip_offset_m)
    spec = dict(type=TYPE, version=VERSION,
                urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                joint_names=list(kin.joint_names), lower=kin.lower.tolist(), upper=kin.upper.tolist(),
                tip_offset_m=float(tip_offset_m),
                chain=[dict(origin=origin.tolist(), axis=None if axis is None else axis.tolist())
                       for origin, axis in kin._chain],
                solver={**deepcopy(DEFAULTS), **solver})
    return validate_piper_xyz_spec(spec)


def validate_piper_xyz_spec(spec):
    """Validate/canonicalize a JSON-only kinematic specification (no CUDA)."""
    if not isinstance(spec, dict) or set(spec) != {
        'type','version','urdf_sha256','joint_names','lower','upper','tip_offset_m','chain','solver'}:
        raise ValueError('Cartesian residual specification has missing or unsupported fields')
    value = deepcopy(spec)
    if value['type'] != TYPE or type(value['version']) is not int or value['version'] != VERSION:
        raise ValueError('Unsupported Cartesian residual specification version/type')
    digest = value['urdf_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Cartesian residual requires the URDF SHA256')
    if value['joint_names'] != [f'joint{i}' for i in range(1,7)]:
        raise ValueError('Cartesian residual requires ordered PiPER joint1..joint6')
    lower, upper = np.asarray(value['lower'],float), np.asarray(value['upper'],float)
    if lower.shape!=(6,) or upper.shape!=(6,) or not np.isfinite([lower,upper]).all() or np.any(lower>=upper):
        raise ValueError('Cartesian residual requires six ordered finite joint limits')
    value['lower'],value['upper']=lower.tolist(),upper.tolist()
    offset=value['tip_offset_m']
    if isinstance(offset,bool) or not isinstance(offset,(int,float)) or not math.isfinite(offset) or offset<0:
        raise ValueError('Cartesian residual TCP offset must be finite and nonnegative')
    value['tip_offset_m']=float(offset)
    chain=value['chain']
    if not isinstance(chain,list) or not chain:
        raise ValueError('Cartesian residual requires the URDF chain')
    revolute=0
    for row in chain:
        if not isinstance(row,dict) or set(row)!={'origin','axis'}:
            raise ValueError('Each Cartesian chain entry requires origin and axis')
        origin=np.asarray(row['origin'],float)
        if (origin.shape!=(4,4) or not np.isfinite(origin).all()
            or not np.allclose(origin[3],[0,0,0,1],rtol=0,atol=1e-12)
            or not np.allclose(origin[:3,:3].T@origin[:3,:3],np.eye(3),rtol=0,atol=1e-8)
            or not np.isclose(np.linalg.det(origin[:3,:3]),1.,rtol=0,atol=1e-8)):
            raise ValueError('Cartesian chain origin must be a rigid transform')
        row['origin']=origin.tolist()
        if row['axis'] is not None:
            axis=np.asarray(row['axis'],float)
            if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,rtol=0,atol=1e-8):
                raise ValueError('Cartesian chain revolute axis must be a unit vector')
            row['axis']=axis.tolist();revolute+=1
    if revolute!=6:
        raise ValueError('Cartesian residual requires six revolute joints')
    settings=value['solver']
    if not isinstance(settings,dict) or set(settings)!=set(DEFAULTS):
        raise ValueError('Cartesian solver specification has missing or unsupported fields')
    if type(settings['iterations']) is not int or not 1<=settings['iterations']<=32:
        raise ValueError('Cartesian solver iterations must be an integer in1..32')
    for key in set(DEFAULTS)-{'iterations','backoff_scales'}:
        x=settings[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:
            raise ValueError(f'Cartesian solver {key} must be finite and positive')
        settings[key]=float(x)
    scales=settings['backoff_scales']
    if (not isinstance(scales,(list,tuple)) or not scales or float(scales[0])!=1.
        or any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<s<=1 for s in scales)
        or any(b>=a for a,b in zip(scales,scales[1:]))):
        raise ValueError('Cartesian backoff scales must strictly descend from1 in(0,1]')
    settings['backoff_scales']=[float(s) for s in scales]
    return value


class PiperCartesianXYZComposer(nn.Module):
    """Stateless FK/IK map: ``base[...,7], delta_metres[...,3] -> joint7``.

    Input arm angles must already be projected into the URDF bounds. Gripper
    values are copied exactly. A local branch bound prevents global IK jumps.
    Failed full translations are retried at fixed smaller scales, then fall
    back to the base command. Gradients follow the accepted numerical solve;
    discrete acceptance/scale selection is locally constant.
    """
    def __init__(self,spec):
        super().__init__()
        self.spec=validate_piper_xyz_spec(spec)
        self.settings=self.spec['solver']
        self.register_buffer('lower',torch.tensor(self.spec['lower'],dtype=torch.float64))
        self.register_buffer('upper',torch.tensor(self.spec['upper'],dtype=torch.float64))
        self.register_buffer('origins',torch.tensor([r['origin'] for r in self.spec['chain']],dtype=torch.float64))
        axes=[];joint_indices=[];joint=0
        for row in self.spec['chain']:
            axis=[0.,0.,0.] if row['axis'] is None else row['axis']
            axes.append(axis);joint_indices.append(-1 if row['axis'] is None else joint)
            joint+=row['axis'] is not None
        axis=torch.tensor(axes,dtype=torch.float64)
        cross=torch.zeros((len(axes),3,3),dtype=torch.float64)
        cross[:,0,1]=-axis[:,2];cross[:,0,2]=axis[:,1]
        cross[:,1,0]=axis[:,2];cross[:,1,2]=-axis[:,0]
        cross[:,2,0]=-axis[:,1];cross[:,2,1]=axis[:,0]
        self.register_buffer('axes',axis);self.register_buffer('cross',cross)
        self.register_buffer('cross_squared',cross@cross)
        self.joint_indices=tuple(joint_indices)

    def _buffers_for(self,q):
        return tuple(x.to(device=q.device,dtype=q.dtype) for x in
                     (self.origins,self.axes,self.cross,self.cross_squared,self.lower,self.upper))

    def fk(self,q,*,jacobian=False):
        """Return TCP transforms (and geometric base-frame6x6 Jacobians)."""
        if q.ndim<1 or q.shape[-1]!=6 or not q.is_floating_point():
            raise ValueError('FK requires floating joint angles [...,6]')
        shape=q.shape[:-1];q=q.reshape(-1,6)
        origins,axes,cross,cross_squared,_,_=self._buffers_for(q)
        eye=torch.eye(3,device=q.device,dtype=q.dtype)
        rotation=eye.expand(len(q),3,3)
        position=q.new_zeros((len(q),3))
        points=[];world_axes=[]
        for index,joint in enumerate(self.joint_indices):
            origin=origins[index]
            position=position+(rotation@origin[:3,3,None]).squeeze(-1)
            rotation=rotation@origin[:3,:3]
            if joint>=0:
                if jacobian:
                    points.append(position);world_axes.append((rotation@axes[index,:,None]).squeeze(-1))
                angle=q[:,joint,None,None]
                turn=eye+torch.sin(angle)*cross[index]+(1-torch.cos(angle))*cross_squared[index]
                rotation=rotation@turn
        position=position+rotation[:,:,2]*self.spec['tip_offset_m']
        transform=torch.cat((torch.cat((rotation,position[:,:,None]),dim=2),
                              q.new_tensor([0.,0.,0.,1.]).expand(len(q),1,4)),dim=1)
        if not jacobian:return transform.reshape(*shape,4,4)
        ax=torch.stack(world_axes,dim=2);displacement=position[:,:,None]-torch.stack(points,dim=2)
        linear=torch.linalg.cross(ax,displacement,dim=1)
        jac=torch.cat((linear,ax),dim=1)
        return transform.reshape(*shape,4,4),jac.reshape(*shape,6,6)

    @staticmethod
    def _orientation_error(target,current):
        relative=target@current.transpose(-1,-2)
        return .5*torch.stack((relative[...,2,1]-relative[...,1,2],
                              relative[...,0,2]-relative[...,2,0],
                              relative[...,1,0]-relative[...,0,1]),dim=-1)

    def _solve(self,base_q,target_position,target_rotation):
        cfg=self.settings
        q=base_q
        lower=torch.maximum(self.lower.to(q),base_q-cfg['max_joint_delta_rad'])
        upper=torch.minimum(self.upper.to(q),base_q+cfg['max_joint_delta_rad'])
        eye=torch.eye(6,device=q.device,dtype=q.dtype)
        weights=q.new_tensor([1.,1.,1.,cfg['rotation_weight'],cfg['rotation_weight'],cfg['rotation_weight']])
        for _ in range(cfg['iterations']):
            transform,jac=self.fk(q,jacobian=True)
            error=torch.cat((target_position-transform[:,:3,3],
                             self._orientation_error(target_rotation,transform[:,:3,:3])),dim=-1)*weights
            jac=jac*weights[None,:,None]
            jt=jac.transpose(-1,-2)
            update=torch.linalg.solve(jt@jac+cfg['damping']**2*eye,(jt@error[:,:,None])).squeeze(-1)
            update=update/torch.clamp(update.abs().amax(dim=-1,keepdim=True)/cfg['max_joint_step_rad'],min=1.)
            q=torch.minimum(torch.maximum(q+update,lower),upper)
        transform=self.fk(q)
        position_error=torch.linalg.vector_norm(transform[:,:3,3]-target_position,dim=-1)
        # Frobenius rotation distance is2sqrt(2)sin(theta/2); unlike acos(trace),
        # it remains accurate for small float32 orientation errors.
        rotation_error=2*torch.asin(torch.clamp(torch.linalg.matrix_norm(
            target_rotation-transform[:,:3,:3],ord='fro')/math.sqrt(8.),max=1.))
        valid=(torch.isfinite(q).all(dim=-1)&(position_error<=cfg['position_tolerance_m'])
               &(rotation_error<=cfg['rotation_tolerance_rad']))
        return q,transform,valid,position_error,rotation_error

    def compose(self,base,delta_m,*,return_diagnostics=False):
        if (base.ndim<1 or delta_m.ndim<1 or base.shape[-1]!=7 or delta_m.shape!=(*base.shape[:-1],3)
            or base.device!=delta_m.device or base.dtype!=delta_m.dtype
            or not base.is_floating_point() or base.dtype not in (torch.float32,torch.float64)):
            raise ValueError('Cartesian composition requires same-device float32/64 base[...,7] and delta[...,3]')
        if not torch.isfinite(base).all() or not torch.isfinite(delta_m).all():
            raise ValueError('Cartesian base actions and deltas must be finite')
        flat=base.reshape(-1,7);delta=delta_m.reshape(-1,3);q0=flat[:,:6]
        if torch.any(q0<self.lower.to(q0)-1e-6) or torch.any(q0>self.upper.to(q0)+1e-6):
            raise ValueError('Project base joints into URDF bounds before Cartesian composition')
        # Exact zero execution/critic replay has no IK work. Keep the solve in
        # actor-gradient paths: although its value is the base action at zero,
        # its derivative with respect to an infinitesimal XYZ action is not zero.
        needs_gradient=torch.is_grad_enabled() and (base.requires_grad or delta_m.requires_grad)
        zero=(delta==0).all(dim=-1) if not needs_gradient else torch.zeros(
            len(flat),dtype=torch.bool,device=base.device)
        if len(flat) and bool(zero.all()):
            if not return_diagnostics:return base
            shape=base.shape[:-1]
            return base,dict(accepted_scale=base.new_ones(shape),
                fallback=torch.zeros(shape,dtype=torch.bool,device=base.device),
                position_error_m=base.new_zeros(shape),rotation_error_rad=base.new_zeros(shape))
        transform0=self.fk(q0);position0=transform0[:,:3,3];rotation0=transform0[:,:3,:3]
        output_q=q0
        accepted=zero
        accepted_scale=zero.to(base.dtype)
        error_p=base.new_zeros(len(flat));error_r=base.new_zeros(len(flat))
        for scale in self.settings['backoff_scales']:
            ids=torch.nonzero(~accepted,as_tuple=True)[0]
            if not len(ids):break
            requested=delta[ids]*scale
            q,achieved,valid,ep,er=self._solve(q0[ids],position0[ids]+requested,rotation0[ids])
            # Numerical root error must not create a correction larger than the
            # requested component envelope (within the documented IK tolerance).
            shift=achieved[:,:3,3]-position0[ids]
            valid=valid&(shift.abs()<=requested.abs()+self.settings['position_tolerance_m']).all(dim=-1)
            selected=torch.where(valid[:,None],q,q0[ids])
            output_q=output_q.index_copy(0,ids,selected)
            accepted=accepted.index_copy(0,ids,valid)
            accepted_scale=accepted_scale.index_copy(0,ids,torch.where(valid,base.new_full(ep.shape,scale),base.new_zeros(ep.shape)))
            error_p=error_p.index_copy(0,ids,torch.where(valid,ep,base.new_zeros(ep.shape)))
            error_r=error_r.index_copy(0,ids,torch.where(valid,er,base.new_zeros(er.shape)))
        result=torch.cat((output_q,flat[:,6:7]),dim=-1).reshape_as(base)
        if not return_diagnostics:return result
        shape=base.shape[:-1]
        return result,dict(accepted_scale=accepted_scale.reshape(shape).detach(),
                           fallback=(~accepted).reshape(shape).detach(),
                           position_error_m=error_p.reshape(shape).detach(),
                           rotation_error_rad=error_r.reshape(shape).detach())

    def forward(self,base,delta_m):
        return self.compose(base,delta_m)
