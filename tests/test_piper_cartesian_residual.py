"""Cartesian action composition is geometric, differentiable and checkpoint-portable."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from pressb.robots.piper.cartesian_residual import (
    PiperCartesianXYZComposer,build_piper_xyz_spec,validate_piper_xyz_spec)
from pressb.robots.piper.kinematics import PiperKinematics

ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf'
pytestmark=pytest.mark.skipif(not URDF.is_file(),reason='Official PiPER asset not available')


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def composer():
    return PiperCartesianXYZComposer(build_piper_xyz_spec(URDF)).double()


def base_action(dtype=torch.float64):
    return torch.tensor([.2,1.1,-1.4,.2,.4,-.3,.04],dtype=dtype)


def independent_errors(base,result,delta):
    kin=PiperKinematics(URDF,tip_offset=.1358)
    b=kin.fk(base.detach().cpu().numpy()[:6]);r=kin.fk(result.detach().cpu().numpy()[:6])
    return np.linalg.norm(r[:3,3]-b[:3,3]-delta), Rotation.from_matrix(b[:3,:3]@r[:3,:3].T).magnitude()


def test_portable_spec_and_fk_match_independent_urdf():
    spec=build_piper_xyz_spec(URDF)
    assert validate_piper_xyz_spec(json.loads(json.dumps(spec)))==spec
    model=PiperCartesianXYZComposer(spec)
    spec['lower'][0]=999  # constructor owns its copy
    assert model.spec['lower'][0]!=999
    kin=PiperKinematics(URDF,tip_offset=.1358)
    q=torch.tensor([[.2,1.1,-1.4,.2,.4,-.3],[-.2,.9,-1.2,.1,.7,.2]],dtype=torch.float64)
    np.testing.assert_allclose(model.fk(q).detach().numpy(),np.stack([kin.fk(x) for x in q.numpy()]),atol=1e-12,rtol=0)
    assert model.spec['joint_names']==kin.joint_names
    assert model.spec['tip_offset_m']==.1358


def test_analytic_jacobian_matches_independent_finite_difference(composer):
    q=base_action()[:6]
    _,jac=composer.fk(q,jacobian=True)
    kin=PiperKinematics(URDF,tip_offset=.1358)
    h=1e-6;expected=np.zeros((6,6))
    for j in range(6):
        delta=np.zeros(6);delta[j]=h
        plus=kin.fk(q.numpy()+delta);minus=kin.fk(q.numpy()-delta)
        expected[:3,j]=(plus[:3,3]-minus[:3,3])/(2*h)
        expected[3:,j]=Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*h)
    np.testing.assert_allclose(jac.numpy(),expected,atol=2e-10,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_xyz_translation_holds_orientation_and_gripper(composer,dtype):
    model=composer.to(dtype=dtype)
    base=base_action(dtype).repeat(2,4,1)
    base[1,:,6]=.1
    delta=torch.tensor([.01,-.01,.01],dtype=dtype).expand(2,4,3)
    result,diag=model.compose(base,delta,return_diagnostics=True)
    assert result.shape==(2,4,7)
    assert torch.equal(result[...,6],base[...,6])
    assert torch.all(diag['accepted_scale']==1)
    assert torch.all(result[...,:6]>=model.lower) and torch.all(result[...,:6]<=model.upper)
    for b,r,d in zip(base.reshape(-1,7),result.reshape(-1,7),delta.reshape(-1,3)):
        p,angle=independent_errors(b,r,d.numpy())
        assert p<1e-5 and angle<1e-4
    assert torch.all((result[...,:6]-base[...,:6]).abs()<=.35+1e-7)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_residual_identity_and_no_solver_state(composer,dtype):
    model=composer.to(dtype=dtype)
    base=base_action(dtype).repeat(3,2,1)
    zeros=torch.zeros((3,2,3),dtype=dtype)
    assert torch.equal(model(base,zeros),base)
    before=model(base,zeros)
    model(base,torch.ones_like(zeros)*.01)
    assert torch.equal(model(base,zeros),before)


def test_non_gradient_zero_path_skips_ik_without_losing_actor_zero_derivative(composer,monkeypatch):
    base=base_action()[None]
    def forbidden(*args,**kwargs):
        raise AssertionError('Zero replay actions must not invoke IK')
    monkeypatch.setattr(composer,'_solve',forbidden)
    result=composer(base,torch.zeros((1,3),dtype=torch.float64))
    assert result is base
    with torch.no_grad():
        assert composer(base,torch.zeros((1,3),dtype=torch.float64,requires_grad=True)) is base
    with pytest.raises(AssertionError,match='invoke IK'):
        composer(base,torch.zeros((1,3),dtype=torch.float64,requires_grad=True))


@pytest.mark.parametrize('values',[[0.,0.,0.],[.007,-.009,.004]])
def test_real_autograd_matches_numerical_derivative(composer,values):
    base=base_action()[None]
    delta=torch.tensor([values],dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([.3,-.7,.6,-.2,.4,1.],dtype=torch.float64)
    result=composer(base,delta)
    gradient=torch.autograd.grad((result[:,:6]*weights).sum(),delta)[0][0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    expected=[];h=1e-6
    for axis in range(3):
        high=delta.detach().clone();low=high.clone();high[0,axis]+=h;low[0,axis]-=h
        expected.append(float(((composer(base,high)[:,:6]-composer(base,low)[:,:6])*weights).sum()/(2*h)))
    np.testing.assert_allclose(gradient.numpy(),expected,rtol=2e-4,atol=2e-5)


def test_unreachable_translation_returns_base_with_zero_gradient(composer):
    base=base_action()[None]
    delta=torch.full((1,3),10.,dtype=torch.float64,requires_grad=True)
    result,diag=composer.compose(base,delta,return_diagnostics=True)
    assert diag['fallback'].all() and (diag['accepted_scale']==0).all()
    assert torch.equal(result,base)
    derivative=torch.autograd.grad(result.sum(),delta)[0]
    assert torch.equal(derivative,torch.zeros_like(derivative))


def test_local_branch_projection_uses_smaller_xyz_not_rotated_target(composer):
    base=torch.tensor([[.1,.667,-.432,-.157,.701,.178,.04]],dtype=torch.float64)
    delta=torch.tensor([[-.017673820561542812,-.0174835933030169,-.015056557089023639]],dtype=torch.float64)
    result,diag=composer.compose(base,delta,return_diagnostics=True)
    assert diag['accepted_scale'].item()==.5 and not diag['fallback'].item()
    position_error,angle=independent_errors(base[0],result[0],delta[0].numpy()*.5)
    assert position_error<5e-5 and angle<1e-3
    assert (result[:,:6]-base[:,:6]).abs().max()<=.35+1e-12
    assert result[0,6]==base[0,6]


@pytest.mark.parametrize('field,value',[
    ('tip_offset_m',-1),('urdf_sha256','bad'),('joint_names',['joint1']*6),('lower',[0]*6),
])
def test_invalid_portable_spec_rejected(field,value):
    spec=build_piper_xyz_spec(URDF);spec[field]=value
    with pytest.raises(ValueError):validate_piper_xyz_spec(spec)


@pytest.mark.parametrize('key,value',[('iterations',0),('damping',0),('backoff_scales',[.5,1]),('max_joint_delta_rad',float('inf'))])
def test_invalid_solver_options_rejected(key,value):
    spec=build_piper_xyz_spec(URDF);spec['solver'][key]=value
    with pytest.raises(ValueError):PiperCartesianXYZComposer(spec)


def test_invalid_actions_fail_before_solving(composer):
    base=base_action()[None];delta=torch.zeros((1,3),dtype=torch.float64)
    with pytest.raises(ValueError):composer(base,torch.zeros(1,4,dtype=torch.float64))
    invalid=base.clone();invalid[0,1]=99
    with pytest.raises(ValueError,match='bounds'):composer(invalid,delta)
    with pytest.raises(ValueError,match='finite'):composer(base,delta+float('nan'))
