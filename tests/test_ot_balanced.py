"""Balanced MMOT analytic cases, gradients, independent groups and safeguards."""
import math
import pytest

torch = pytest.importorskip('torch')
from mpsae import ot, methods
from mpsae.ot_balanced import matching_gaps
from mpsae.config import FF2048_PRESET
from dataclasses import replace


@pytest.fixture(scope='module', autouse=True)
def restore_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def balanced(cost, **kwargs):
    return ot.matching_gap_from_cost(cost, mass=1, eta=.2, max_iters=100,
        max_refine_iters=500, tol=1e-10, solver_dtype=torch.float64, **kwargs)


def test_constant_cost_has_uniform_joint_plan_and_known_gap():
    torch.set_num_threads(1)
    n=8
    cost=torch.full((n,n,n),2/3,dtype=torch.float64,requires_grad=True)
    loss,stats,delta=balanced(cost)
    reference=torch.zeros_like(cost)
    d=torch.arange(n);reference[d,d,d]=1/n
    plan=reference-delta
    torch.testing.assert_close(plan,torch.full_like(plan,1/n**3),atol=1e-13,rtol=1e-12)
    assert loss.item()==pytest.approx(2*.2*math.log(n),abs=1e-12)
    assert stats['marginal_l1']<1e-10 and stats['mass_error']<1e-10
    torch.testing.assert_close(torch.autograd.grad(loss,cost)[0],delta)


def test_gradient_matches_finite_differences_of_resolved_problem():
    torch.manual_seed(91)
    cost=torch.rand((3,3,3),dtype=torch.float64,requires_grad=True)
    loss,_,delta=balanced(cost)
    grad,=torch.autograd.grad(loss,cost)
    torch.testing.assert_close(grad,delta)
    for index in ((0,0,0),(1,1,1),(0,1,2),(2,1,0)):
        perturb=torch.zeros_like(cost);perturb[index]=1e-5
        numeric=(balanced(cost.detach()+perturb)[0]-balanced(cost.detach()-perturb)[0])/2e-5
        assert float(numeric)==pytest.approx(float(grad[index]),abs=2e-8)


def test_production_group_batched_scalar_parity_and_group_independence():
    torch.manual_seed(1227)
    pre=torch.randn(32,128)
    views=[]
    for k in (3,6,12):
        values,indices=pre.topk(k,dim=1)
        views.append(torch.zeros_like(pre).scatter(1,indices,values).relu())
    costs=torch.stack((ot.angular_cost(*views),torch.full((32,32,32),2/3),torch.rand(32,32,32)))
    losses,records=ot._batched_matching_gap_from_costs(costs,mass=1,eta=.2,max_iters=100,
        tol=1e-4,require_feasible=True,max_refine_iters=500)
    for i,cost in enumerate(costs):
        single,stats,_=ot.matching_gap_from_cost(cost,mass=1,max_iters=100,max_refine_iters=500)
        torch.testing.assert_close(single,losses[i],atol=2e-6,rtol=1e-6)
        assert records[i]['marginal_l1']<1e-6
        assert stats['converged']
    changed=costs.clone();changed[1]=torch.rand_like(changed[1])
    other,_,_=matching_gaps(changed,eta=.2,max_iters=100,max_refine_iters=500,tol=1e-4)
    torch.testing.assert_close(losses[[0,2]],other[[0,2]],atol=2e-6,rtol=1e-6)


def test_bounded_failure_and_fp64_refinement():
    torch.manual_seed(4)
    cost=torch.rand((1,8,8,8))
    _,records,_=matching_gaps(cost,eta=.2,max_iters=1,max_refine_iters=200,tol=1e-11)
    assert records[0]['solver_float64'] and records[0]['refinement_iterations']>0
    assert records[0]['marginal_l1']<=1e-11
    with pytest.raises(RuntimeError,match='did not converge'):
        matching_gaps(cost,eta=.01,max_iters=1,max_refine_iters=1,tol=1e-14)
    with pytest.raises(ValueError):
        ot.greenkhorn(cost[0],1,.2,100,1e-4)  # partial solver still rejects full mass


def test_full_training_path_with_tail_backprop_and_identity():
    torch.manual_seed(3)
    views=[torch.rand(11,20,requires_grad=True) for _ in range(3)]
    loss,stats=ot.matching_gap(*views,mass=1,microbatch=4,max_iters=100,max_refine_iters=500)
    gradients=torch.autograd.grad(loss,views)
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in gradients)
    assert stats['groups']==3 and stats['marginal_l1']<1e-6
    fast,_=ot._matching_gap_batched(*views,mass=1,microbatch=4,max_iters=100,max_refine_iters=500)
    torch.testing.assert_close(fast,loss,atol=2e-6,rtol=1e-6)
    assert 'solver' not in methods.describe('mpsae',8,FF2048_PRESET)['ot']
    assert methods.describe('mpsae',8,replace(FF2048_PRESET,ot_mass=1))['ot']['constraints']=='balanced'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_balanced_training_matches_cpu():
    torch.manual_seed(2)
    cpu=[torch.rand(67,64,requires_grad=True) for _ in range(3)]
    gpu=[x.detach().cuda().requires_grad_() for x in cpu]
    a,_=ot.matching_gap(*cpu,mass=1,microbatch=32)
    b,_=ot.matching_gap(*gpu,mass=1,microbatch=32)
    torch.testing.assert_close(a,b.cpu(),atol=2e-5,rtol=2e-5)
    for x,y in zip(torch.autograd.grad(a,cpu),torch.autograd.grad(b,gpu)):
        torch.testing.assert_close(x,y.cpu(),atol=2e-5,rtol=2e-5)
