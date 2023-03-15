import os
import warp as wp
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys
sys.path.append(PARENT_DIR)

from _bounce_once_warp import BounceOnce
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, 'bounce_once.yaml'))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.THIS_DIR = THIS_DIR

integrator = wp.sim.SemiImplicitIntegrator()
system = BounceOnce(
    cfg,
    integrator=integrator,
    adapter='cuda',
    render=True,
)
loss = system.compute_loss()
system.render()

print("------------Task 1: Compliant Model (Warp)-----------")
print(f"loss: {loss}")

particle_f = system.states[1].particle_f if not system.custom_integrator else system.states[0].external_particle_f
x_grad = system.check_grad(system.states[0].particle_q)
v_grad = system.check_grad(system.states[0].particle_qd)
ctrl0_grad = system.check_grad(particle_f)

x_grad_num = system.check_grad_numerical(system.states[0].particle_q)
v_grad_num = system.check_grad_numerical(system.states[0].particle_qd)
ctrl0_grad_num = system.check_grad_numerical(particle_f)
print(f"diff-sim analytical gradient of final height w.r.t. initial position dl/dx0: {x_grad}")
print(f"numerical gradient of final height w.r.t. initial position dl/dx0: {x_grad_num}")
print(f"diff-sim analytical gradient of final height w.r.t. initial velocity dl/dv0: {v_grad}")
print(f"numerical gradient of final height w.r.t. initial velocity dl/dv0: {v_grad_num}")
print(f"diff-sim analytical gradient of final height w.r.t. initial ctrl dl/du0: {ctrl0_grad}")
print(f"numerical gradient of final height w.r.t. initial ctrl dl/du0: {ctrl0_grad_num}")
