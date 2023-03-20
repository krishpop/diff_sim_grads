import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
import sys

sys.path.append(PARENT_DIR)
from utils.customized_integrator_euler import CustomizedSymplecticEulerIntegrator
from _ground_wall_warp import GroundWall
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sns
from tqdm import tqdm
import warp

from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(os.path.join(THIS_DIR, "ground_wall.yaml"))
cli_cfg = OmegaConf.from_cli()

cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
cfg.dt = cfg.simulation_time / cfg.steps  # 1./480
cfg.name = os.path.basename(__file__)[:-3]
cfg.name += f"_mu_{cfg.customized_mu}"
cfg.THIS_DIR = THIS_DIR

std = 0.2
runs = 2
np.random.seed(0)
# warp.rand_init(0)
thetas = np.linspace(-np.pi, np.pi, 10)
special_thetas = np.append(np.linspace(-np.pi, np.pi, 9), -2.126)
grads = []
trajectories = {}
for theta in tqdm(thetas):
    for i in range(runs):
        # add noise to theta
        noise_sample = np.random.normal(scale=std)
        theta_noisy = theta + noise_sample
        # use theta to set angle
        u = 5.0
        cfg.init_vel[0] = math.cos(theta_noisy) * u
        cfg.init_vel[1] = math.sin(theta_noisy) * u

        system = GroundWall(
            cfg,
            integrator=CustomizedSymplecticEulerIntegrator(),
            adapter="cpu",
            render=False,
        )
        loss = system.compute_loss()

        x_grad = system.check_grad(system.states[0].particle_q)
        v_grad = system.check_grad(system.states[0].particle_qd)
        theta_grad = np.tan(v_grad[0, 1] / v_grad[0, 0])
        zog_old = (
            1 / std**2
            * u
            * loss.numpy()[0]
            * noise_sample
            * np.array([-np.sin(theta_noisy), np.cos(theta_noisy)])
        )
        zog_theta_old = np.tan(zog_old[1] / zog_old[0])
        zog = (
            u * loss.numpy()[0] * np.array([-np.sin(theta_noisy), np.cos(theta_noisy)])
        )
        zog_theta = np.tan(zog[1] / zog[0])
        grads.append(
            {
                "run": i,
                "th": theta,
                "FoG": theta_grad,
                "ZoG": zog_theta,
                "ZoG_old": zog_theta_old,
                "s_cost": loss.numpy()[0],
            }
        )
        ctrl0_grad = system.check_grad(system.states[0].external_particle_f)
        # x = []
        # for i in range(len(system.states)):
        #     x.append(system.states[i].particle_q.numpy())
        # x = np.array(x).squeeze()
        # trajectories.update({theta_noisy: x})

# Now also get deterministic cost
cost = []
for theta in thetas:
    # use theta to set angle
    u = 5.0
    cfg.init_vel[0] = math.cos(theta) * u
    cfg.init_vel[1] = math.sin(theta) * u
    system = GroundWall(
        cfg,
        integrator=CustomizedSymplecticEulerIntegrator(),
        adapter="cuda",
        render=False,
    )
    loss = system.compute_loss()
    cost.append(loss.numpy()[0])

grads = pd.DataFrame(grads)
grads.to_csv("task_2")
np.save("cost_landscape", cost)
np.save("trajectories", trajectories)

# plot results
plt.figure()
sns.lineplot(grads, x="th", y="FoG", label="FoBG")
sns.lineplot(grads, x="th", y="ZoG", label="ZoBG")
sns.lineplot(grads, x="th", y="s_cost", label="F(th)")

plt.plot(thetas, cost, label="f(th)")
plt.legend()
plt.savefig("theta_grads.png")

plt.figure()
plt.scatter(cfg.init_pos[0], cfg.init_pos[1], label="init")
plt.scatter(cfg.target[0], cfg.target[1], label="target")
plt.hlines(y=0.0, xmin=-3, xmax=2, linewidth=2, color="black")
plt.vlines(x=cfg.wall_x, ymin=0.0, ymax=2.0, linewidth=2, color="black")

for k, v in trajectories.items():
    plt.plot(v[:, 0], v[:, 1], label="th={:.2f}".format(k))

plt.legend()
plt.axis("equal")
plt.savefig("trajectories.png")
