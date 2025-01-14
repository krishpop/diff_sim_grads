import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import math

wp.init()

class BounceOnce:

    render_time = 0.0

    def __init__(self, cfg, integrator, render=True, profile=False, adapter='cuda',
                 noise=None, seed=123):

        self.frame_dt = 1.0/60.0
        self.frame_steps = int(cfg.simulation_time/self.frame_dt)
        self.sim_dt = cfg.dt
        self.sim_steps = cfg.steps
        self.sim_substeps = int(self.sim_steps/self.frame_steps)
        self.eps = cfg.eps
        self.noise = noise
        if self.noise is None:
            self.num_samples = 1
        else:
            self.num_samples = cfg.num_samples
        if self.noise is not None:
            np.random.seed(seed)

        builder = warp.sim.ModelBuilder()

        # default up axis is y
        builder.add_particle(pos=(cfg.init_pos[0], cfg.init_pos[1], 0.0), vel=(cfg.init_vel[0], cfg.init_vel[1], 0.0), mass=1.0)

        self.device = adapter
        self.profile = profile

        self.model = builder.finalize(adapter, requires_grad=True)

        self.model.ground = True
        self.model.gravity[1] = 0 # no gravity
        self.model.particle_radius = cfg.radius
        # type of simulation
        self.model.customized_particle_ground_wall = False
        self.model.customized_particle_bounce_once = True

        self.custom_integrator  = not isinstance(
                integrator, (wp.sim.SemiImplicitIntegrator, wp.sim.XPBDIntegrator))
        # soft contact properties
        if self.custom_integrator:
            self.model.customized_kn = cfg.customized_kn
        else:
            self.model.soft_contact_ke = cfg.customized_kn
            self.model.soft_contact_kf = 0.0
            self.model.soft_contact_kd = 1.e+1
            self.model.soft_contact_mu = 0.
            self.model.soft_contact_margin = 10.0
        #if isinstance(self.integrator, wp.sim.SemiImplicitIntegrator):
        #else:
        #    sle
        # self.model.customized_kd = self.customized_kd
        # self.model.customized_kf = self.customized_kf
        # self.model.customized_mu = self.customized_mu

        self.integrator = integrator

        self.loss = wp.zeros(1, dtype=wp.float32, device=adapter, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps+1):
            state = self.model.state(requires_grad=True)
            if self.custom_integrator:
                state.external_particle_f = wp.array([
                    [0, 0, 0]
                ], dtype=wp.vec3, device=adapter, requires_grad=True)
            self.states.append(state)

        self.save_dir = os.path.join(cfg.THIS_DIR, cfg.result_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        if render:
            self.stage = warp.sim.render.SimRenderer(
                self.model,
                os.path.join(self.save_dir, cfg.name+".usd")
            )


    @wp.kernel
    def terminal_loss_kernel(
        pos: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=float),
    ):
        mask = wp.vec3(0., 1., 0.)
        wp.atomic_add(loss, 0, wp.dot(pos[0], mask))

    @wp.kernel
    def step_kernel(
        x: wp.array(dtype=wp.vec3),
        grad: wp.array(dtype=wp.vec3),
        alpha: float
    ):
        tid = wp.tid()
        # gradient descent step
        x[tid] = x[tid] - grad[tid]*alpha

    def compute_loss(self):

        self.loss.zero_()
        for i in range(self.sim_steps):

            self.states[i].clear_forces()

            if not self.custom_integrator:
                wp.sim.collide(self.model, self.states[i])

            self.integrator.simulate(
                self.model,
                self.states[i],
                self.states[i+1],
                self.sim_dt
            )

        # # compute loss on final state
        wp.launch(self.terminal_loss_kernel, dim=1, inputs=[self.states[-1].particle_q,self.loss], device=self.device)
        return self.loss

    def render(self):

        for i in range(0, self.sim_steps, self.sim_substeps):

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_points("particles", self.states[i].particle_q.numpy(), radius=self.model.particle_radius)
            self.stage.end_frame()

            self.render_time += self.frame_dt

        self.stage.save()

    def get_gradient(self):
        tape = wp.Tape()

        with tape:
            self.compute_loss()
        print(f"Height: {self.loss}")
        loss_np = self.loss.numpy()[0]
        tape.backward(self.loss)

        # get gradient
        x = self.states[0].particle_q
        x_grad = tape.gradients[self.states[0].particle_q].numpy()[0][1]
        tape.zero()
        return x_grad

    def check_grad(self, param):
        for _ in range(self.num_samples):
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param].numpy()[0][1]
        tape.zero()

        return x_grad_analytic

    def check_grad_numerical(self, x):
        x0 = x.numpy()
        x2, x1 = x0 + self.eps, x0 - self.eps
        x.assign(x1)
        y1 = self.compute_loss().numpy()[0]
        x.assign(x2)
        y2 = self.compute_loss().numpy()[0]
        numerical = (y2 - y1) / (2 * self.eps)
        return numerical
