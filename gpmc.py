import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import gpflow
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from gpflow.utilities import print_summary
# from multiclass_classification import plot_from_samples, colors

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

np.random.seed(42)

X = np.linspace(0,2*np.pi, 40)
Y = (np.sin(X)+1).astype(np.float64)
# Y = np.random.poisson((np.sin(X)+1)).astype(np.float64)

plt.figure()
plt.plot(X, Y, "x")
plt.xlabel("input $X$")
plt.ylabel("output $Y$")
plt.title("toy dataset")
plt.show()

data = (X[:, None], Y[:, None])
data2 = (X, Y)


kernel = gpflow.kernels.Matern32()# + gpflow.kernels.Constant()
likelihood = gpflow.likelihoods.Poisson()
model = gpflow.models.SGPMC(data2, kernel,likelihood = likelihood)
# model = gpflow.models.GPMC(data, kernel, likelihood)

# model.likelihood.variance.assign(0.01)
# model.kernel.lengthscales.assign(0.1)

# model.kernel.kernels[0].lengthscales.prior = tfd.Poisson(f64(0.25))
# model.kernel.kernels[0].variance.prior = tfd.Poisson(f64(6.0))
# model.kernel.kernels[1].variance.prior = tfd.Poisson(f64(3.0))

# model.kernel.kernels[0].lengthscales.prior = tfd.Normal(f64(1.0), f64(2.0))
# model.kernel.kernels[0].variance.prior = tfd.Normal(f64(1.0), f64(2.0))
# model.kernel.kernels[1].variance.prior = tfd.Normal(f64(1.0), f64(2.0))

model.kernel.lengthscales.prior = tfd.Normal(f64(0.5), f64(0.25))
model.kernel.variance.prior = tfd.Normal(f64(2.0), f64(2.0))
# model.kernel.kernels[1].variance.prior = tfd.Normal(f64(1.0), f64(2.0))

# model.likelihood.variance.assign(11)
# model.kernel.lengthscales.assign(1)

print_summary(model)

optimizer = gpflow.optimizers.Scipy()
maxiter = ci_niter(3000)
_ = optimizer.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=maxiter)
)
# We can now start HMC near maximum a posteriori (MAP)

num_burnin_steps = ci_niter(600)
num_samples = ci_niter(1000)

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)

adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)

print_summary(model)


@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )


samples, _ = run_chain_fn()

print_summary(model)


Xtest = np.linspace(0, 6, 100)[:, None]
f_samples = []

for i in range(num_samples):
    # Note that hmc_helper.current_state contains the unconstrained variables
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    f = model.predict_f_samples(Xtest, 5)
    f_samples.append(f)
f_samples = np.vstack(f_samples)


rate_samples = np.exp(f_samples[:, :, 0])

(line,) = plt.plot(Xtest, np.mean(rate_samples, 0), lw=2)
plt.fill_between(
    Xtest[:, 0],
    np.percentile(rate_samples, 5, axis=0),
    np.percentile(rate_samples, 95, axis=0),
    color=line.get_color(),
    alpha=0.2,
)

plt.plot(X, Y, "kx", mew=2)
_ = plt.ylim(-1.25,3)

# parameter_samples = hmc_helper.convert_to_constrained_values(samples)
# param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(model).items()}
# name_to_index = {param_to_name[param]: i for i, param in enumerate(model.trainable_parameters)}
# hyperparameters = [
#     ".kernel.kernels[0].lengthscales",
#     ".kernel.kernels[0].variance",
#     ".kernel.kernels[1].variance",
# ]

parameter_samples = hmc_helper.convert_to_constrained_values(samples)
param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(model).items()}
name_to_index = {param_to_name[param]: i for i, param in enumerate(model.trainable_parameters)}
hyperparameters = [
    ".kernel.lengthscales",
    ".kernel.variance",
]


plt.figure(figsize=(8, 4))
for param_name in hyperparameters:
    plt.plot(parameter_samples[name_to_index[param_name]], label=param_name)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel("HMC iteration")
_ = plt.ylabel("hyperparameter value")

