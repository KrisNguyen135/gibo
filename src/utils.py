import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


# plt.style.use("bmh")
# plt.rcParams["image.cmap"] = "Blues"


def plot_along_direction(x, y, model, direction, obj_func, delta_x=0.01, num_steps=101, num_obj_samples=5):
    sample_values = torch.tensor(
        [obj_func(model.train_inputs[0][i]) for i in range(model.train_inputs[0].shape[-2])]
    )

    print(model.covar_module.base_kernel.lengthscale.detach().numpy())
    print(model.covar_module.outputscale.item())
    print(model.likelihood.noise.item())

    direction = torch.nn.functional.normalize(direction)

    assert num_steps % 2 == 1
    half_num_steps = num_steps // 2

    test_x = []
    x_ticks = []
    for i in range(half_num_steps, 0, -1):
        test_x.append(x - delta_x * i * direction)
        x_ticks.append(- delta_x * i)
    for i in range(half_num_steps + 1):
        test_x.append(x + delta_x * i * direction)
        x_ticks.append(delta_x * i)

    test_x = torch.vstack(test_x)

    with torch.no_grad():
        pred_distribution = model.posterior(test_x).mvn
        mean = pred_distribution.mean
        cov = pred_distribution.covariance_matrix
        lower, upper = model.posterior(test_x, observation_noise=True).mvn.confidence_region()

        np.random.seed(0)
        samples = np.random.multivariate_normal(mean.detach().numpy(), cov.detach().numpy(), size=3)

    test_y = []
    for i in tqdm(range(num_steps)):
        test_y.append([])
        for _ in range(num_obj_samples):
            test_y[-1].append(obj_func(test_x[[i], :]))

    test_y = torch.tensor(test_y)
    print(test_y.shape)

    plt.figure(figsize=(8, 6))

    # x_ticks = (np.arange(num_steps) - half_num_steps) * delta_x

    for j in range(num_obj_samples):
        plt.scatter(
            x_ticks,
            test_y[:, j],
            label="objective function values",
            c="C1",
            marker=".",
            alpha=0.5,
        )

    plt.plot(
        x_ticks,
        test_y.mean(axis=1),
        label="objective function",
        c="C1",
    )

    plt.fill_between(
        x_ticks,
        test_y.mean(axis=1) - 2 * test_y.std(axis=1),
        test_y.mean(axis=1) + 2 * test_y.std(axis=1),
        color="C1",
        alpha=0.3,
    )

    plt.plot(x_ticks, mean, label="mean")
    plt.fill_between(x_ticks, lower, upper, alpha=0.3, label="95% CI")

    # for i in range(3):
    #     plt.plot(x_ticks, samples[i, :], linestyle="--", alpha=0.5)

    # plt.scatter(0, y, c="k", marker="D", s=100, label="incumbent")

    # scatter_mask = torch.isclose(
    #     torch.nn.functional.cosine_similarity(
    #         direction, model.train_inputs[0] - x
    #     ).abs(),
    #     torch.ones(1)
    # )
    #
    # scatter_x = (model.train_inputs[0][scatter_mask, :] - x).pow(2).sum(axis=1).sqrt()
    # scatter_y = model.train_targets[scatter_mask]
    #
    # distance_mask = scatter_x <= half_num_steps * delta_x
    #
    # plt.scatter(
    #     scatter_x[distance_mask],
    #     scatter_y[distance_mask],
    #     c="orange",
    #     marker="X",
    #     s=200,
    #     label="observations",
    # )

    plt.scatter(
        0,
        y,
        label="incumbent",
        c="k",
        marker="x",
        s=100,
    )

    distances = (model.train_inputs[0] - x).pow(2).sum(axis=1).sqrt()
    distances = distances[distances < plt.xlim()[1]]

    # plt.scatter(distances, plt.ylim()[0] * np.ones_like(distances), marker="|", s=200, c="k")

    # plt.xlabel("alpha")
    # plt.ylabel("obj. values")
    # plt.title("predictions at (incumbent + alpha * direction)")
    # plt.legend(fontsize=20)

    plt.xlim((-0.75, 0.75))
    plt.ylim((-0.5, 1.5))

    # plt.show()
