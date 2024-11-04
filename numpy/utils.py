import numpy as np
import matplotlib.pyplot as plt


def plot_transformation(T, v1, v2, vector_name='v'):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    v1_transformed = T @ v1
    v2_transformed = T @ v2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    vmin = np.floor(np.min([v1, v2, v1_transformed, v2_transformed, (v1_transformed + v2_transformed)]) - 0.5)
    vmax = np.ceil(np.max([v1, v2, v1_transformed, v2_transformed, (v1_transformed + v2_transformed)]) + 0.5)
    ax.set_xticks(np.arange(vmin, vmax))
    ax.set_yticks(np.arange(vmin, vmax))
    plt.axis([vmin, vmax, vmin, vmax])

    plt.quiver([0, 0], [0, 0], [v1[0, 0], v2[0, 0]], [v1[1, 0], v2[1, 0]], color=color_original, angles='xy',
               scale_units='xy', scale=1)
    plt.plot([0, v2[0, 0], v1[0, 0] + v2[0, 0], v1[0, 0]],
             [0, v2[1, 0], v1[1, 0] + v2[1, 0], v1[1, 0]],
             color=color_original)

    v1_sgn = 0.02 * (vmax - vmin) * np.sign(v1.flatten())
    ax.text(v1[0, 0] + v1_sgn[0], v1[1, 0] + v1_sgn[1], f'${vector_name}_1$', fontsize=14, color=color_original)

    v2_sgn = 0.02 * (vmax - vmin) * np.sign(v2.flatten())
    ax.text(v2[0, 0] + v2_sgn[0], v2[1, 0] + v2_sgn[1], f'${vector_name}_2$', fontsize=14, color=color_original)

    plt.quiver([0, 0], [0, 0], [v1_transformed[0, 0], v2_transformed[0, 0]],
               [v1_transformed[1, 0], v2_transformed[1, 0]],
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, v2_transformed[0, 0], v1_transformed[0, 0] + v2_transformed[0, 0], v1_transformed[0, 0]],
             [0, v2_transformed[1, 0], v1_transformed[1, 0] + v2_transformed[1, 0], v1_transformed[1, 0]],
             color=color_transformed)

    v1_transformed_sgn = 0.04 * (vmax - vmin) * np.sign(v1_transformed.flatten())
    ax.text(v1_transformed[0, 0] + v1_transformed_sgn[0] - 0.1 * (1 if v1_transformed[0, 0] < 0 else 0),
            v1_transformed[1, 0] + v1_transformed_sgn[1] - 0.05 * (1 if v1_transformed[1, 0] < 0 else 0),
            f'$T({vector_name}_1)$', fontsize=14, color=color_transformed)

    v2_transformed_sgn = 0.04 * (vmax - vmin) * np.sign(v2_transformed.flatten())
    ax.text(v2_transformed[0, 0] + v2_transformed_sgn[0] - 0.1 * (1 if v1_transformed[0, 0] < 0 else 0),
            v2_transformed[1, 0] + v2_transformed_sgn[1] - 0.05 * (1 if v1_transformed[1, 0] < 0 else 0),
            f'$T({vector_name}_2)$', fontsize=14, color=color_transformed)

    plt.gca().set_aspect("equal")
    plt.show()
    return fig


# Testing the function with given inputs
A = np.array([[2, 3], [2, 1]])
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

plot_transformation(A, e1, e2, vector_name='e')
