import numpy as np
import matplotlib.pyplot as plt

def plot_shape(shape, ax, color='blue', label=None):
    ax.plot(shape[:, 0], shape[:, 1], 'o-', color=color, label=label)
    ax.set_aspect('equal')

def main():
    np.random.seed(0)
    theta = np.linspace(0, 2*np.pi, 50)
    shape = np.c_[np.cos(theta), np.sin(theta)]
    noise = 0.05 * np.random.randn(*shape.shape)
    shape_noisy = shape + noise

    fig, ax = plt.subplots()
    plot_shape(shape_noisy, ax, color='blue', label='Sample Shape')
    ax.legend()
    plt.title("Mean Shape Example")
    plt.savefig("results/mean_shape.png", dpi=150)
    print("Saved: results/mean_shape.png")

if __name__ == "__main__":
    main()
