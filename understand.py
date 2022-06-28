from numpy.random import randn, randint


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    print(x_input.shape)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    print(z_input.shape)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    print(labels.shape)
    return [z_input, labels]


print(generate_latent_points(3*70, 100))
