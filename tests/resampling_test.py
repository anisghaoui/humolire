import numpy as np

from humolire.PDR.Resampling import multinomial, systematic, residual, stratified, kong_criterion, pham_criterion, \
    metropolis, rejection

np.set_printoptions(precision=3)


def main():
    seed = 0  # fix this seed for repeatability
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    N = 10
    U = np.array(sorted(rng.uniform(0, 1, size=N)))
    weights = rng.uniform(0, 1, size=N)
    weights = weights / np.sum(weights)

    # emmanuel's hand computed values, if you don't use the random ones
    # weights = np.array([0.01, 0.09, 0.23, 0.01, 0.04, 0.12, 0.18, 0.06, 0.08, 0.18])
    # U = np.array([0.05, 0.09, 0.17, 0.29, 0.35, 0.42, 0.55, 0.64, 0.78, 0.92])

    print(f"Kong criterion = {kong_criterion(weights)}")
    print(f"Pham criterion = {pham_criterion(weights)}")

    print("\nmultinomial:")
    print(f"before weights {weights}")

    new_i = multinomial(weights, U)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")

    print("\nstratified:")
    print(f"before weights {weights}")

    new_i = stratified(weights)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")

    print("\nsystematic:")
    print(f"before weights {weights}")

    new_i = systematic(weights)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")

    print("\nresidual:")
    print(f"before weights {weights}")

    new_i = residual(weights, rng=rng)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")

    print("\nmetropolis:")
    print(f"before weights {weights}")

    new_i = metropolis(weights)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")

    print("\nrejection:")
    print(f"before weights {weights}")

    new_i = rejection(weights)
    new_weights = weights[new_i] / np.sum(weights[new_i])

    print(f"U =\t\t\t   {U}")
    print(f"after weights  {new_i}")
    print(f"Kong criterion = {kong_criterion(new_weights)}")
    print(f"Pham criterion = {pham_criterion(new_weights)}")


if __name__ == "__main__":
    main()
