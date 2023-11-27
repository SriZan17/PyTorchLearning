def test_make_circles():
    n_samples = 1000
    X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)
    assert (
        len(X) == n_samples
    ), f"Test failed: Expected {n_samples} samples, got {len(X)}"
    assert (
        len(y) == n_samples
    ), f"Test failed: Expected {n_samples} labels, got {len(y)}"
    assert (
        X.shape[1] == 2
    ), f"Test failed: Expected 2 features, got {X.shape[1]} features"
