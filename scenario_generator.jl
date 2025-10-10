using Distributions
using LinearAlgebra
using Random

function generate_scenarios(mu, sigma::Vector{Float64}, corr_matrix::Matrix{Float64}, J::Int; epsilon::Float64=1e-10, seed::Int=42)
    # Set up the random number generator
    rng = MersenneTwister(seed)

    # Number of intervals
    S = length(mu)

    # Create the covariance matrix Σ
    Σ = diagm(sigma) * corr_matrix * diagm(sigma)

    # Ensure Σ is symmetric
    Σ = (Σ + Σ') / 2

    # Add small perturbation for numerical stability
    Σ += epsilon * I(S)

    # Cholesky decomposition
    L = cholesky(Σ).L

    # Generate standard normal samples with fixed RNG
    rnd = randn(rng, S, J)

    # Generate scenarios: X = μ + LZ
    scenarios = (mu .+ L * rnd)'

    return scenarios
end