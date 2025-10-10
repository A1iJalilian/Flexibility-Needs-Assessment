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

function generate_spatiotemporal_scenarios(mu::Matrix{Float64},
                                           sigma::Matrix{Float64},
                                           corr_space::Matrix{Float64},
                                           rho_time::Float64,
                                           J::Int;
                                           epsilon::Float64 = 1e-10,
                                           seed::Int = 42)

    rng = MersenneTwister(seed)
    N, T = size(mu)

    # Time correlation matrix (T×T)
    corr_time = [rho_time^(abs(i-j)) for i in 1:T, j in 1:T]

    # Combined covariance: space ⊗ time
    Σ = kron(corr_time, corr_space)

    # Standard deviations (flattened)
    σ_vec = vec(sigma)
    Σ = diagm(σ_vec) * Σ * diagm(σ_vec)
    Σ = (Σ + Σ') / 2 + epsilon * I(N*T)

    # Cholesky factor
    L = cholesky(Σ).L

    # Random draws
    rnd = randn(rng, N*T, J)
    scen = (vec(mu) .+ L * rnd)'

    # Reshape to (J × N × T)
    return reshape(scen, J, N, T)
end
