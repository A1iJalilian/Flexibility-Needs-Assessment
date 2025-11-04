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

function gen_scenarios(profile, settings; normalize_by=nothing, is_pv::Bool=false)

    # Extract parameters
    T = settings.T
    t0 = settings.t0
    N_scen = settings.N_scen
    rho_time = settings.rho_time
    forecast_err = is_pv ? settings.forecast_err_pv : settings.forecast_err

    # --- Convert DataFrame to numeric matrix if needed ---
    prof_mat = profile isa DataFrame ? Matrix(profile) : profile
    n = size(prof_mat, 2)

    # --- Mean and standard deviation for each time step ---
    mu = Array{Float64}(undef, n, T)
    sigma = Array{Float64}(undef, n, T)
    for t in 1:T
        mu[:, t] = collect(prof_mat[t + t0 - 1, :])
        sigma[:, t] = mu[:, t] .* forecast_err
    end

    # --- Spatial correlation matrix ---
    corr_matrix = fill(forecast_err, n, n) + I(n) * (1 - forecast_err)

    # --- Generate spatio-temporal scenarios ---
    scenarios = generate_spatiotemporal_scenarios(mu, sigma, corr_matrix, rho_time, N_scen)

    # --- Post-processing ---
    if !isnothing(normalize_by)
        scenarios ./= normalize_by
    end

    # --- Clip PV scenarios to [0, 1] ---
    if is_pv
        @. scenarios = clamp(scenarios, 0.0, 1.0)
    end

    return scenarios
end
