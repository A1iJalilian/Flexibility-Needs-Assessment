using PowerModelsDistribution
using Plots
using Random, StatsBase, LinearAlgebra
using Serialization
include("PowerNetworkData.jl")
include("CalcNetParam.jl")
include("OptMats.jl")
include("Scenario_generator.jl")
include("model.jl")

# Load the distribution network model  
network = power_network("feeder 2")
eng = network.eng
data_math = transform_data_model(eng)

#################################################################################
# Study parameters
Base.@kwdef struct SimulationSettings
    V0::Float64 = 1.0                     # Base voltage at the substation
    V0_min::Float64 = 0.9                 # Minimum voltage
    V0_max::Float64 = 1.1                 # Maximum voltage
    ρ::Int = 4                            # Number of sides in the polygonal approximation of the circle
    T::Int = 24                           # Number of time periods
    t0::Int = 3000                        # Starting time period
    N_scen::Int = 50                      # Number of scenarios
    forecast_err::Float64 = 0.3           # Forecast error (fraction)
    forecast_err_pv::Float64 = 0.2        # PV forecast error (fraction)
    rho_time::Float64 = 0.8               # Temporal correlation coefficient
    pf_min::Float64 = 0.85                # Minimum power factor
    pf_max::Float64 = 0.95                # Maximum power factor
    ψ::Float64 = 0.05                     # Voltage unbalance limit
    cδψ::Float64 = 500.0                  # Voltage unbalance violation cost
    P_max::Vector{Float64} = [5.0]        # Maximum active power limits for loads (per unit)
    Q_max::Vector{Float64} = [2.0]        # Maximum reactive power limits for loads (per unit)
    pv_c::Vector{Float64} = [3, 5, 7]    # PV capacities at each load (per unit)
    λrD::Float64 = 10.0                   # Downward reserve price ($/kW)
    λrU::Float64 = 8.0                    # Upward reserve price ($/kW)
end

# Sets and parameters
settings = SimulationSettings()
s_base = data_math["settings"]["sbase"]
L = collect(keys(data_math["load"]))                                    # Set of loads
P_max = Dict(l => rand(settings.P_max) ./ s_base for l in L)                         # Maximum power limits for loads
Q_max = Dict(l => rand(settings.Q_max) ./ s_base for l in L)                         # Maximum reactive power limits for loads
λrD = fill(settings.λrD, length(L))
λrU = fill(settings.λrU, length(L))
Pmax_vec = [P_max[l] for l in L]
Pmin_vec = -Pmax_vec
Qmax_vec = [Q_max[l] for l in L]
pv_capacity = rand(settings.pv_c, length(L)) .* rand([0, 1], length(L)) ./ s_base     # PV capacities at each load

#################################################################################
# Generate load and PV scenarios
load_scenarios = gen_scenarios(network.load_profiles, settings; normalize_by=s_base)
pv_scenarios = gen_scenarios(network.pv_profile, settings; is_pv=true)


hatξ_t = [vcat(
    1.5 .* permutedims(load_scenarios[:, :, t]),  # (n_loads × N_scen)
    pv_scenarios[:, 1, t]'                 # (1 × N_scen)
) for t in 1:settings.T]

n_loads = size(load_scenarios, 2)
nrows = ceil(Int, n_loads / 6)

plt = plot(layout=(nrows, 6), size=(1800, 250 * nrows))

for i in 1:n_loads
    net_load = (1.5 .* load_scenarios[1:settings.N_scen, i, :]' .- pv_scenarios[1:settings.N_scen, 1, :]' .* pv_capacity[i]) .* s_base
    plot!(plt[i], 1:settings.T, net_load,
        legend=false, fontfamily="Times", fontsize=10,
        xlabel="Hour", ylabel="Load (kW)",
        title="Load #$i")
end

display(plt)

#################################################################################
# Optimization matrices
A, B, C, D_t_ext, flex_constraints = calculateABCD(data_math, settings, P_max, Q_max, pv_capacity)

# Solve the model
include("model.jl")
result = model_CVaR(A, B, C, D_t_ext, flex_constraints, hatξ_t, Pmax_vec, Pmin_vec, Qmax_vec, λrD, λrU, settings.cδψ;
    θ=0.05 / s_base, ε=0.1, Δt=1.0, p_norm=2)

result = model_CVaR_CCG(A, B, C, D_t_ext, flex_constraints, hatξ_t, Pmax_vec, Pmin_vec, Qmax_vec, λrD, λrU, settings.cδψ;
    θ=0.05 / s_base, ε=0.1, Δt=1.0, p_norm=2, solver=Gurobi.Optimizer,
    tol=1e-7, max_iters=50, initial_active=false, quiet=false)

result = model_MILP_CCG(A, B, C, D_t_ext, flex_constraints, hatξ_t, Pmax_vec, Pmin_vec, Qmax_vec, λrD, λrU, settings.cδψ;
    θ=0.05 / s_base, ε=0.1, Δt=1.0, p_norm=2, solver=Gurobi.Optimizer, 
    tol=1e-7, max_iters=200, initial_active=false, quiet=false,
    bigM=10e8, compute_bigM=false, bigM_factor=10.0)

n_loads = size(result[:P_plus], 1)
T = size(result[:P_plus], 2)
nrows = ceil(Int, n_loads / 6)

plt = plot(layout=(nrows, 6), size=(1800, 250 * nrows))

for i in 1:n_loads
    # Extract data for load i
    p_plus = result[:P_plus][i, :] .* s_base
    p_minus = result[:P_minus][i, :] .* s_base

    # Plot both upward & downward flexibility
    plot!(plt[i], 1:T, p_plus, label="Upward", color=:blue)
    plot!(plt[i], 1:T, p_minus, label="Downward", color=:red)

    plot!(plt[i],
        xlabel="Time Period",
        ylabel="Power (kW)",
        title="Load #$i Flexibility"
    )
end

display(plt)
