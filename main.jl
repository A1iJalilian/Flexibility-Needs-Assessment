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
V0 = 1                                      # Base voltage at the substation
V0_min = 0.95                               # Minimum voltage
V0_max = 1.05                               # Maximum voltage
s_base = data_math["settings"]["sbase"]     # Base apparent power
ρ = 4                                       # Number of sides in the polygonal approximation of the circle
T = 24                 # Number of time periods
t0 = 100                # Starting time period
N_scen = 100          # Number of scenarios
forecast_err = 0.3     # Forecast error (fraction)
forecast_err_pv = 0.2  # PV forecast error (fraction)
rho_time = 0.8         # Temporal correlation coefficient
pf_min, pf_max = 0.85, 0.95


# Sets
R, X = compute_R_X(data_math)                                           # Compute R and X for all loads and buses
L2load = compute_lines_downstream_loads(data_math)                      # Compute downstream loads for each line
L = collect(keys(data_math["load"]))                                    # Set of loads
P_max = Dict(l => rand([5]) ./ s_base for l in L)                         # Maximum power limits for loads
Q_max = Dict(l => rand([2.0]) ./ s_base for l in L)                         # Maximum reactive power limits for loads
pv_capacity = rand([3, 4, 5], length(L)) .* rand([0, 1], length(L))  ./ s_base     # PV capacities at each load
#################################################################################
# Generate load scenarios
mu = Array{Float64}(undef, length(L), T)
sigma = Array{Float64}(undef, length(L), T)

for t in 1:T
    mu[:, t] = collect(network.load_profiles[t+t0-1, :])   # mean load at time t
    sigma[:, t] = mu[:, t] .* forecast_err
end
corr_matrix = fill(forecast_err, length(L), length(L)) + I(length(L)) * (1 - forecast_err)

# Generate all scenarios with temporal correlation (N_scen × L × T)
load_scenarios = generate_spatiotemporal_scenarios(mu, sigma, corr_matrix, rho_time, N_scen)

# Generate PV scenarios (N_scen × 1 ×T)
mu = Array{Float64}(undef, 1, T)
sigma = Array{Float64}(undef, 1, T)
for t in 1:T
    mu[:, t] = collect(network.pv_profile[t+t0-1, :])   # mean PV at time t
    sigma[:, t] = mu[:, t] .* forecast_err_pv
end
corr_matrix_pv = fill(forecast_err_pv, 1, 1) + I(1) * (1 - forecast_err_pv)
pv_scenarios = generate_spatiotemporal_scenarios(mu, sigma, corr_matrix_pv, rho_time, N_scen)

# plot(1:T, load_scenarios[1:100, 5, :]', legend=false,
#      xlabel="Hour", ylabel="Load (kW)",
#      title="Sample Temporal Load Scenarios")
#################################################################################
# Optimization matrices
# Calculating the constraints matrices
A_v, B_v, C_v = Voltage_eq_mats(V0, V0_min, V0_max, L, R, X)
A_l, B_l, C_l = compute_line_flow_constraints(L, L2load, data_math, ρ)
A_c, B_c, C_c = compute_customer_constraints_box(L, P_max, Q_max)
A = [A_v; A_l; A_c]
B = [B_v; B_l; B_c]
C = [C_v; C_l; C_c];
AB_red, C = remove_redundant_constraints(hcat(A, B), C)
A, B = AB_red[:, 1:length(L)], AB_red[:, length(L)+1:end]
;

# --- Compute D matrix with random power factors ---
γ_pf = rand(length(L)) .* (pf_max - pf_min) .+ pf_min
γ = sqrt.(1 ./(γ_pf.^2) .- 1)
Γ_t = [Diagonal(γ .* (1 .+ 0.02 .* randn(length(L)))) for t in 1:T]
D_t = [A + B * Γ_t[t] for t in 1:T]

# Compute the PV-related sensitivity column
d_pv_col = A * pv_capacity    # M×1

# Extend D_t for all time steps
D_t_ext = [hcat(D_t[t], d_pv_col) for t in 1:T]

λrD = fill(10.0, length(L))
λrU = fill(8.0, length(L))
Pmax_vec = [P_max[l] for l in L]
Pmin_vec = -Pmax_vec
Qmax_vec = [Q_max[l] for l in L]

# Build hatξ_t as a vector of ((n+1)×N) matrices
hatξ_t = [vcat(
             permutedims(load_scenarios[:, :, t]) ./ s_base,     # (n_loads × N_scen)
             pv_scenarios[:, 1, t]'                              # (1 × N_scen)
         ) for t in 1:T]


# Solve the model
include("model.jl")
result = model_feasibility(A, B, C, D_t_ext, hatξ_t, Pmax_vec, Pmin_vec, Qmax_vec, λrD, λrU;
               θ=0.1/s_base, ε=0.1, Δt=1.0, T=T, p_norm=2)

plot(result[:P_plus]'.*s_base, xlabel="Time Period", ylabel="Power (kW)",
     title="Optimal Upward Flexibility Schedule", legend=false)

plot(result[:P_minus]'.*s_base, xlabel="Time Period", ylabel="Power (kW)",
     title="Optimal Downward Flexibility Schedule", legend=false)