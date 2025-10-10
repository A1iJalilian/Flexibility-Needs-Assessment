using PowerModelsDistribution
using Plots
using Random, StatsBase, LinearAlgebra
using Serialization
include("PowerNetworkData.jl")
include("CalcNetParam.jl")
include("OptMats.jl")
include("Scenario_generator.jl")

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
N_scen = 1000          # Number of scenarios
forecast_err = 0.2     # Forecast error (fraction)
rho_time = 0.8         # Temporal correlation coefficient

# Sets
R, X = compute_R_X(data_math)                                           # Compute R and X for all loads and buses
L2load = compute_lines_downstream_loads(data_math)                      # Compute downstream loads for each line
L = collect(keys(data_math["load"]))                                    # Set of loads
P_max = Dict(l => rand([5]) ./ s_base for l in L)                         # Maximum power limits for loads
Q_max = Dict(l => rand([2.0]) ./ s_base for l in L)                         # Maximum reactive power limits for loads

#################################################################################
# Generate load scenarios
mu = Array{Float64}(undef, length(L), T)
sigma = Array{Float64}(undef, length(L), T)

for t in 1:T
    mu[:, t] = collect(network.load_profiles[t, :])   # mean load at time t
    sigma[:, t] = mu[:, t] .* forecast_err
end
corr_matrix = fill(forecast_err, length(L), length(L)) + I(length(L)) * (1 - forecast_err)

# Generate all scenarios with temporal correlation
load_scenarios = generate_spatiotemporal_scenarios(mu, sigma, corr_matrix, rho_time, N_scen)

plot(1:T, load_scenarios[1:20, 1, :]', legend=false,
     xlabel="Hour", ylabel="Load (kW)",
     title="Sample Temporal Load Scenarios")
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