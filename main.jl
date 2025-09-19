using PowerModelsDistribution
using Plots
using Random, StatsBase
using Serialization
include("PowerNetworkData.jl")
include("CalcNetParam.jl")

# Load the distribution network model  
network = power_network("feeder 2")
eng = network.eng
data_math = transform_data_model(eng)

# Study parameters
V0 = 1                                      # Base voltage at the substation
V0_min = 0.95                               # Minimum voltage
V0_max = 1.05                               # Maximum voltage
s_base = data_math["settings"]["sbase"]     # Base apparent power

# Sets
R, X = compute_R_X(data_math)                                           # Compute R and X for all loads and buses
#test_R_X_computation(R, X, V0, data_math)                               # Test the R and X computation
L2load = compute_lines_downstream_loads(data_math)                      # Compute downstream loads for each line
L = collect(keys(data_math["load"]))                                    # Set of loads

# Calculating the constraints matrices
A_M, A_N, B_M, B_N, C = Voltage_eq_mats(V0, V0_min, V0_max, comm_loads, noncomm_loads, R, X, fixed_load)
D_M_p, D_N_p, D_M_q, D_N_q, E = compute_line_flow_constraints(comm_loads, noncomm_loads, L2load, data_math, ρ)
A_M_p_c, A_N_p_c, B_M_p_c, B_N_p_c, C_p_c = compute_customer_constraints_box(comm_loads, noncomm_loads, P_max, Q_max)
A_M = [A_M; D_M_p; A_M_p_c]
A_N = [A_N; D_N_p; A_N_p_c]
B_M = [B_M; D_M_q; B_M_p_c]
B_N = [B_N; D_N_q; B_N_p_c]
C = [C; E; C_p_c];
A_red, B_red = remove_redundant_constraints(hcat(A_M, A_N, B_M, B_N), C)
A_M, A_N, B_M, B_N = A_red[:, 1:M], A_red[:, M+1:M+N], A_red[:, M+N+1:N+2*M], A_red[:, N+2*M+1:2*M+2*N]
C = B_red