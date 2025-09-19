using PowerModelsDistribution
using Plots
using Random, StatsBase
using Serialization
include("PowerNetworkData.jl")
include("CalcNetParam.jl")
include("OptMats.jl")
include("Test_funcs.jl")
include("model.jl")
include("plot_funcs.jl")
include("KPI.jl")

# Load the distribution network model  
network = power_network("feeder 2")
eng = network.eng
Random.seed!(1234)
for l in keys(eng["load"])
    eng["load"][l]["pd_nom"] = rand(0.0:0.1:1.0)   # Set all loads to random values
    eng["load"][l]["qd_nom"] = rand(0.0:0.1:0.3)
end
data_math = transform_data_model(eng)

# Study parameters
V0 = 1                                      # Base voltage at the substation
V0_min = 0.95                               # Minimum voltage
V0_max = 1.05                               # Maximum voltage
s_base = data_math["settings"]["sbase"]     # Base apparent power
ρ = 4                                       # Half the number of edges in polygonal approximation of the line limit
ρ_c = 2                                     # Half the number of edges in polygonal approximation of customer capacity

# Sets
R, X = compute_R_X(data_math)                                           # Compute R and X for all loads and buses
#test_R_X_computation(R, X, V0, data_math)                               # Test the R and X computation
L2load = compute_lines_downstream_loads(data_math)                      # Compute downstream loads for each line
L = collect(keys(data_math["load"]))                                    # Set of loads
N_total = length(L)                                                     # Total number of loads
fixed_load = Dict(l => (data_math["load"][l]["pd"],
    data_math["load"][l]["qd"]) for l in L)         # Fixed loads
P_max = Dict(l => rand([5]) ./ s_base for l in L)                         # Maximum power limits for loads
Q_max = Dict(l => rand([2.0]) ./ s_base for l in L)                         # Maximum reactive power limits for loads
Cap = Dict(l => rand([7.0]) ./ 1 for l in L);                          # Customer capacity limits
σ = 0.0                                                                 # fairness parameter

M = 3                                                                   # Number of communicating loads
N = N_total - M                                                         # Number of non-communicating loads
Random.seed!(9234)  # For reproducibility
comm_loads = sort(sample(L, M; replace=false))                          # Randomly select M communicating loads
noncomm_loads = setdiff(L, comm_loads)                                  # Non-communicating loads


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

#results = optimize_model(A_N, A_M, B_M, B_N, C, noncomm_loads, comm_loads, s_base)
results = optimize_model_fair(A_N, A_M, B_M, B_N, C, noncomm_loads, comm_loads, s_base, Cap, σ)
println(results[:objective_value])
# println(results[:ellipsoid_volume])

A_M_red, b_RHS_red = remove_redundant_constraints(A_M, results[:RHS])
sum([polytope_volume(A_M_red, b_RHS_red; N=1_000_000) for i in 1:100]) / 100

# Show results 
# Plot the network with communicating loads
plt = plot_network(eng, comm_loads, data_math)
#Plots.savefig(plt, "network_with_comm_loads.pdf")


# Plot the bar from P_minus to P_plus
plt = bar(1:N, results[:P_plus_opt], fillto=results[:P_minus_opt],
    xlabel="Load Index", ylabel="Power (kW)", label=false,
    #title="Non-Communicating Loads Power Ranges", label="Power Range",
    fontfamily="Times", size=(400, 200), bar_width=0.5,
    color=:dodgerblue4, lw=0)
# Plots.savefig(plt, "non_communicating_loads_power_ranges.pdf")



println("Problem termination status: \t\t", results[:termination_status])
println("Ellipsoid volume (communicating loads): ", results[:ellipsoid_volume])
println("Hyperrectangle volume (non-communicating loads): ", results[:hyperrectangle_volume])
println("Total volume: ", results[:total_volume])

# Plot the results for communicating loads
bar(1:M, results[:P_comm_max], fillto=results[:P_comm_min], label="Power Range", xlabel="Load Index", ylabel="Power (kW)",
    title="Communicating Loads Power Ranges")

comm_loads_names = [replace(data_math["load"][l]["name"], "load" => "") for l in comm_loads]
if M == 3
    # plt = plot_ellipsoid(results[:W_opt], results[:c_opt], comm_loads)

    # visualize_polyhedron_makie
    plt = plot_3d_polyhedron_and_ellipsoid(A_M_red, b_RHS_red, results[:W_opt], results[:c_opt], comm_loads)
    # save("poly_ellipsoid_3D.html", plt)
    # save("poly_ellipsoid.png", plt, px_per_unit=10)
end
#Plots.savefig(plt, "ellipsoid_communicating_loads.pdf")

# Plot the results for reactive power
bar(1:M, results[:Q_Mp], fillto=results[:Q_Mm], label="Reactive Power (kVAR)",
    xlabel="Communicating Load Index", ylabel="Reactive Power (kVAR)",
    title="Reactive Power for Communicating Loads", color=:green, lc=:green, lw=3)

bar(1:N, results[:Q_Np], fillto=results[:Q_Nm], label="Reactive Power (kVAR)",
    xlabel="Non-Communicating Load Index", ylabel="Reactive Power (kVAR)",
    title="Reactive Power for Non-Communicating Loads", color=:red, lc=:red, lw=3)



#################################################################################
###################### Fairness study
#################################################################################
function Gini(x::Vector{Float64})
    n = length(x)
    x_sorted = sort(x)
    numerator = sum((2*i - n - 1) * x_sorted[i] for i in 1:n)
    denominator = n * sum(x_sorted)
    return numerator / denominator
end

σs = 0:0.1:1.0
Random.seed!(9234)  # For reproducibility
Cap = Dict(l => rand([7.0, 5.0, 3.0]) ./ 1 for l in L);
fairness_results = Dict(σ => optimize_model_fair(A_N, A_M, B_M, B_N, C, noncomm_loads, comm_loads, s_base, Cap, σ) for σ in σs)
# Calculate total volume per case
V_B1 = log(π^(M / 2) / gamma(M / 2 + 1))
for res in values(fairness_results)
    A_fair_red, b_fair_red = remove_redundant_constraints(A_M, res[:RHS])
    V_poly = sum([polytope_volume(A_fair_red, b_fair_red; N=1_000_000) for i in 1:100]) / 100
    res[:total_volume] = V_poly * res[:hyperrectangle_volume]
    res[:V_poly] = V_poly
    ωs = [(prod(Cap[l] for l in comm_loads))^(1/M); [Cap[l] for l in noncomm_loads]]
    V_tot = [V_poly^(1/M); [res[:P_plus_opt][i] - res[:P_minus_opt][i] for i in 1:N]]
    V_sur = [res[:ellipsoid_volume]^(1/M); [res[:P_plus_opt][i] - res[:P_minus_opt][i] for i in 1:N]]
    res[:Gini_tot] = Gini(V_tot ./ ωs)
    res[:Gini_sur] = Gini(V_sur ./ ωs)

    ωs_sum = [(sum(Cap[l] for l in comm_loads)); [Cap[l] for l in noncomm_loads]]
    t_base = [res[:t_obj]+V_B1; res[:t_log]]
    res[:Gini_t] = Gini(t_base ./ ωs_sum)
end
# Plot Gini vs σ
plt = Plots.plot(σs, [fairness_results[σ][:Gini_tot] for σ in σs],
    label=false, xlabel="σ", fontfamily="Times",
    ylabel="Gini idx", size=(300, 150), lw=3, lc=:navy,
    legend=:right)
savefig(plt, "gini_vs_sigma.pdf")

# Plots.plot!(σs, [fairness_results[σ][:Gini_sur] for σ in σs],
#     label="Surrogate volume")

# Plots.plot!(σs, [fairness_results[σ][:Gini_t] for σ in σs],
    # label="Gini Coefficient for Auxiliary")

# Plot total volume vs σ
plt = Plots.plot(σs, [fairness_results[σ][:total_volume] for σ in σs],
    label=false, xlabel="σ", fontfamily="Times",
    yscale=:log10, size=(300, 150), lw=3, lc=:firebrick,
    ylabel="Volume", legend=:right)
savefig(plt, "total_volume_vs_sigma.pdf")

# Plots.plot!(σs, [fairness_results[σ][:ellipsoid_volume] *
#                  fairness_results[σ][:hyperrectangle_volume] for σ in σs],
#     label="Surrogate Volume")

#################################################################################
###################### Voltage adversarial optimization test
#################################################################################
results[:Am], results[:RHS] = A_M_red, b_RHS_red
V_max, V_min = Voltage_adversarial_test(eng, results, noncomm_loads, comm_loads)
# # Plot the maximum and minimum voltage for each bus-phase
bus_phase = keys(V_max)
plt = bar(1:length(bus_phase), [V_max[(b, p)] for (b, p) in bus_phase], fillto=[V_min[(b, p)] for (b, p) in bus_phase],
    label="Voltage Range", xlabel="Load bus", ylabel="Voltage (p.u.)", legend=false,
    #title="Voltage Range per Load Bus",
    bar_width=0.5,
    color=:dodgerblue4, lw=0, fontfamily="Times",
    ylims=(minimum(V0_min) - 0.01, maximum(V0_max) + 0.01),
    size=(400, 200)
)
hline!([V0_min], label="Minimum Voltage", color=:red, linestyle=:dash, lw=2)
hline!([V0_max], label="Maximum Voltage", color=:red, linestyle=:dash, lw=2)

savefig(plt, "voltage_boxplot.pdf")

#################################################################################
###################### Line flow adversarial optimization test
#################################################################################
results[:Am], results[:RHS] = A_M_red, b_RHS_red
line_max = LineFlow_adversarial_test(eng, results, noncomm_loads, comm_loads)

# # Plot the maximum line flow for each line-phase
plt = plot_line_capacity_utilization_adversarial(data_math, line_max)
savefig(plt, "line_capacity_utilization.pdf")
#################################################################################
###################### Voltage adversarial MC test
#################################################################################
# volts, pqs, line_ss = monte_carlo_feasibility_test(eng, results, noncomm_loads, comm_loads; num_trials=20000);

# # Save the voltages and line flows to a file
# Serialization.serialize("voltages.jls", volts)
# Serialization.serialize("line_flows.jls", line_ss)
# Serialization.serialize("pqs.jls", pqs)

# volts = Serialization.deserialize("voltages.jls")
# line_ss = Serialization.deserialize("line_flows.jls")


# plt = plot_voltage_boxplot(volts, V0_min, V0_max, [data_math["load"][l]["load_bus"] for l in keys(data_math["load"])])
# savefig(plt, "voltage_boxplot.pdf")
# #plot_top_line_flows_2D(data_math, line_ss, 10)
# plt = plot_line_capacity_utilization(data_math, line_ss)
# savefig(plt, "line_capacity_utilization.pdf")

################################################################################
###################### Run the coordination evaluation study
#################################################################################
# results_summary = run_coordination_study(L, fixed_load, P_max, Q_max, R, X, V0, V0_min, V0_max, s_base, 10);
add_total_volume!(results_summary)

results_summary = Serialization.deserialize("results_summary.jls")
plt = plot_total_volume_boxplot_matrix(results_summary)
savefig(plt, "total_volume_boxplot_matrix.pdf")

plt = plot_computation_time_boxplot_matrix(results_summary)
savefig(plt, "computation_time_boxplot_matrix.pdf")

plot_objective_function(results_summary)

###############################################################################
# Compute DOE-based power injection ranges
###############################################################################
DOE_based_power_range!(results_summary)
P_inj_max = [getindex.(results_summary[M], :P_inj_max) for M in keys(results_summary)]
plt = plot_power_ranges(results_summary)
savefig(plt, "power_ranges.pdf")

using WGLMakie, Bonito
plt = plot_3d_polyhedron_and_ellipsoid_LScene(A_M_red, b_RHS_red, results[:W_opt], results[:c_opt], comm_loads)
Page(exportable=true, offline=true)
open("figure.html", "w") do io
    show(io, MIME"text/html"(), plt)
end


####################################################################################
# Visualize the polyhedron and its inscribed box for M=3, trial 10
####################################################################################
sample = 10
A_temp, b_temp = remove_redundant_constraints(results_summary[3][sample][:A_M],
    results_summary[3][sample][:RHS])
plt = plot_polyhedron_with_inscribed_box(A_temp, b_temp-0.5*A_temp[:,2])
save("polyhedron_with_inscribed_box.png", plt, px_per_unit=10)




s = 10
res = results_summary[3][s]
plt = bar(1:N, res[:P_plus_opt], fillto=res[:P_minus_opt],
    xlabel="Load Index", ylabel="Power (kW)", label=false,
    #title="Non-Communicating Loads Power Ranges", label="Power Range",
    fontfamily="Times", size=(400, 200), bar_width=0.5,
    color=:dodgerblue4, lw=0)

s = 1
res = results_summary[3][s]
A_temp, b_temp = remove_redundant_constraints(results_summary[3][s][:A_M],
    results_summary[3][s][:RHS])
if M == 3
    # plt = plot_ellipsoid(results[:W_opt], results[:c_opt], comm_loads)

    # visualize_polyhedron_makie
    plt = plot_3d_polyhedron_and_ellipsoid(A_temp, b_temp, res[:W_opt], res[:c_opt], comm_loads)
    # save("poly_ellipsoid_3D.html", plt)
    # save("poly_ellipsoid.png", plt, px_per_unit=10)
end