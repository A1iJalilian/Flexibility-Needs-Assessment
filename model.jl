using JuMP, LinearAlgebra, SparseArrays
using HiGHS, Gurobi


function model_CVaR(A, B, C, D_t, flex_constraints,
    hatξ_t, Pmax, Pmin, Qmax, λrD, λrU, cδψ;
    θ=0.1, ε=0.05, Δt=1.0, T=24, p_norm=2,
    solver=Gurobi.Optimizer)

    M, n = size(A)
    N = size(hatξ_t[1], 2)
    println("Building DR model with M=$M, n=$n, N=$N, T=$T, norm=$p_norm")

    # --- Helper: dual norm
    function dual_norm(vec::AbstractVector, p)
        if p == 2
            return norm(vec, 2)
        elseif p == 1
            return norm(vec, Inf)
        elseif p == Inf
            return norm(vec, 1)
        else
            error("Unsupported norm: $p. Choose 1, 2, or Inf.")
        end
    end

    build_time = @elapsed begin
        # --- Precompute constants for all t
        D_rows_t = [[view(D_t[t], m, :) for m in 1:M] for t in 1:T]
        D_dualnorm_t = [[dual_norm(D_rows_t[t][m], p_norm) for m in 1:M] for t in 1:T]

        Const_term_t = [Array{Float64}(undef, M, N) for t in 1:T]
        for t in 1:T, m in 1:M, i in 1:N
            Const_term_t[t][m, i] = dot(D_rows_t[t][m], hatξ_t[t][:, i]) + C[m]
        end

        # -----------------------------
        # JuMP model
        # -----------------------------
        model = JuMP.Model(solver)
        set_silent(model)

        # Variables
        @variable(model, P_plus[1:n, 1:T] >= 0)
        @variable(model, P_minus[1:n, 1:T] <= 0)
        @variable(model, Q[1:n, 1:T])
        @variable(model, τ[1:T])
        @variable(model, s[1:N, 1:T] >= 0)
        @variable(model, δψ[1:M, 1:T] >= 0)   # Voltage unbalance violation

        # Bounds
        for t in 1:T
            @constraint(model, P_plus[:, t] .<= Pmax)
            @constraint(model, P_minus[:, t] .>= Pmin)
            @constraint(model, -Qmax .<= Q[:, t])
            @constraint(model, Q[:, t] .<= Qmax)
        end

        # --- CVaR constraints ---
        for t in 1:T
            # (1) ε N τ_t - eᵀ s_{:,t} ≥ θ N
            @constraint(model, ε * N * τ[t] - sum(s[i, t] for i = 1:N) >= θ * N)

            # (2) For each i,m
            for i in 1:N
                for m in 1:M
                    const_term = Const_term_t[t][m, i]
                    denom = D_dualnorm_t[t][m]
                    if denom == 0.0
                        warn("Denominator in CVaR constraint is zero for t=$t, m=$m.")
                    end
                    @constraint(model,
                        (const_term + δψ[m, t]*flex_constraints[m] -
                         dot(view(A, m, :), P_plus[:, t] + P_minus[:, t]) -
                         dot(view(B, m, :), Q[:, t])) / denom
                        >= τ[t] - s[i, t])
                end
            end
        end

        # --- Objective: flexibility activation cost ---
        @expression(model, total_Pplus[j=1:n], sum(P_plus[j, t] for t in 1:T))
        @expression(model, total_Pminus[j=1:n], sum(P_minus[j, t] for t in 1:T))

        g_cost = sum(λrD[j] * total_Pplus[j] for j=1:n) * Δt -
                 sum(λrU[j] * total_Pminus[j] for j=1:n) * Δt

        f_cost = cδψ * sum(δψ[m, t] for m=1:M, t=1:T)

        @objective(model, Min, g_cost + f_cost)
    end
    println("Model built in $(round(build_time, digits=2)) seconds.")

    solve_time = @elapsed begin
        optimize!(model)
    end
    println("Model solved in $(round(solve_time, digits=2)) seconds.")

    stat = termination_status(model)
    println("Solver status: ", stat)

    if stat in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        return Dict(
            :status => stat,
            :objective => objective_value(model),
            :P_plus => value.(P_plus),
            :P_minus => value.(P_minus),
            :Q => value.(Q),
            :tau => value.(τ),
            :s => value.(s),
            :build_time => build_time,
            :solve_time => solve_time
        )
    else
        return Dict(:status => stat,
                    :build_time => build_time,
                    :solve_time => solve_time)
    end
end
