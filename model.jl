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
                        (const_term + δψ[m, t] * flex_constraints[m] -
                         dot(view(A, m, :), P_plus[:, t] + P_minus[:, t]) -
                         dot(view(B, m, :), Q[:, t])) / denom
                        >=
                        τ[t] - s[i, t])
                end
            end
        end

        # --- Objective: flexibility activation cost ---
        @expression(model, total_Pplus[j=1:n], sum(P_plus[j, t] for t in 1:T))
        @expression(model, total_Pminus[j=1:n], sum(P_minus[j, t] for t in 1:T))

        g_cost = sum(λrD[j] * total_Pplus[j] for j = 1:n) * Δt -
                 sum(λrU[j] * total_Pminus[j] for j = 1:n) * Δt

        f_cost = cδψ * sum(δψ[m, t] for m = 1:M, t = 1:T)

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

function model_CVaR_CCG(A, B, C, D_t, flex_constraints,
    hatξ_t, Pmax, Pmin, Qmax, λrD, λrU, cδψ;
    θ=0.1, ε=0.05, Δt=1.0, T=24, p_norm=2,
    solver=Gurobi.Optimizer, tol=1e-7, max_iters=200, initial_active=true, quiet=true)

    M, n = size(A)
    N = size(hatξ_t[1], 2)
    if !quiet
        println("Building DR MILP-CCG model with M=$M, n=$n, N=$N, T=$T, norm=$p_norm")
    end

    # --- Helper: dual norm
    dual_norm(vec::AbstractVector, p) = p == 2 ? norm(vec, 2) :
                                        p == 1 ? norm(vec, Inf) :
                                        p == Inf ? norm(vec, 1) :
                                        error("Unsupported norm: $p. Choose 1, 2, or Inf.")

    # --- Precompute constants for all t
    # D_rows_t[t][m] = row m of D_t[t]
    D_rows_t = [[view(D_t[t], m, :) for m in 1:M] for t in 1:T]
    D_dualnorm_t = [[dual_norm(D_rows_t[t][m], p_norm) for m in 1:M] for t in 1:T]

    # Const_term_t[t][m,i] = dot(D_rows_t[t][m], hatξ_t[t][:, i]) + C[m]
    Const_term_t = [Array{Float64}(undef, M, N) for t in 1:T]
    for t in 1:T, m in 1:M, i in 1:N
        Const_term_t[t][m, i] = dot(D_rows_t[t][m], hatξ_t[t][:, i]) + C[m]
    end

    # --- build JuMP model (master)
    model = JuMP.Model(solver)
    JuMP.set_silent(model)

    # Variables
    @variable(model, P_plus[1:n, 1:T] >= 0)
    @variable(model, P_minus[1:n, 1:T] <= 0)
    @variable(model, Q[1:n, 1:T])
    @variable(model, τ[1:T])
    @variable(model, s[1:N, 1:T] >= 0)
    @variable(model, δψ[1:M, 1:T] >= 0)   # Voltage unbalance violation

    # Bounds constraints
    for t in 1:T
        @constraint(model, P_plus[:, t] .<= Pmax)
        @constraint(model, P_minus[:, t] .>= Pmin)
        @constraint(model, -Qmax .<= Q[:, t])
        @constraint(model, Q[:, t] .<= Qmax)
    end

    # CVaR base constraint per t (ε N τ_t - eᵀ s_{:,t} ≥ θ N)
    for t in 1:T
        @constraint(model, ε * N * τ[t] - sum(s[i, t] for i = 1:N) >= θ * N)
    end

    # Objective: flexibility activation cost + g_cost
    @expression(model, total_Pplus[j=1:n], sum(P_plus[j, t] for t in 1:T))
    @expression(model, total_Pminus[j=1:n], sum(P_minus[j, t] for t in 1:T))

    g_cost = sum(λrD[j] * total_Pplus[j] for j = 1:n) * Δt -
             sum(λrU[j] * total_Pminus[j] for j = 1:n) * Δt
    f_cost = cδψ * sum(δψ[m, t] for m = 1:M, t = 1:T)

    @objective(model, Min, g_cost + f_cost)

    # Prepare helpers for fast dot-products
    A_rows = [view(A, m, :) for m in 1:M]
    B_rows = [view(B, m, :) for m in 1:M]

    # Active constraints bookkeeping: keys are (t,i,m)
    active_constraints = Dict{Tuple{Int,Int,Int},JuMP.ConstraintRef}()

    # Optionally add one initial cut per (t,i)
    if initial_active
        if !quiet
            println("Adding initial heuristic cuts (one per (t,i)) ...")
        end
        for t in 1:T, i in 1:N
            # pick conservative initial m: minimize Const_term/denom over m
            best_m = 1
            best_val = Inf
            for m in 1:M
                denom = D_dualnorm_t[t][m]
                lp = Const_term_t[t][m, i] / denom
                if lp < best_val
                    best_val = lp
                    best_m = m
                end
            end
            denom = D_dualnorm_t[t][best_m]
            expr = (Const_term_t[t][best_m, i] + δψ[best_m, t] * flex_constraints[best_m] -
                    dot(A_rows[best_m], P_plus[:, t] + P_minus[:, t]) -
                    dot(B_rows[best_m], Q[:, t])) / denom
            cref = @constraint(model, expr >= τ[t] - s[i, t])
            active_constraints[(t, i, best_m)] = cref
        end
    end

    # --- CCG main loop ---
    iter = 0
    num_added_total = 0
    while true
        iter += 1
        if iter > max_iters
            @warn "Reached max CCG iterations = $max_iters; stopping."
            break
        end

        optimize!(model)
        stat = termination_status(model)
        if stat ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            @warn "Master solver not optimal (status=$stat). Returning current solution."
            break
        end


        # Pull values
        Pp_val = value.(P_plus)    # n x T array
        Pm_val = value.(P_minus)
        Q_val = value.(Q)
        δψ_val = value.(δψ)
        τ_val = value.(τ)
        s_val = value.(s)

        # Separation: find violated (t,i) by scanning m
        violations = Vector{Tuple{Int,Int,Int,Float64}}()  # (t,i,m,violation_amount)
        for t in 1:T
            # Precompute dot products A*m*(P_plus+P_minus) and B*(Q) for this t
            Psum_t = Pp_val[:, t] + Pm_val[:, t]        # n-vector
            A_dot = A * Psum_t                          # M-vector (A is M x n)
            B_dot = B * Q_val[:, t]                     # M-vector (B is M x n)
            δflex_t = δψ_val[:, t] .* flex_constraints  # M-vector

            for i in 1:N
                rhs = τ_val[t] - s_val[i, t]
                # lhs numerator vector for all m: Const_term_t[t][:,i] + δflex_t - A_dot - B_dot
                lhs_num = Const_term_t[t][:, i] .+ δflex_t .- A_dot .- B_dot
                # divide by denom (handle zeros)
                denom_vec = copy(D_dualnorm_t[t])
                lhs_vec = lhs_num ./ denom_vec
                best_lhs, best_m = findmin(lhs_vec)
                if best_lhs + tol < rhs
                    push!(violations, (t, i, best_m, rhs - best_lhs))
                end
            end
        end

        if isempty(violations)
            if !quiet
                println("CCG converged at iteration $iter; no violated constraints.")
            end
            break
        end

        # Add violated constraints (avoid duplicates)
        num_added = 0
        for (t, i, m, amount) in violations
            key = (t, i, m)
            if !haskey(active_constraints, key)
                denom = D_dualnorm_t[t][m]
                expr = (Const_term_t[t][m, i] + δψ[m, t] * flex_constraints[m] -
                        dot(A_rows[m], P_plus[:, t] + P_minus[:, t]) -
                        dot(B_rows[m], Q[:, t])) / denom
                cref = @constraint(model, expr >= τ[t] - s[i, t])
                active_constraints[key] = cref
                num_added += 1
            end
        end

        num_added_total += num_added
        if !quiet
            println("Iter $iter: found $(length(violations)) violations, added $num_added new constraints (total added = $num_added_total).")
        end

        if num_added == 0
            if !quiet
                println("CCG converged at iteration $iter; no new constraints added.")
            end
            break
        end
    end

    # --- finalize and return ---
    stat = termination_status(model)
    if stat in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        result = Dict(
            :status => stat,
            :objective => objective_value(model),
            :P_plus => value.(P_plus),
            :P_minus => value.(P_minus),
            :Q => value.(Q),
            :tau => value.(τ),
            :s => value.(s),
            :δψ => value.(δψ),
            :build_time => nothing,
            :solve_time => nothing,
            :active_constraints => keys(active_constraints)  # iterator of (t,i,m)
        )
        return result
    else
        return Dict(:status => stat, :active_constraints => keys(active_constraints))
    end
end

