using JuMP, MosekTools, LinearAlgebra, SpecialFunctions
using HiGHS
#########################################################################################
#########################   Voltage Equation Matrices for Power Flow Analysis   #########
#########################################################################################
function Voltage_eq_mats(V0, V_min, V_max, loads, R, X)

    # Extract all unique (b, φ) pairs from the keys of R
    bf_pairs = sort!(collect(unique((b, φ) for (b, φ, l) in keys(R))))
    nrows = 2 * length(bf_pairs)

    # Number of loads
    nloads = length(loads)

    # Initialize matrices
    A = zeros(nrows, nloads)
    B = zeros(nrows, nloads)
    C = zeros(nrows)

    # Create a row index map: (b, φ) => row index
    row_idx = Dict{Tuple{Int,Int},Int}()
    for (i, bf) in enumerate(bf_pairs)
        row_idx[bf] = i
    end

    @inbounds for ((b, φ), i) in row_idx
        idx = i
        idx_neg = i + length(bf_pairs)

        # Fill A and B for each load
        for (li, l_id) in enumerate(loads)
            A[idx, li] = R[(b, φ, l_id)]
            A[idx_neg, li] = -R[(b, φ, l_id)]
            B[idx, li] = X[(b, φ, l_id)]
            B[idx_neg, li] = -X[(b, φ, l_id)]
        end

        # Voltage bounds constraints (no fixed load impact)
        C[idx] = V_max^2 - V0^2
        C[idx_neg] = V0^2 - V_min^2
    end

    return A, B, C
end

#########################################################################################
#########################   Line flow limits matrices   #########
#########################################################################################
function compute_line_flow_constraints(loads, L2load, data_math, ρ::Int)
    # Sorted line identifiers
    lf_pairs = sort!(collect(keys(L2load)))
    n_lines = length(lf_pairs)
    nloads = length(loads)

    # Index map
    load_idx = Dict(l => i for (i, l) in enumerate(loads))

    # Temporary storage (vectors of vectors)
    Dp_rows = Vector{Vector{Float64}}()
    Dq_rows = Vector{Vector{Float64}}()
    E_vals = Float64[]

    for r in 0:(2*ρ-1)
        θ = (π * r) / ρ
        α = cos(θ) - sin(θ)
        β = cos(θ) + sin(θ)

        for lf in lf_pairs
            b, ϕ = lf
            ϕ = Int(ϕ)
            downstream = L2load[lf]

            # --- S_max (limit) ---
            if haskey(data_math["branch"], b)
                S_max = data_math["branch"][b]["c_rating_a"][ϕ]
            elseif startswith(b, "tr")
                b_id = b[3:end]
                if haskey(data_math["transformer"], b_id)
                    S_total = data_math["transformer"][b_id]["sm_ub"]
                    nphases = length(data_math["transformer"][b_id]["f_connections"])
                    S_max = S_total / nphases
                else
                    error("Transformer ID $b_id not found in data_math.")
                end
            else
                error("Element $b not found in branch or transformer.")
            end

            # --- Skip if S_max is Inf ---
            if !isfinite(S_max)
                continue
            end

            # Row arrays
            Dp_row = zeros(nloads)
            Dq_row = zeros(nloads)

            for l in downstream
                if haskey(load_idx, l)
                    j = load_idx[l]
                    Dp_row[j] = α
                    Dq_row[j] = β
                end
            end

            # Append row
            push!(Dp_rows, Dp_row)
            push!(Dq_rows, Dq_row)
            push!(E_vals, √2 * S_max)  # no pd/qd contribution
        end
    end

    # Stack into matrices
    Dp = vcat(Dp_rows...)
    Dq = vcat(Dq_rows...)
    E = collect(E_vals)

    # Reshape row-wise into matrices
    n_valid = length(E_vals)
    Dp = reshape(Dp, n_valid, nloads)
    Dq = reshape(Dq, n_valid, nloads)

    return Dp, Dq, E
end


#########################################################################################
#########################   Customer constraints matrices   #########
#########################################################################################
function compute_customer_constraints_box(loads, P_max, Q_max)
    n = length(loads)

    # Identity matrix
    A = Matrix{Float64}(I, n, n)

    # Build block rows: enforce |P| ≤ P_max and |Q| ≤ Q_max
    Dp = vcat( A, -A, zeros(n, n), zeros(n, n) )
    Dq = vcat( zeros(n, n), zeros(n, n), A, -A )

    # RHS vector E, stacked in the same row order
    P_vals = [Float64(P_max[l]) for l in loads]
    Q_vals = [Float64(Q_max[l]) for l in loads]

    E = vcat(P_vals, P_vals, Q_vals, Q_vals)

    return Dp, Dq, E
end


#########################################################################################
#########################   Remove redundant constraints   ##############################
#########################################################################################
function remove_redundant_constraints(A, b; tol=1e-7, optimizer=HiGHS.Optimizer)
    m, n = size(A)
    keep = trues(m)

    # First pass: remove zero rows
    for i in 1:m
        if all(abs.(A[i, :]) .<= tol)
            if b[i] < -0.1
                @warn "Removing infeasible constraint at row $(i): 0 ≤ $(b[i])"
            end
            keep[i] = false
        end
    end

    for i in 1:m
        if !keep[i]
            continue
        end
        # Build model
        model = JuMP.Model(optimizer)
        set_silent(model)

        # Variables (free, unbounded)
        @variable(model, x[1:n])

        # Add all constraints except i
        for j in 1:m
            if j == i || !keep[j]
                continue
            end
            @constraint(model, dot(A[j, :], x) <= b[j] + tol)
        end

        # Maximize a_i^T x
        @objective(model, Max, dot(A[i, :], x))

        optimize!(model)

        status = termination_status(model)
        if status == MOI.OPTIMAL
            val = objective_value(model)
            if val <= b[i] + tol
                keep[i] = false  # redundant
            end
        end
    end

    return A[keep, :], b[keep]
end

#########################################################################################
#########################   Voltage unbalance equation matrices   #######################
#########################################################################################
function Voltage_unbalance_eq_mats(R_3ph, X_3ph, psi::Real, V0)

    # ----------------------------
    # 1. Extract structure
    # ----------------------------
    monitored_points = unique([(b, φ) for (b, φ, _) in keys(R_3ph)])
    sort!(monitored_points)

    loads = unique([l for (_, _, l) in keys(R_3ph)])
    n_v = length(monitored_points)
    n_loads = length(loads)

    # ----------------------------
    # 2. Build R and X matrices
    # ----------------------------
    R_mat = zeros(n_v, n_loads)
    X_mat = zeros(n_v, n_loads)

    for (v_idx, (b, φ)) in enumerate(monitored_points)
        for (l_idx, l) in enumerate(loads)
            R_mat[v_idx, l_idx] = R_3ph[(b, φ, l)]
            X_mat[v_idx, l_idx] = X_3ph[(b, φ, l)]
        end
    end

    # ----------------------------
    # 3. Build F' (deviation operator)
    # ----------------------------
    unique_nodes = unique(first.(monitored_points))
    n_nodes = length(unique_nodes)
    n_phases = Int(length(monitored_points) / n_nodes)
    @assert n_phases == 3 "Expected three phases per node."

    F_block = (1/3) * [
         2  -1  -1
        -1   2  -1
        -1  -1   2
    ]

    F_prime = zeros(n_v, n_v)
    for (i, node) in enumerate(unique_nodes)
        idx = (3*(i-1) + 1):(3*i)
        F_prime[idx, idx] .= F_block
    end

    # Stack upper and lower inequality parts
    F = [F_prime; -F_prime]

    # ----------------------------
    # 4. Construct A, B, C
    # ----------------------------
    A = F * R_mat
    B = F * X_mat

    v0_vec = isa(V0, Number) ? fill(V0, n_v) : V0
    C = psi .* ones(size(F, 1)) - F * v0_vec

    return A, B, C
end

#########################################################################################
#########################   Main ABCD calculation function   ############################
#########################################################################################
function calculateABCD(data_math, settings, P_max, Q_max, pv_capacity)

    V0 = settings.V0
    V0_min = settings.V0_min
    V0_max = settings.V0_max
    T = settings.T
    ρ = settings.ρ
    ψ = settings.ψ
    pf_min = settings.pf_min
    pf_max = settings.pf_max

    R, X = compute_R_X(data_math)                                           # Compute R and X for all loads nodes
    R_3ph, X_3ph = compute_R_X(data_math; three_phase=true, load=false)     # Compute R and X for only three-phase buses
    L2load = compute_lines_downstream_loads(data_math)                      # Compute downstream loads for each line
    L = collect(keys(data_math["load"]))                                    # Set of loads

    A_v, B_v, C_v = Voltage_eq_mats(V0, V0_min, V0_max, L, R, X)
    A_l, B_l, C_l = compute_line_flow_constraints(L, L2load, data_math, ρ)
    A_c, B_c, C_c = compute_customer_constraints_box(L, P_max, Q_max)
    A = [A_v; A_l; A_c]
    B = [B_v; B_l; B_c]
    C = [C_v; C_l; C_c]
    AB_red, C = remove_redundant_constraints(hcat(A, B), C)
    A, B = AB_red[:, 1:length(L)], AB_red[:, length(L)+1:end]
    flex_constraints = zeros(size(A, 1))

    A_u, B_u, C_u = Voltage_unbalance_eq_mats(R_3ph, X_3ph, ψ, V0)
    AB_u_red, C_u_red = remove_redundant_constraints(hcat(A_u, B_u), C_u)
    A_u_red, B_u_red = AB_u_red[:, 1:length(L)], AB_u_red[:, length(L)+1:end]
    flex_constraints = vcat(flex_constraints, ones(size(A_u_red, 1)))

    A = vcat(A, A_u_red)
    B = vcat(B, B_u_red)
    C = vcat(C, C_u_red)

    # --- Compute D matrix with random power factors ---
    γ_pf = rand(length(L)) .* (pf_max - pf_min) .+ pf_min
    γ = sqrt.(1 ./ (γ_pf .^ 2) .- 1)
    Γ_t = [Diagonal(γ .* (1 .+ 0.02 .* randn(length(L)))) for t in 1:T]
    D_t = [A + B * Γ_t[t] for t in 1:T]

    # Compute the PV-related sensitivity column
    d_pv_col = -A * pv_capacity    # M×1

    # Extend D_t for all time steps
    D_t_ext = [hcat(D_t[t], d_pv_col) for t in 1:T]

    return A, B, C, D_t_ext, flex_constraints
end

