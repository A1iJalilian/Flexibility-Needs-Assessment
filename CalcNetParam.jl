##############################################################################
#################      Voltage Equation Matrices      #################
##############################################################################
function find_path_from_substation(node, network_data)
    # Initialize the path as an empty array
    path = []

    # Extract branch data from network_data
    branches = network_data["branch"]

    # We will start from the provided node and trace back to the substation
    current_node = node

    # Keep iterating until we reach the substation
    while true
        found_line = false

        # Iterate through each branch to find a line that connects to current_node via t_bus
        for (line_id, line_data) in branches
            t_bus = line_data["t_bus"]

            # Check if the current node is connected by t_bus
            if t_bus == current_node
                # Add this line to the path
                push!(path, line_id)

                # Update current_node to the f_bus of this line (the node on the other side)
                current_node = line_data["f_bus"]

                # Mark that we found a connecting line
                found_line = true

                # Break the loop as we found the correct line
                break
            end
        end

        # If no connecting line is found, raise an error
        if !found_line
            break
        end
    end

    # Return the path from the original node to the substation
    return path
end

function get_impedance(i, φ₁, j, φ₂, network_data)
    # Find paths from substation to nodes i and j
    path_i = find_path_from_substation(i, network_data)
    path_j = find_path_from_substation(j, network_data)

    # Find common lines between these paths
    common_lines = intersect(Set(path_i), Set(path_j))

    r_total = 0.0
    x_total = 0.0
    for line in common_lines
        r_total += network_data["branch"][line]["br_r"][φ₁, φ₂]
        x_total += network_data["branch"][line]["br_x"][φ₁, φ₂]
    end

    return r_total, x_total
end

function calculate_voltage_sensitivity(i, φ₁, j, φ₂, network_data)
    # Get resistance and reactance for the common path
    r, x = get_impedance(i, φ₁, j, φ₂, network_data)

    φ = deg2rad.([0 -120 120])

    # Calculate R and X using the provided formula
    R = 2 * (cos(φ[φ₁] - φ[φ₂]) * r + sin(φ[φ₁] - φ[φ₂]) * x)
    X = 2 * (-sin(φ[φ₁] - φ[φ₂]) * r + cos(φ[φ₁] - φ[φ₂]) * x)

    return R, X
end

function compute_R_X(data_math)
    L = collect(keys(data_math["load"]))
    R = Dict()  # To store the r_total values
    X = Dict()  # To store the x_total values

    for l1 in L
        bus_l1 = data_math["load"][l1]["load_bus"]
        φ₁ = data_math["load"][l1]["connections"][1]
        for l2 in L
            bus_l2 = data_math["load"][l2]["load_bus"]
            φ₂ = data_math["load"][l2]["connections"][1]
            # Calculate the total resistance and reactance between the two loads
            b_int = bus_l1 isa String ? parse(Int, bus_l1) : bus_l1
            r_total, x_total = calculate_voltage_sensitivity(b_int, φ₁, bus_l2, φ₂, data_math)

            # Store the results in dictionaries
            R[(b_int, φ₁, l2)] = r_total
            X[(b_int, φ₁, l2)] = x_total
        end
    end

    return R, X
end

##########################################################################################
############ line flow modeling parameters
##########################################################################################
function compute_lines_downstream_loads(data_math)
    buses = data_math["bus"]
    branches = data_math["branch"]
    transformers = get(data_math, "transformer", Dict())
    loads = data_math["load"]

    # Step 1: Build undirected graph
    graph = Dict()
    for (_, br) in branches
        a, b = string(br["f_bus"]), string(br["t_bus"])
        push!(get!(graph, a, Set()), b)
        push!(get!(graph, b, Set()), a)
    end
    for (_, tf) in transformers
        a, b = string(tf["f_bus"]), string(tf["t_bus"])
        push!(get!(graph, a, Set()), b)
        push!(get!(graph, b, Set()), a)
    end

    # Step 2: Find slack buses
    slack_buses = [
        string(bus_id)
        for (bus_id, b) in buses
        if b["bus_type"] == 3
    ]
    isempty(slack_buses) && error("No slack bus found.")

    # Step 3: BFS to determine hierarchy
    visited = Set()
    parent_of = Dict()
    children = Dict()
    queue = copy(slack_buses)

    while !isempty(queue)
        bus = popfirst!(queue)
        push!(visited, bus)
        for neighbor in graph[bus]
            if neighbor ∉ visited
                parent_of[neighbor] = bus
                push!(get!(children, bus, []), neighbor)
                push!(queue, neighbor)
                push!(visited, neighbor)
            end
        end
    end

    # Step 4: Downstream bus sets
    all_bus_ids = [string(bid) for bid in keys(buses)]
    down_map = Dict(b => Set([b]) for b in all_bus_ids)
    remainings = Set(all_bus_ids)
    leaves = [b for b in all_bus_ids if !haskey(children, b)]

    iter = 0
    while !isempty(remainings) && iter < length(all_bus_ids)
        next_leaves = []
        for bus in leaves
            if bus ∉ remainings
                continue
            end
            delete!(remainings, bus)
            if haskey(parent_of, bus)
                parent = parent_of[bus]
                union!(down_map[parent], down_map[bus])
                if all(c ∉ remainings for c in get(children, parent, []))
                    push!(next_leaves, parent)
                end
            end
        end
        leaves = next_leaves
        iter += 1
    end

    iter >= length(all_bus_ids) && error("Network may not be radial.")

    # Step 5: Map load buses
    load_bus_map = Dict()
    for (load_id, load_data) in loads
        bus = string(load_data["load_bus"])
        push!(get!(load_bus_map, bus, []), load_id)
    end

    # Step 6: Output dictionary (check connection match)
    result = Dict()

    function process_element(id, a, b, f_conns)
        if haskey(parent_of, b) && parent_of[b] == a
            parent, child = a, b
        elseif haskey(parent_of, a) && parent_of[a] == b
            parent, child = b, a
        else
            return
        end

        buses_down = down_map[child]
        loads_filtered = Dict(conn => [] for conn in f_conns)

        for bus in buses_down
            for load_id in get(load_bus_map, bus, [])
                load_conns = loads[load_id]["connections"]
                for conn in f_conns
                    if conn in load_conns
                        push!(loads_filtered[conn], load_id)
                    end
                end
            end
        end

        for (conn, load_ids) in loads_filtered
            if !isempty(load_ids)
                result[(id, conn)] = load_ids
            end
        end
    end

    for (br_id, br) in branches
        a, b = string(br["f_bus"]), string(br["t_bus"])
        process_element(br_id, a, b, br["f_connections"])
    end
    for (tf_id, tf) in transformers
        a, b = string(tf["f_bus"]), string(tf["t_bus"])
        process_element("tr$tf_id", a, b, tf["f_connections"])
    end

    return result
end

