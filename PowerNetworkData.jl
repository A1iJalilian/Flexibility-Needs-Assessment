using PowerModelsDistribution
using Plots
silence!()

struct power_network
    name
    network_data_dir
    orig_eng
    eng

    function power_network(name::String)
        network_data_dir = joinpath(name, "feeder.dss")
        eng = parse_file(network_data_dir)
        reduced_eng = deepcopy(eng)
        reduce_size!(reduced_eng)

        return new(name, network_data_dir, eng, reduced_eng)
    end
end



# =============================================================================
#                              reduce_size  FUNCTION
# =============================================================================
# Function to reduce the size of the network
#   by removing buses and lines that meet specific criteria.
# -----------------------------------------------------------------------------
function reduce_size!(eng)

    # Helper function to find lines connected to a bus
    function find_lines_connected_to_bus(eng, bus_id)
        lines_connected = []
        for (line_id, line_info) in eng["line"]
            if line_info["f_bus"] == bus_id || line_info["t_bus"] == bus_id
                push!(lines_connected, (line_id, line_info))
            end
        end
        return lines_connected
    end

    # Iterate over buses to find and remove the target buses
    buses_to_remove = []
    for (bus_id, bus_info) in eng["bus"]
        # Condition 1: Check if the bus has no load connected to it
        load_connected = any(load_info["bus"] == bus_id for load_info in values(eng["load"]))
        if load_connected
            continue
        end

        # Find lines connected to this bus
        lines_connected = find_lines_connected_to_bus(eng, bus_id)

        # Condition 2: Check if the bus only connects two lines
        if length(lines_connected) != 2
            continue
        end

        # Unpack the connected lines
        line1_id, line1_info = lines_connected[1]
        line2_id, line2_info = lines_connected[2]

        # Condition 3: Check if the "line_code" of the two lines is the same
        if line1_info["linecode"] != line2_info["linecode"]
            continue
        end

        # All conditions met, prepare to remove this bus
        push!(buses_to_remove, bus_id)

        # Update the remaining line's information
        new_length = line1_info["length"] + line2_info["length"]

        if line1_info["f_bus"] == line2_info["f_bus"] || line1_info["t_bus"] == line2_info["t_bus"]
            error("f_bus and t_bus in this network don't match line flow direction.")
        end
    
        if line1_info["f_bus"] == bus_id
            new_t_bus = line1_info["t_bus"]
            new_f_bus = line2_info["f_bus"]
            new_t_connection = line1_info["t_connections"]
            new_f_connection = line2_info["f_connections"]
        else
            new_f_bus = line1_info["f_bus"]
            new_t_bus = line2_info["t_bus"]
            new_f_connection = line1_info["f_connections"]
            new_t_connection = line2_info["t_connections"]
        end

        remaining_line_id = line1_id
        eng["line"][remaining_line_id]["length"] = new_length
        eng["line"][remaining_line_id]["f_bus"] = new_f_bus
        eng["line"][remaining_line_id]["t_bus"] = new_t_bus
        eng["line"][remaining_line_id]["f_connections"] = new_f_connection
        eng["line"][remaining_line_id]["t_connections"] = new_t_connection

        # Remove the other line
        delete!(eng["line"], line2_id)
    end

    # Remove the buses
    for bus_id in buses_to_remove
        delete!(eng["bus"], bus_id)
    end

    return eng
end




# =============================================================================
#                               PLOT_NETWORK FUNCTION
# =============================================================================
#   The following section defines the Plot_Network function, which is used
#   to visualize the power network.
# -----------------------------------------------------------------------------
function plot_network(eng)
    # Extract all bus coordinates
    all_bus_coords = Dict(bus_id => (eng["bus"][bus_id]["lat"], eng["bus"][bus_id]["lon"]) for bus_id in keys(eng["bus"]))

    # Extract load bus coordinates
    load_bus_coords = Dict(bus_id => (eng["bus"][bus_id]["lat"], eng["bus"][bus_id]["lon"]) for (load_id, load) in eng["load"] for bus_id in [load["bus"]])

    # Extract transformer coordinates (use the first bus)
    transformer_coords = Dict(trans_id => (eng["bus"][transformer["bus"][1]]["lat"], eng["bus"][transformer["bus"][1]]["lon"]) for (trans_id, transformer) in eng["transformer"])

    # Initialize the plot
    Plots.plot(
        #background_color = :transparent,  # Make background transparent
        grid = false,  # Remove grid
        framestyle = :none,  # Remove axis labels and titles
        legend = false  # Remove legend
    )

    # Plot lines
    for (line_id, line) in eng["line"]
        f_bus = line["f_bus"]
        t_bus = line["t_bus"]
        f_coord = (eng["bus"][f_bus]["lat"], eng["bus"][f_bus]["lon"])
        t_coord = (eng["bus"][t_bus]["lat"], eng["bus"][t_bus]["lon"])

        Plots.plot!(
            [f_coord[2], t_coord[2]],  # Longitudes for x-axis
            [f_coord[1], t_coord[1]],  # Latitudes for y-axis
            line = (:solid, 3, :blue), 
            legend = false  # Remove legend
        )
    end

    # Plot all buses
    """
    scatter!(
        [coord[2] for coord in values(all_bus_coords)],  # Longitude for x-axis
        [coord[1] for coord in values(all_bus_coords)],  # Latitude for y-axis
        marker = (:circle, 2.5, :black),  # White circles
        legend = false  # Remove legend
    )
    """

    # Plot load buses 
    Plots.scatter!(
        [coord[2] for coord in values(load_bus_coords)],  # Longitude for x-axis
        [coord[1] for coord in values(load_bus_coords)],  # Latitude for y-axis
        marker = (:cross, 3, :red),  # red "+" markers
        legend = false  # Remove legend
    )

    # Plot transformers
    for (trans_id, coord) in transformer_coords
        for size in [8, 7, 5]
            Plots.scatter!([coord[2]], [coord[1]], 
                     marker = (:square, size),
                     markercolor = size == 8 ? :black : size == 7 ? :white : :black,
                     markerstrokewidth = 1)
        end
    end

    Plots.plot!() 
end



# =============================================================================
#                               PLOT_NETWORK_WITH_COMM_LOADS FUNCTION
# =============================================================================#   The following section defines the Plot_Network_With_Comm_Loads function,
function plot_network(eng, comm_loads, data_math)
    # Extract all bus coordinates
    all_bus_coords = Dict(bus_id => (eng["bus"][bus_id]["lat"], eng["bus"][bus_id]["lon"]) for bus_id in keys(eng["bus"]))

    # Extract load bus coordinates
    load_bus_coords = Dict(
        load_id => (eng["bus"][load["bus"]]["lat"], eng["bus"][load["bus"]]["lon"])
        for (load_id, load) in eng["load"]
    )

    # Map comm_loads (data_math names) to eng load IDs
    comm_eng_loads = Set([
        data_math["load"][name]["name"]
        for name in comm_loads
        if haskey(data_math["load"], name) && haskey(data_math["load"][name], "name")
    ])

    # Extract transformer/substation coordinates
    if haskey(eng, "voltage_source")
        substation_bus = eng["voltage_source"]["source"]["bus"]
        substation_coords = (eng["bus"][substation_bus]["lat"], eng["bus"][substation_bus]["lon"])
    else
        substation_coords = (NaN, NaN)
    end

    # Initialize the plot
    Plots.plot(
        grid = false,
        framestyle = :none,
        legend = false,
        size = (500, 300)
    )

    # Plot lines with variable width
    max_line_width = 4
    cm_ub_max = maximum([mean(eng["linecode"][lc]["cm_ub"]) for lc in keys(eng["linecode"])])
    for (_, line) in eng["line"]
        lc = line["linecode"]
        rating = mean(eng["linecode"][lc]["cm_ub"])
        line_width = rating / cm_ub_max * max_line_width

        f_bus, t_bus = line["f_bus"], line["t_bus"]
        f_coord = (eng["bus"][f_bus]["lat"], eng["bus"][f_bus]["lon"])
        t_coord = (eng["bus"][t_bus]["lat"], eng["bus"][t_bus]["lon"])

        Plots.plot!(
            [f_coord[2], t_coord[2]],  # longitudes (x)
            [f_coord[1], t_coord[1]],  # latitudes (y)
            line = (:solid, line_width, :black),
            legend = false
        )
    end

    # Plot all load buses as red dots
    for (load_id, coord) in load_bus_coords
        Plots.scatter!(
            [coord[2]], [coord[1]],
            marker = (:circle, 2, :red),
            markerstrokewidth = 0,
            legend = false
        )
    end

    # Plot communication loads (highlighted with nested circles)
    for load_id in comm_eng_loads
        if haskey(load_bus_coords, load_id)
            coord = load_bus_coords[load_id]
            x, y = coord[2], coord[1]

            # Outer transparent ring
            Plots.scatter!([x], [y],
                marker = (:circle, 9),
                markercolor = :white,
                markerstrokecolor = RGBA(0.49, 0.75, 0.93, 0.8),
                markerstrokewidth = 1.2,
                legend = false
            )

            # Middle ring
            Plots.scatter!([x], [y],
                marker = (:circle, 6.5),
                markercolor = :white,
                markerstrokecolor = RGBA(0.49, 0.75, 0.93, 0.9),
                markerstrokewidth = 1.0,
                legend = false
            )

            # Inner filled dot
            Plots.scatter!([x], [y],
                marker = (:circle, 4.5),
                markercolor = RGBA(0.36, 0.67, 0.93, 1.0),
                markerstrokewidth = 0,
                legend = false
            )
        end
    end

    # Annotate load buses with load IDs (shortened)
    function find_name_in_data_math(load_id, data_math)
        for (name, load) in data_math["load"]
            if load["name"] == load_id
                return name
            end
        end
    end
    for (load_id, coord) in load_bus_coords
        orig_id = find_name_in_data_math(load_id, data_math)
        Plots.annotate!(coord[2], coord[1], Plots.text(orig_id, "times", :grey18, 8))
    end

    # Plot the substation symbol if available
    if !isnan(substation_coords[1])
        lat, lon = substation_coords
        for size in [5, 3.5, 2]
            Plots.scatter!([lon], [lat],
                marker = (:square, size),
                markercolor = size == 5 ? :black : size == 3.5 ? :white : :black,
                markerstrokewidth = 0.5
            )
        end
    end

    Plots.plot!()
end


#########################################################################################
########################   Add generators to load buses function  ########################
#########################################################################################
function add_generators_to_load_buses!(eng, pg_lb=-10, pg_ub=10, qg_lb=-10, qg_ub=10)
    load_ids = keys(eng["load"])
    num_loads = length(load_ids)

    # Ensure inputs are vectors of appropriate length if they are scalars
    pg_lb_vec = typeof(pg_lb) <: Number ? fill(pg_lb, num_loads) : pg_lb
    pg_ub_vec = typeof(pg_ub) <: Number ? fill(pg_ub, num_loads) : pg_ub
    qg_lb_vec = typeof(qg_lb) <: Number ? fill(qg_lb, num_loads) : qg_lb
    qg_ub_vec = typeof(qg_ub) <: Number ? fill(qg_ub, num_loads) : qg_ub

    for (i, load_id) in enumerate(load_ids)
        load = eng["load"][load_id]
        bus_id = load["bus"]
        connections = load["connections"]
        if !isempty(connections)
            gen_id = "gen_$load_id"
            num_connections = length(connections)

            # Fill the parameters for each connection
            pg_lb_filled = fill(pg_lb_vec[i], num_connections)
            pg_ub_filled = fill(pg_ub_vec[i], num_connections)
            qg_lb_filled = fill(qg_lb_vec[i], num_connections)
            qg_ub_filled = fill(qg_ub_vec[i], num_connections)

            # Add generator to the "eng"
            add_generator!(eng, gen_id, bus_id, connections;
                           pg_lb=pg_lb_filled,
                           pg_ub=pg_ub_filled,
                           qg_lb=qg_lb_filled,
                           qg_ub=qg_ub_filled,
                           cost_pg_model=2, cost_pg_parameters=[0.0, 1.0, 0.0])
        end
    end
end


function get_gen_id_for_load(model, load_id)
    # Ensure load_id is Int
    if !(load_id isa Int)
        load_id = parse(Int, string(load_id))
    end

    try
        # Try assuming model is pm
        load = ref(model, 0, :load, load_id)
        load_name = load["name"]
        bus_id = load["load_bus"]
        gen_name = "gen_" * load_name

        for (gen_id, gen) in ref(model, 0, :gen)
            if gen["name"] == gen_name && gen["gen_bus"] == bus_id
                return gen_id
            end
        end
    catch
        # Fallback: assume model is data_math
        if haskey(model, "data_model") && model["data_model"] == MATHEMATICAL
            load = model["load"][string(load_id)]
            load_name = load["name"]
            bus_id = load["load_bus"]
            gen_name = "gen_" * load_name

            for (gen_id, gen) in model["gen"]
                if gen["name"] == gen_name && gen["gen_bus"] == bus_id
                    return gen_id
                end
            end
        else
            error("Unsupported model type")
        end
    end

    error("Generator not found for load $load_id")
end



