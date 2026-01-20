# ==============================================================================
# 01_Solve_Background.jl
# Solves the background dynamics and visualizes both inputs (V) and outputs (phi)
# ==============================================================================

using DifferentialEquations
using Plots
using LaTeXStrings
using Interpolations

# 1. SETUP & INCLUDES # ------------------------------------------------------------------------------
# Make plots interactive (zoom/pan enabled)
#plotly() 

# Load your physics definitions

include("00_Physics.jl") 

println("--- Starting Simulation ---")

# 2. INPUT CHECK: PLOT THE POTENTIAL # ------------------------------------------------------------------------------
println("1. Generating Potential Check...")

# Define range for visual check (-1 to 10 Planck masses)

ϕ_check_S = range(-1*rMP, 10*rMP, length=200)

# Calculate V and V'

V_check_S = V_S.(ϕ_check_S)
dV_check_S = dV_S.(ϕ_check_S)

# Create the Input Plot

p_pot_S = plot(ϕ_check_S ./ rMP, V_check_S, 
    label=L"V(\phi)", ylabel="Potential", title="Input: Potential", lw=3)
p_slope_S = plot(ϕ_check_S ./ rMP, dV_check_S, 
    label=L"V'(\phi)", ylabel="Slope", title="Input: Slope", lw=2, color=:red, ls=:dash)

# Display Input Check (Layout: 2 row, 1 columns)

display(plot(p_pot_S, p_slope_S, layout=(2,1), size=(800, 800)))
println("   -> Potential plots displayed.")

## Let us repeat the same for the T-Model potential

# Define range for visual check (-10 to 10 Planck masses)

ϕ_check_T = range(-10*rMP, 10*rMP, length=200)

# Calculate V and V' for T-Model

V_check_T = V_T.(ϕ_check_T)
dV_check_T = dV_T.(ϕ_check_T)

# Create the Input Plot for T-Model

p_pot_T = plot(ϕ_check_T ./ rMP, V_check_T, 
    label=L"V_{T}(\phi)", ylabel="Potential", title="Input: T-Model Potential", lw=3)
p_slope_T = plot(ϕ_check_T ./ rMP, dV_check_T, 
    label=L"V'_{T}(\phi)", ylabel="Slope", title="Input: T-Model Slope", lw=2, color=:red, ls=:dash)

# Display Input Check for T-Model (Layout: 2 row, 1 columns)

display(plot(p_pot_T, p_slope_T, layout=(2,1), size=(800, 800)))
println("   -> T-Model Potential plots displayed.")


# 3. SOLVE THE DYNAMICS (ODE) # ------------------------------------------------------------------------------

function inflation_system!(du, u, p, t)

    # Unpack parameters

    V_func, dV_func = p

    ϕ = u[1]
    dϕ = u[2]
    N = u[3]
    η = u[4]
    
    # Friedmann Equation

    H = √(4*pi/(3 * MP^2)*(dϕ^2 + 2*V_func(ϕ)))
    
    # Equations of Motion

    du[1] = dϕ
    du[2] = -3*H*dϕ - dV_func(ϕ)
    du[3] = H # dN/dt = H
    du[4] = exp(-N)

end

# Initial Conditions

t_i = -100 * 10^5 / rMP
t_f =  100 * 10^5 / rMP
ϕ_i  = 1.0877 * MP
dϕ_i = 0.0

N_i = 0.0
η_i = 0.0

u0 = [ϕ_i, dϕ_i, N_i, η_i]
tspan = (t_i, t_f)

# Solve

println("2. Solving Starobinsky Dynamics...")

p_S = (V_S, dV_S)

prob_S = ODEProblem(inflation_system!, u0, tspan, p_S)

sol_S = solve(prob_S, Tsit5(), reltol=1e-8, abstol=1e-8)

println("   -> Solution found.")

# T-Model

println("2. Solving T-Model Dynamics...")

p_T = (V_T, dV_T)

prob_T = ODEProblem(inflation_system!, u0, tspan, p_T)

sol_T = solve(prob_T, Tsit5(), reltol=1e-8, abstol=1e-8)

println("   -> Solution found.")


# 4. OUTPUT CHECK: PLOT THE SOLUTIONS # ------------------------------------------------------------------------------

println("Generating Output Plots...")
#plotly()

# Plot Field Evolution

p1 = plot(sol_S.t, sol_S[1,:], 
    label="Starobinsky", ylabel=L"\phi / M_{Pl}", title="Field Evolution", lw=2)
plot!(sol_T.t, sol_T[1,:], 
    label="T-Model", ls=:dash, lw=2) # Add T-Model to same plot

# Plot Hubble Evolution
# We need to calculate H again for plotting since it's not saved in 'sol'

H_S = [sqrt((0.5*sol_S(t)[2]^2 + V_S(sol_S(t)[1]))/(3*MP^2)) for t in sol_S.t]
H_T = [sqrt((0.5*sol_T(t)[2]^2 + V_T(sol_T(t)[1]))/(3*MP^2)) for t in sol_T.t]

p2 = plot(sol_S.t, H_S, 
    label="Starobinsky", ylabel=L"H(t)", title="Expansion History", lw=2)
plot!(sol_T.t, H_T, 
    label="T-Model", ls=:dash, lw=2)

final_plot = plot(p1, p2, layout=(2,1), size=(800, 800))
display(final_plot)

## Let us compute the slow-roll parameters and plot them

println("Generating Slow-Roll Parameter Plots...")

# Slow-Roll Parameters for Starobinsky

ϵ_S = [ (MP^2 / (4*pi)) * (0.5 * (sol_S(t)[2]^2) / (0.5 * sol_S(t)[2]^2 + V_S(sol_S(t)[1]))) for t in sol_S.t]

η_S = [ (MP^2 / (4*pi)) * ( (sol_S(t)[2]^2 - sol_S(t)[1] * dV_S(sol_S(t)[1])) / (0.5 * sol_S(t)[2]^2 + V_S(sol_S(t)[1])) ) for t in sol_S.t]

# Slow-Roll Parameters for T-Model

ϵ_T = [ (MP^2 / (4*pi)) * (0.5 * (sol_T(t)[2]^2) / (0.5 * sol_T(t)[2]^2 + V_T(sol_T(t)[1]))) for t in sol_T.t]

η_T = [ (MP^2 / (4*pi)) * ( (sol_T(t)[2]^2 - sol_T(t)[1] * dV_T(sol_T(t)[1])) / (0.5 * sol_T(t)[2]^2 + V_T(sol_T(t)[1])) ) for t in sol_T.t]

# Plot Slow-Roll Parameters

p3 = plot(sol_S.t, ϵ_S, 
    label="Starobinsky ϵ", ylabel="Slow-Roll Parameters", title="Slow-Roll Parameters", lw=2)

plot!(sol_S.t, η_S,
    label="Starobinsky η", lw=2)

plot!(sol_T.t, ϵ_T,
    label="T-Model ϵ", ls=:dash, lw=2)

plot!(sol_T.t, η_T,
    label="T-Model η", ls=:dash, lw=2)

final_sr_plot = plot(p3, size=(800, 400))

display(final_sr_plot)

# 5. OBTAINING CONFORMAL TIME # ------------------------------------------------------------------------------

println("Solve dynamics again for selected potential...")

p = (V_S, dV_S)

prob = ODEProblem(inflation_system!, u0, tspan, p)

sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12,dtmax=1e4)

println("   -> Solution found.")

println("Calculating Scale Factor & Conformal Time...")

times = sol.t
N_vals = sol[3, :] # This is ln(a_raw)
η_raw_vals = sol[4, :]

# Calculate Scaling

a_raw_end = exp(N_vals[end])
scaling_factor_a = 175.0 / a_raw_end
a_vals = exp.(N_vals) .* scaling_factor_a # This is the final a(t)

# Construct η array (with η=0 at t=0)

state_at_zero = sol(0.0) 
eta_offset = 0

η_vals = η_raw_vals .- eta_offset



println("Adaptive integration complete. Offset applied: ", eta_offset)

# We map η -> t using linear interpolation

t_of_eta = linear_interpolation(η_vals, times)

println("   -> Conformal time range: [$(η_vals[1]), $(η_vals[end])]")

# ==============================================================================
# 6. BACKGROUND PLOTS
# ==============================================================================

println("Generating Geometry Checks...")

# 1. Scale Factor vs Time (a(t))
# Should be exponential (straight line on log scale)
p_a_t = plot(times, a_vals, 
    yscale = :log10,
    xlabel = "Time t", ylabel = "log(a)", 
    title = "Scale Factor a(t)", 
    color = :blue, lw = 2, legend = false)

# 2. Conformal Time vs Time (η(t))
# Should be monotonic increasing
p_eta_t = plot(times, η_vals,
    xlims=(-39, .20),ylims=(0.6, 0.65),  
    xlabel = "Time t", ylabel = "Conformal η", 
    title = "Conformal Time η(t)", 
    color = :purple, lw = 2, legend = false)

# 3. Inverse Check: t vs t(η)
# We calculate t from eta using our interpolation object
η_test_range = range(η_vals[1], η_vals[end], length=200)
t_reconstructed = t_of_eta.(η_test_range) # The dot means "apply to every element"

p_inv = plot(η_test_range, t_reconstructed, 
    xlabel = "Conformal η", ylabel = "Reconstructed t",
    title = "Inverse Check t(η)", 
    color = :red, lw = 2, legend = false)

# 4. Scale Factor vs Conformal Time (a(η))
# We map η -> t -> a(t) to check the full chain
# This is crucial for the Mukhanov-Sasaki equations later
t_mapped = t_of_eta.(η_test_range)
# We need to interpolate a(t) as well to plot it against these new times
# Or simpler: just plot existing η_vals vs a_vals (since they correspond 1-to-1)
p_a_eta = plot(η_vals, a_vals, 
    yscale = :log10,
    xlabel = "Conformal η", ylabel = "log(a)",
    title = "Scale Factor a(η)", 
    color = :green, lw = 2, legend = false)

# Combine into a 2x2 Grid
final_check = plot(p_a_t, p_eta_t, p_inv, p_a_eta, layout=(2,2), size=(900, 700))

display(final_check)
println("Plots displayed.")

## Let us now plot closer to the end of inflation

println("Generating End-of-Inflation Checks...")

# Define a zoomed-in range near the end of inflation

t_zoom_start = 0
t_zoom_end = times[end]
zoom_indices = findall(t -> t >= t_zoom_start && t <= t_zoom_end, times)

times_zoom = times[zoom_indices]
a_vals_zoom = a_vals[zoom_indices]
η_vals_zoom = η_vals[zoom_indices]
N_vals_zoom = N_vals[zoom_indices]

# 1. Scale Factor vs Time (a(t)) - Zoomed

p_a_t_zoom = plot(times_zoom, a_vals_zoom, 
    yscale = :log10,
    xlabel = "Time t", ylabel = "log(a)", 
    title = "Scale Factor a(t) - End of Inflation", 
    color = :blue, lw = 2, legend = false)

# 2. Conformal Time vs Time (η(t)) - Zoomed

p_eta_t_zoom = plot(times_zoom, η_vals_zoom, 
    xlabel = "Time t", ylabel = "Conformal η", 
    title = "Conformal Time η(t) - End of Inflation", 
    color = :purple, lw = 2, legend = false)

# 3. Number of e-Folds vs Time (N(t)) - Zoomed

p_N_t_zoom = plot(times_zoom, N_vals_zoom, 
    xlabel = "Time t", ylabel = "N(t)", 
    title = "Number of e-Folds N(t) - End of Inflation", 
    color = :orange, lw = 2, legend = false)

# Combine into a 2x2 Grid

final_zoom_check = plot(p_a_t_zoom, p_eta_t_zoom, p_N_t_zoom, layout=(2,2), size=(900, 700))
display(final_zoom_check)
println("Zoomed plots displayed.")

# ==============================================================================
# 7. Inflaton and Ricci scalar as function of conformal time
# ==============================================================================

println("Generating Inflaton and Ricci Scalar vs Conformal Time...")

# Interpolate ϕ(t) and H(t) to get them as functions of η

ϕ_of_t = linear_interpolation(times, sol[1, :])

H_of_t = linear_interpolation(times, [sqrt((0.5*sol(t)[2]^2 + V_S(sol(t)[1]))/(3*MP^2)) for t in times])

# Map η -> t -> ϕ(t) and H(t)

ϕ_vals_eta = ϕ_of_t.(t_of_eta.(η_vals))

H_vals_eta = H_of_t.(t_of_eta.(η_vals))

# Calculate Ricci Scalar R(η) = 6(2H^2 + dH/dt)

dH_dt_vals = similar(H_vals_eta)

for i in 2:length(H_vals_eta)-1
    dt_local = t_of_eta(η_vals[i+1]) - t_of_eta(η_vals[i-1])
    dH_dt_vals[i] = (H_vals_eta[i+1] - H_vals_eta[i-1]) / dt_local
end

# Forward and backward difference for the edges

dH_dt_vals[1] = (H_vals_eta[2] - H_vals_eta[1]) / (t_of_eta(η_vals[2]) - t_of_eta(η_vals[1]))

dH_dt_vals[end] = (H_vals_eta[end] - H_vals_eta[end-1]) / (t_of_eta(η_vals[end]) - t_of_eta(η_vals[end-1]))

R_vals_eta = 6.0 * (2.0 .* H_vals_eta .^2 .+ dH_dt_vals)

# Plot ϕ(η)

p_phi_eta = plot(η_vals, ϕ_vals_eta ./ rMP, 
    xlims=(η_vals[end-1850], η_vals[end]),ylims=(-0.5, 5.2),
    xlabel = "Conformal η", ylabel = L"\phi / M_{Pl}", 
    title = "Inflaton Field vs Conformal Time", 
    color = :blue, lw = 2, legend = false)

# Plot R(η)

p_R_eta = plot(η_vals, R_vals_eta, 
    xlims=(η_vals[end-1810], η_vals[end]),ylims=(-0.7, 1.2),
    xlabel = "Conformal η", ylabel = "Ricci Scalar R", 
    title = "Ricci Scalar vs Conformal Time", 
    color = :red, lw = 2, legend = false)

# Combine into a 2-row layout

final_phi_R_plot = plot(p_phi_eta, p_R_eta, layout=(2,1), size=(800, 700))

display(final_phi_R_plot)

println("Inflaton and Ricci Scalar plots displayed.")

println("--- Simulation Completed ---")