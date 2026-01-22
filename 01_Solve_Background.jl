# ==============================================================================
# 01_Solve_Background.jl
# Solves the background dynamics for Starobinsky and T-Model potentials
# ==============================================================================

using DifferentialEquations
using Plots
using LaTeXStrings
using Interpolations

# 256 bits gives you ~77 decimal digits (vs 16 in standard Float64).
# This is enough to see the 10^-28 changes in eta easily.
setprecision(BigFloat, 256)

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

p_V_S = plot(ϕ_check_S ./ rMP, V_check_S, label=false, xlabel=L"\phi", ylabel=L"V(\phi)", lw=3)

p_dV_S = plot(ϕ_check_S ./ rMP, dV_check_S, 
    label=false, xlabel=L"\phi", ylabel=L"V'(\phi)", lw=2, color=:red, ls=:dash)

# Display Input Check (Layout: 2 row, 1 columns)

display(plot(p_V_S, p_dV_S, layout=(2,1), size=(800, 800)))
println("   -> Potential plots displayed.")

## Let us repeat the same for the T-Model potential

# Define range for visual check (-10 to 10 Planck masses)

ϕ_check_T = range(-10*rMP, 10*rMP, length=200)

# Calculate V and V' for T-Model

V_check_T = V_T.(ϕ_check_T)
dV_check_T = dV_T.(ϕ_check_T)

# Create the Input Plot for T-Model

p_V_T = plot(ϕ_check_T ./ rMP, V_check_T, 
    label=false, xlabel=L"\phi", ylabel=L"V(\phi)", lw=3)
p_dV_T = plot(ϕ_check_T ./ rMP, dV_check_T, 
    label=false, xlabel=L"\phi", ylabel=L"V'(\phi)", lw=2, color=:red, ls=:dash)

# Display Input Check for T-Model (Layout: 2 row, 1 columns)

display(plot(p_V_T, p_dV_T, layout=(2,1), size=(800, 800)))
println("   -> T-Model Potential plots displayed.")


# 3. SOLVE THE DYNAMICS (ODE) # ------------------------------------------------------------------------------

function inflation_system!(du, u, p, t)

    # Unpack parameters

    V_func, dV_func = p

    ϕ = u[1]
    dϕ = u[2]
    N = u[3]
    
    # Friedmann Equation

    H = √(4*pi/(3 * MP^2)*(dϕ^2 + 2*V_func(ϕ)))
    
    # Equations of Motion

    du[1] = dϕ
    du[2] = -3*H*dϕ - dV_func(ϕ)
    du[3] = H # dN/dt = H

end

# Initial Conditions

#t_i = -100 * 10^5 / rMP
#t_f =  100 * 10^5 / rMP
#ϕ_i  = 1.0877 * MP
#dϕ_i = 0.0

#N_i = 0.0
#η_i = 0.0

t_i = BigFloat(-100 * 10^5) / rMP
t_f =  BigFloat(100 * 10^5) / rMP
ϕ_i  = BigFloat(1.0877) * MP
dϕ_i = BigFloat(0.0)

N_i = BigFloat(0.0)

u0 = [ϕ_i, dϕ_i, N_i]
tspan = (t_i, t_f)

# Solve

println("2. Solving Starobinsky Dynamics...")

p_S = (V_S, dV_S)
prob_S = ODEProblem(inflation_system!, u0, tspan, p_S)
sol_S = solve(prob_S, Tsit5(), reltol=1e-8, abstol=1e-8)

ϕ_S(t) = sol_S(t)[1]
dϕ_S(t) = sol_S(t)[2]

println("   -> Solution found.")

# T-Model

println("2. Solving T-Model Dynamics...")

ϕ_T(t) = sol_T(t)[1]
dϕ_T(t) = sol_T(t)[2]

p_T = (V_T, dV_T)

prob_T = ODEProblem(inflation_system!, u0, tspan, p_T)

sol_T = solve(prob_T, Rodas5P(), reltol=1e-8, abstol=1e-8)

println("   -> Solution found.")


# 4. OUTPUT CHECK: PLOT THE SOLUTIONS # ------------------------------------------------------------------------------

println("Generating Output Plots...")
#plotly()

# Plot Inflaton and Derivative Evolution

p1 = plot(sol_S.t, ϕ_S.(sol_S.t), 
    label="Starobinsky", xlabel=L"t", ylabel=L"\phi(t)", lw=2)
    plot!(sol_T.t, ϕ_T.(sol_T.t), 
    label="T-Model", ls=:dash, lw=2) # Add T-Model to same plot

p2 = plot(sol_S.t, dϕ_S.(sol_S.t), 
    label="Starobinsky", xlabel=L"t", ylabel=L"\phi^{\prime}(t)", lw=2)
    plot!(sol_T.t, dϕ_T.(sol_T.t), 
    label="T-Model", ls=:dash, lw=2) # Add T-Model to same plot

# Plot Ricci Scalar Evolution

R_S(t) = 8*pi/MP^2*(4*V_S(ϕ_S(t)) - dϕ_S(t)^2)
R_T(t) = 8*pi/MP^2*(4*V_T(ϕ_T(t)) - dϕ_T(t)^2)
p3 = plot(sol_S.t, R_S, 
    label="Starobinsky", xlabel=L"t", ylabel=L"R(t)", lw=2)
    plot!(sol_T.t, R_T,
    label="T-Model", ls=:dash, lw=2)

final_plot = plot(p1, p2, layout=(2,1), size=(800, 800))
display(final_plot)

# Plot Hubble Evolution
# We need to calculate H again for plotting since it's not saved in 'sol'

H_S(t) = sqrt(4*pi/(3*MP^2)*(dϕ_S(t)).^2 .+ 2*V_S.(ϕ_S(t)))
H_T(t) = sqrt(4*pi/(3*MP^2)*(dϕ_T(t)).^2 .+ 2*V_T.(ϕ_T(t))) 

dH_S(t) = -4*pi/MP^2 * dϕ_S(t)^2
dH_T(t) = -4*pi/MP^2 * dϕ_T(t)^2

p4 = plot(sol_S.t, H_S.(sol_S.t), 
    label="Starobinsky", ylabel=L"H(t)", title="Expansion History", lw=2)
plot!(sol_T.t, H_T.(sol_T.t), 
    label="T-Model", ls=:dash, lw=2)

final_plot = plot(p3, p4, layout=(2,1), size=(800, 800))
display(final_plot)

## Let us compute the slow-roll parameters and plot them

println("Generating Slow-Roll Parameter Plots...")

# 1st Slow-Roll Parameter for Starobinsky

ϵ_S(t) = -dH_S(t) / H_S(t)^2

# 1st Slow-Roll Parameter for T-Model

ϵ_T(t) = -dH_T(t) / H_T(t)^2

# Plot Slow-Roll Parameters

p = plot(sol_S.t, ϵ_S.(sol_S.t), 
    label="Starobinsky ϵ", ylabel="Slow-Roll Parameters", title="Slow-Roll Parameters", lw=2)
plot!(sol_T.t, ϵ_T.(sol_T.t),
    label="T-Model ϵ", ls=:dash, lw=2)

final_sr_plot = plot(p, size=(800, 400))

display(final_sr_plot)

# 5. OBTAINING CONFORMAL TIME # ------------------------------------------------------------------------------

println("Solve dynamics again for selected potential...")

p = (V_S, dV_S)

prob = ODEProblem(inflation_system!, u0, tspan, p)

sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

ϕ(t) = sol(t)[1]
dϕ(t) = sol(t)[2]
N(t) = sol(t)[3]

R(t) = 8*pi/MP^2*(4*V_S(ϕ(t)) - dϕ(t)^2)
H(t) = sqrt(4*pi/(3*MP^2)*(dϕ(t))^2 + 2*V_S(ϕ(t)))

times = sol.t

println("   -> Solution found.")

println("Calculating Scale Factor & Conformal Time...")

# Calculate a(t) and t(η)

a_raw_end = exp(N(t_f))
scaling_factor_a = 175.0 / a_raw_end
a(t) = exp(N(t)) * scaling_factor_a

# Let us obtain conformal time

function eta_t_system!(u, p, t)
    return 1.0/a(t)
end

η_in=BigFloat(0.0)

η_ode = ODEProblem(eta_t_system!, η_in, tspan)
η_sol = solve(η_ode, Rodas5P(), reltol=1e-14, abstol=1e-14, dtmax=1.0e-2)
 
η(t) = η_sol(t) - η_sol(0.0)

η_i = η(t_i)
η_f = η(t_f)

plot(η_sol.t, η.(η_sol.t), 
    label=false, xlabel=L"t", ylabel=L"\eta(t)", lw=2)

# We map η -> t using linear interpolation

t_of_eta = LinearInterpolation(η.(η_sol.t), η_sol.t, extrapolation_bc=Line())


# ==============================================================================
# 6. BACKGROUND PLOTS
# ==============================================================================

println("Generating Geometry Checks...")

# 1. Scale Factor vs Time (a(t))
# Should be exponential (straight line on log scale)
p_a_t = plot(times, a.(times), 
    yscale = :log10,
    xlabel = "Time t", ylabel = "log(a)", 
    color = :blue, lw = 2, legend = false)

# 2. Conformal Time vs Time (η(t))
# Should be monotonic increasing
p_eta_t = plot(times, η.(times),
    xlims=:automatic,ylims=:automatic,  
    xlabel = "Time t", ylabel = "Conformal η", 
    color = :purple, lw = 2, legend = false)

p_eta_t_2 = plot(η_sol.t, η.(η_sol.t),
    xlims=:automatic,ylims=:automatic,  
    xlabel = "Time t", ylabel = "Conformal η",  
    color = :purple, lw = 2, legend = false)    

# 3. Inverse Check: t vs t(η)
# We calculate t from eta using our interpolation object

p_t_eta = plot(η.(η_sol.t), t_of_eta.(η.(η_sol.t)), 
    xlabel = "Conformal η", ylabel = "Reconstructed t",
    color = :red, lw = 2, legend = false)

# 4. Scale Factor vs Conformal Time (a(η))

p_a_eta = plot(η.(η_sol.t), a.(η_sol.t), 
    yscale = :log10,
    xlabel = "Conformal η", ylabel = "log(a)",
    color = :green, lw = 2, legend = false)

# Combine into a 2x2 Grid
final_check = plot(p_a_t, p_eta_t, p_t_eta, p_a_eta, layout=(2,2), size=(900, 700))

display(final_check)
println("Plots displayed.")

# ==============================================================================
# 7. Inflaton and Ricci scalar as function of conformal time
# ==============================================================================

println("Generating Inflaton and Ricci Scalar vs Conformal Time...")

# Interpolate ϕ(t) and H(t) to get them as functions of η

ϕ_η(η) =  ϕ(t_of_eta(η))
dϕ_η(η) = dϕ(t_of_eta(η))

R_η(η) = 8*pi/MP^2*(4*V_S(ϕ_η(η)) - dϕ_η(η)^2)

a_η(η) = a(t_of_eta(η))

p_ϕ_η = plot(η.(η_sol.t), ϕ_η.(η.(η_sol.t)),
    xlims=(-1,η_f),ylims=(-1e5,0.7*1e6),   
    label=false, xlabel=L"\eta", ylabel=L"\phi(\eta)", lw=2, title="Inflaton vs Conformal Time")

p_R_η = plot(η.(η_sol.t), R_η.(η.(η_sol.t)),
    xlims=(-1,η_f), ylims=(-2,22),
    label=false, xlabel=L"\eta", ylabel=L"R(\eta)", lw=2, title="Ricci Scalar vs Conformal Time")

p_a_η = plot(η.(η_sol.t), a_η.(η.(η_sol.t)),
    xlims=(-1,η_f), ylims=(-10,200),
    label=false, xlabel=L"\eta", ylabel=L"a(\eta)", lw=2, title="Scale Factor vs Conformal Time")

final_η_plot = plot(p_ϕ_η, p_R_η, p_a_η, layout=(3,1), size=(800, 1200))

display(final_η_plot)

println("Plots displayed.")

println("--- Simulation Completed ---")