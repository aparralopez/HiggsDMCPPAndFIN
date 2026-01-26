# ==============================================================================
# 01_Solve_Background_Modular.jl
# Modular solver with internal time selection based on potential choice
# ==============================================================================

using DifferentialEquations
using Interpolations

# Note: This relies on V_S and V_T being defined in the global scope 
# (i.e., you must include "00_Physics.jl" before including this script).

function solve_background(V_func, dV_func)

    println("--- Starting Background Solver ---")

    # 1. AUTO-DETECT TIME RANGES
    # We check which function V_func is identical to.
    
    if V_func === V_S
        println("   -> Detected Potential: Starobinsky")
        t_i_coeff = -100 * 10^5
        t_f_coeff =  100 * 10^5

        ϕ_i  = BigFloat(1.0877) * MP
        dϕ_i  = BigFloat(0) * MP
        
    elseif V_func === V_T
        println("   -> Detected Potential: T-Model")
        # Example: T-Model might need a wider range or different start
        t_i_coeff = -200 * 10^5 
        t_f_coeff =  200 * 10^5

        ϕ_i  = BigFloat(1.0877) * MP
        dϕ_i  = BigFloat(0) * MP
    
    elseif V_func === V_Q
        println("   -> Detected Potential: Quadratic")
        t_i_coeff = -15 * 10^5
        t_f_coeff =  150 * 10^5

        ϕ_i  = BigFloat(3.0) * MP
        dϕ_i  = BigFloat(0) * MP

    # You can add more potentials here:
    # elseif V_func === V_Quadratic
    #    t_i_coeff = ...
        
    else
        # Default fallback if the function is not recognized
        println("   -> Potential not explicitly recognized. Using default times.")
        t_i_coeff = -100 * 10^5
        t_f_coeff =  100 * 10^5
    end

    # 2. SETUP PARAMETERS (Apply rMP scaling)
    t_i = BigFloat(t_i_coeff) / rMP
    t_f = BigFloat(t_f_coeff) / rMP
    
    # Standard Initial Conditions
    N_i  = BigFloat(0.0)

    u0 = [ϕ_i, dϕ_i, N_i]
    tspan = (t_i, t_f)
    p = (V_func, dV_func)

    println("   -> Time range set: [$(t_i_coeff), $(t_f_coeff)] / rMP")

    # 3. DEFINE SYSTEM (Time Domain)
    function inflation_system!(du, u, p, t)
        V, dV = p
        ϕ = u[1]
        dϕ = u[2]
        N = u[3]
        
        # Friedmann Equation
        H = sqrt(4*pi/(3 * MP^2)*(dϕ^2 + 2*V(ϕ)))
        
        # Equations of Motion
        du[1] = dϕ
        du[2] = -3*H*dϕ - dV(ϕ)
        du[3] = H 
    end

    # 4. SOLVE DYNAMICS
    println("   -> Solving ODE (t)...")
    prob = ODEProblem(inflation_system!, u0, tspan, p)
    sol = solve(prob, Rodas5P(), reltol=1e-12, abstol=1e-12)

    ϕ(t)  = sol(t)[1]
    dϕ(t) = sol(t)[2]
    N(t)  = sol(t)[3]

    # Calculate Scale Factor a(t)
    a_raw_end = exp(N(t_f))
    scaling_factor_a = 175.0 / a_raw_end
    a(t) = exp(N(t)) * scaling_factor_a

    # 5. SOLVE CONFORMAL TIME (η)
    println("   -> Solving Conformal Time η...")

    function eta_t_system!(u, p, t)
        return 1.0/a(t)
    end

    η_in = BigFloat(0.0)
    η_ode = ODEProblem(eta_t_system!, η_in, tspan)
    η_sol = solve(η_ode, Rodas5P(), reltol=1e-14, abstol=1e-14, dtmax=1.0e-2)

    η(t) = η_sol(t) - η_sol(0.0)
    
    η_i = η(t_i)
    η_f = η(t_f)
    
    # 6. INTERPOLATION (Map η -> t)
    println("   -> Generating Interpolations...")
    t_of_eta = LinearInterpolation(η.(η_sol.t), η_sol.t, extrapolation_bc=Line())

    # 7. DEFINE OUTPUT FUNCTIONS
    ϕ_η(η_val)  = ϕ(t_of_eta(η_val))
    dϕ_η(η_val) = dϕ(t_of_eta(η_val))
    a_η(η_val)  = a(t_of_eta(η_val))

    function R_η(η_val)
        val_ϕ = ϕ_η(η_val)
        val_dϕ = dϕ_η(η_val)
        return 8*pi/MP^2 * (4*V_func(val_ϕ) - val_dϕ^2)
    end

    println("   -> Solution Ready.")

    return (
        a = a_η,
        R = R_η,
        
        η_i = η_i,
        η_f = η_f
    )
end