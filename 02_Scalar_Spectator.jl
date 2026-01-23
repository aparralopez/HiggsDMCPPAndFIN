# ==============================================================================
# 02_Scalar_Spectator.jl
# Solves production for a scalar field without any couplings other than non-minimal to the curvature
# ==============================================================================

using DifferentialEquations
using Plots
using LaTeXStrings
using Interpolations
using ForwardDiff
#using SpecialFunctions
using Base.Threads # To check if threading is on
#using PyCall # For evaluating Hankels with complex orders with mpmath
using ArbNumerics # For evaluating Hankels with complex orders natively
#using Conda

# 1. SETUP & INCLUDES # --------------------------------------------------------------------------

include("00_Physics.jl")
include("01_Solve_Background.jl")

#a(η) = a_η(η)
#R(η) = R_η(η)

println("Background solved. Starting Scalar Spectator Field Production...")

#const mp = pyimport("mpmath")
#mp.dps = 50  # set decimal places

#function hankelh1_py(nu, z)
    # PyCall handles the Complex conversion automatically
#    return mp.hankel1(nu, z)
#end

# Let us define a Hankel frunction from ArbNumerics

setprecision(ArbFloat,256)

function hankelh1_an(nu,z)
    nu_an = ArbComplex(nu)
    z_an = ArbComplex(z)
    hankel=ArbNumerics.besselj(nu_an, z_an) + im*ArbNumerics.bessely(nu_an, z_an)
    return Complex(BigFloat(real(hankel)), BigFloat(imag(hankel)))
    #return hankel
end

# We define a custom rule for Dual number to work with ForwardDiff

function hankelh1_an(nu, z::Complex{<:ForwardDiff.Dual{T}}) where T
    # 1. Unwrap the input: Separate Value and Partials
    #    z = x + iy, where x and y are Dual numbers
    z_val_real = ForwardDiff.value(real(z))
    z_val_imag = ForwardDiff.value(imag(z))
    z_val      = Complex(z_val_real, z_val_imag)
    
    #    Get the partial derivatives of the input
    partials_real = ForwardDiff.partials(real(z))
    partials_imag = ForwardDiff.partials(imag(z))

    # 2. Compute Function Value and Derivative (Analytic)
    #    H_val = H(z)
    H_val = hankelh1_an(nu, z_val)
    
    #    slope = H'(z) = H_{nu-1}(z) - (nu/z)*H_nu(z)
    H_prev = hankelh1_an(nu - 1, z_val)
    slope  = H_prev - (nu / z_val) * H_val
    
    # 3. Apply Chain Rule to Real and Imaginary parts separately
    #    Let H(z) = U + iV. Let z = x + iy.
    #    The total derivative is slope * dz.
    #    slope = Sr + i*Si
    #    dz (partials) = P_real + i*P_imag
    
    Sr = real(slope)
    Si = imag(slope)
    
    #    Real Part partials: (Sr * P_real) - (Si * P_imag)
    #    Imag Part partials: (Sr * P_imag) + (Si * P_real)
    new_partials_real = Sr * partials_real - Si * partials_imag
    new_partials_imag = Sr * partials_imag + Si * partials_real

    # 4. Construct the output as Complex{Dual}
    #    We build two separate Dual numbers: one for Real part, one for Imag part
    result_real = ForwardDiff.Dual{T}(real(H_val), new_partials_real)
    result_imag = ForwardDiff.Dual{T}(imag(H_val), new_partials_imag)
    
    return Complex(result_real, result_imag)
end

# Helper: Handle Real Dual inputs (promotes to Complex)
function hankelh1_an(nu, z::ForwardDiff.Dual)
    return hankelh1_an(nu, Complex(z))
end

hankelh1_an(-1im,1im)

ForwardDiff.derivative(z -> hankelh1_an(-1im, z), -1)

# 2. DEFINE SPECTATOR FIELD EQUATIONS # ---------------------------------------------------------

## We first define the mode frequency

function ω(η, k, m, ξ)
    return sqrt(k.^2 .+ a_η.(η).^2 .* (m.^2 + (ξ - 1/6) * R_η.(η)))
end

dω(η, k, m, ξ) = ForwardDiff.derivative(η -> ω(η, k, m, ξ), η)

d2ω(η, k, m, ξ) = ForwardDiff.derivative(η -> dω(η, k, m, ξ), η)

ω(0,1,1,1)
dω(0,1,1,1)

# Let us plot the frequency for a sample mode to check it looks OK

k_sample = 1.0
m_sample = 1.0
ξ_sample = 1.0

η_vals = range(-0.2, η_f, length=1000)
ω_vals = ω.(η_vals,k_sample,m_sample,ξ_sample)

p_ω = plot(η_vals, ω_vals,
    label=false, xlabel=L"\eta", ylabel=L"\omega(\eta)", lw=2, title="Mode Frequency vs Conformal Time")

display(p_ω)

# Also the derivatives

dω_vals = dω.(η_vals, k_sample, m_sample, ξ_sample)

p_dω = plot(η_vals, dω_vals,
    label=false, xlabel=L"\eta", ylabel=L"\omega'(\eta)", lw=2, title="Mode Frequency Derivative vs Conformal Time")

display(p_dω)

d2ω_vals = d2ω.(η_vals, k_sample, m_sample, ξ_sample)

p_d2ω = plot(η_vals, d2ω_vals,
    label=false, xlabel=L"\eta", ylabel=L"\omega''(\eta)", lw=2, title="Mode Frequency 2nd Derivative vs Conformal Time")

display(p_d2ω)

## Now we define the system of ODEs to solve for each mode

function mode_equations!(dχ, χ, params, η)
    k, m, ξ = params
    ω_val = ω(η, k, m, ξ)

    dχ[1] = χ[2]
    dχ[2] = -ω_val^2* χ[1]
end

# 3. Solve with approximate SR initial conditions # ----------------------------------------------------------------

η_SR = -100

function ω_dS(η, k, m, ξ)
    μ2=(m^2 + ξ*R(t_i))/H(t_i)^2 - 2
    return sqrt(k^2 + μ2/η^2 + 0im)
end

function τ(η, k, m, ξ)
    return ω(η, k, m, ξ)/ω_dS(η, k, m, ξ)*(η-η_SR)+η_SR-1/H(t_i)
end

function v_SR(η, k, m, ξ)
    μ = sqrt((m^2 + ξ*R(t_i))/H(t_i)^2 - 2 +0im)
    nu = sqrt(1/4 - μ^2 + 0im)
    A = sqrt(pi/(2*k)*exp(1im*pi*(nu+1/2)))
    return sqrt(-k*τ(η, k, m, ξ)) * A * hankelh1_an(nu,-k*τ(η, k, m, ξ))
end

ω(-1,1,1,1)/ω_dS(-1,1,1,1)

hankelh1_an(-2im,100)

τ(η_SR,1,1,1)

v_SR(η_SR,1,1,1)

# Let us compute the derivative of the modes

dv_SR(η, k, m, ξ) = ForwardDiff.derivative(η -> v_SR(η, k, m, ξ), η)

dv_SR(η_SR,1,1,1)

# Let us compute the wronskian to check the normalization

function wronskian_SR(η, k, m, ξ)
    v = v_SR(η, k, m, ξ)
    dv = dv_SR(η, k, m, ξ)
    return v * conj(dv) - conj(v) * dv
end

wronskian_SR(η_SR,1,1,1)

wronskian_SR(-1,100,1,1)

# Something's wrong here... the wronskian is not constant (and not 2pi * i). The derivative of the Hankel is working, so maybe I defined something wrong in the mode.

################

# Let us now solve the mode equations with these initial conditions
function solve_mode(k, m, ξ)
    # Initial conditions from SR vacuum
    v0 = v_SR(η_SR, k, m, ξ)
    dv0 = dv_SR(η_SR, k, m, ξ)

    χ0 = [v0, dv0]

    params = (k, m, ξ)
    η_span = (η_SR, η_f)

    println("Solving mode k=$k, m=$m, ξ=$ξ from η=$η_SR to η=$η_f")
    println("  -> Initial conditions: v0=$(v0), dv0=$(dv0)")

    prob = ODEProblem(mode_equations!, χ0, η_span, params)

    sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9)

    #v = sol_re[end][1] + im * sol_im[end][1]
    #dv = sol_re[end][2] + im * sol_im[end][2]

    return sol
end

mode_test=solve_mode(1.0, 1.0, 1.0)

mode_test.t
v_test = mode_test[1,:]
dv_test = mode_test[2,:]

function wronskian(v,dv)
    return v * conj(dv) - conj(v) * dv
end

wronskian.(v_test, dv_test)

# This is perfect!

#######

function compute_beta_k(k, m, ξ)
    # 1. Run the solver
    # We pass save_everystep=false because we only need the final value
    # This prevents storing thousands of points in RAM
    params = (k, m, ξ)
    η_span = (η_SR, η_f)
    
    # Re-using your logic (assuming v_SR/dv_SR are defined globally or passed)
    v0 = v_SR(η_SR, k, m, ξ)
    dv0 = dv_SR(η_SR, k, m, ξ)
    χ0 = [v0, dv0]

    prob = ODEProblem(mode_equations!, χ0, η_span, params)
    
    # SOLVE: Note save_everystep=false!
    sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9, save_everystep=false)

    # 2. Extract Final State (at η_f)
    v_final = sol[1, end]
    dv_final = sol[2, end]

    # 3. Compute Beta Squared
    # Define your frequency omega at the final time
    # (assuming standard Minkowski-like vacuum at end)
    ω_f = sqrt(k^2 + m^2) # Or whatever your effective mass is at η_f
    
    # Standard Bogoliubov formula (check your specific normalization!)
    # β = (2ω)^{-1/2} * (i v' + ω v) * phase...
    # |β|^2 = (1/4ω^2) * |v' - iω v|^2 
    # (Note: Formula depends on your Wronskian normalization)
    
    beta_sq = (1 / (4 * ω_f)) * abs2(dv_final - im * ω_f * v_final)

    return beta_sq
end

using Base.Threads

function scan_k_modes(k_array, m, ξ)
    # Pre-allocate result array
    # Use Float64 to save space, unless you really need BigFloat for the output
    results = zeros(Float64, length(k_array)) 

    println("Starting scan on $(nthreads()) threads...")

    # The threaded loop
    @threads for i in 1:length(k_array)
        k = k_array[i]
        
        # Compute and store
        # converting result to Float64 for storage if desired
        results[i] = Float64(compute_beta_k(k, m, ξ))
    end

    return results
end

# 1. Define your range of k (e.g., logarithmic)
k_values = 10 .^ range(-2, 2, length=100)

# 2. Run the scan
spectrum = scan_k_modes(k_values, 1.0, 4.0)

# 3. Plot
using Plots
plot(k_values, spectrum, xscale=:log10, yscale=:log10, 
     ylabel="|β_k|²", xlabel="k", title="Particle Production")

########

function solve_mode(k, m, ξ; kwargs...) 
    # ... (Your existing setup code: v0, dv0, params, etc.) ...
    
    # Initial conditions
    v0 = v_SR(η_SR, k, m, ξ)
    dv0 = dv_SR(η_SR, k, m, ξ)
    χ0 = [v0, dv0]
    
    params = (k, m, ξ)
    η_span = (η_SR, η_f)

    # Note: Removed the print statements so the parallel loop doesn't spam your console
    prob = ODEProblem(mode_equations!, χ0, η_span, params)

    # Pass kwargs... into solve. 
    # This allows the caller to control save_everystep, tolerances, etc.
    sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9; kwargs...)

    return sol
end

function compute_beta_sq(k, m, ξ)
    # 1. Call your solver
    # We pass save_everystep=false to optimize speed/RAM
    sol = solve_mode(k, m, ξ; save_everystep=false)

    # 2. Extract Final State
    # sol[end] gives the vector [v, dv] at the final time step
    v_f  = sol[1, end] 
    dv_f = sol[2, end]

    # 3. Define frequency at final time (Minkowski / Adiabatic vacuum assumption)
    # Ensure this matches your specific model's late-time dispersion relation
    ω_f = sqrt(k^2 + m^2) 

    # 4. Bogoliubov Coefficient Calculation
    # |β|^2 = (1/4ω^2) * |v' - iωv|^2
    # This checks how much "negative frequency" exists in the mode
    term = dv_f - (im * ω_f * v_f)
    beta_sq = (1.0 / (4 * ω_f^2)) * abs2(term) # 4*w^2 or 4*w? check normalization

    # Note: Standard normalization is usually 1/(2w) for quantization, 
    # leading to the prefactor 1/(4w^2) here. Double check your derivation!
    
    return beta_sq
end

using Base.Threads

function get_spectrum(k_array, m, ξ)
    # Pre-allocate output array
    n_k = length(k_array)
    beta_squared = zeros(Float64, n_k)

    println("Scanning $n_k modes on $(nthreads()) threads...")

    @threads for i in 1:n_k
        k = k_array[i]
        
        # Call the physics function
        # converting to Float64 to save space (optional)
        beta_squared[i] = Float64(compute_beta_sq(k, m, ξ))
    end

    return beta_squared
end