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
#using PyCall # For evaluating Hankels with complex orders
using ArbNumerics
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
    hankel=ArbNumerics.besselj(ArbComplex(nu),ArbComplex(z)) + im*ArbNumerics.bessely(ArbComplex(nu),ArbComplex(z))
    return Complex(BigFloat(real(hankel)), BigFloat(imag(hankel)))
    #return hankel
end

#hankelh1_an(-1im,1im)

# 2. DEFINE SPECTATOR FIELD EQUATIONS # ---------------------------------------------------------

## We first define the mode frequency

function ω(η, k, m, ξ)
    return sqrt(k.^2 .+ a_η.(η).^2 .* (m.^2 + (ξ - 1/6) * R_η.(η)))
end

dω(η, k, m, ξ) = ForwardDiff.derivative(η -> ω(η, k, m, ξ), η)

d2ω(η, k, m, ξ) = ForwardDiff.derivative(η -> dω(η, k, m, ξ), η)

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

function mode_equations!(dχ, χ, η, params)
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
    return ω(η, k, m, ξ) / ω_dS(η, k, m, ξ)
    *(η-η_SR)+η_SR -1/H(t_i)
end

function v_SR(η, k, m, ξ)
    μ = sqrt((m^2 + ξ*R(t_i))/H(t_i)^2 - 2 +0im)
    nu = sqrt(1/4 - μ^2 + 0im)
    A = sqrt(pi/(2*k)*exp(1im*pi*(nu+1/2)))
    return sqrt(-k*τ(η, k, m, ξ)) * A * hankelh1_an(nu,-k*τ(η, k, m, ξ))
end

v_SR(-1,1,1,1)

# Let us compute the derivative of the modes

dv_SR(η, k, m, ξ) = ForwardDiff.derivative(η -> v_SR(η, k, m, ξ), η)

dv_SR(-1,1,1,1)

# Let us compute the wronskian to check the normalization

function wronskian_SR(η, k, m, ξ)
    v = v_SR(η, k, m, ξ)
    dv = dv_SR(η, k, m, ξ)
    return v * conj(dv) - conj(v) * dv
end

wronskian_SR(-1,1,1,1)

# -----------------------------------------------------------
# A. Base Function (Standard Math)
# -----------------------------------------------------------
function my_hankelh1(nu, z)
    # Convert to ArbComplex
    nu_arb = ArbComplex(nu)
    z_arb  = ArbComplex(z)
    
    # Calculation
    res = ArbNumerics.hankelh1(nu_arb, z_arb)
    
    # Return as Complex{BigFloat}
    return Complex(BigFloat(real(res)), BigFloat(imag(res)))
end

# -----------------------------------------------------------
# B. The Custom Rule (Corrected)
# -----------------------------------------------------------
function my_hankelh1(nu, z::Complex{<:ForwardDiff.Dual{T}}) where T
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
function my_hankelh1(nu, z::ForwardDiff.Dual)
    return my_hankelh1(nu, Complex(z))
end

# Test Case
# Define a function f(x) = Real( Hankel(nu, i*x) )
# The argument 'z' is purely imaginary and depends on x
f(x) = real(my_hankelh1(1.5, x * im))

# Calculate derivative at x = 2.5
x_point = 2.5
grad = ForwardDiff.derivative(f, x_point)

println("Success! Gradient: ", grad)

################

# Let us now solve the mode equations with these initial conditions

function solve_mode(k, m, ξ)
    # Initial conditions from SR vacuum
    v0 = v_SR(η_SR, k, m, ξ)
    dv0 = BigFloat(0.0)

    χ0 = [real(v0), real(dv0)]
    χ0_im = [imag(v0), imag(dv0)]

    params = (k, m, ξ)
    η_span = (η_SR, η_f)

    println("Solving mode k=$k, m=$m, ξ=$ξ from η=$η_SR to η=$η_f")
    println("  -> Initial conditions: v0=$(v0), dv0=$(dv0)")

    prob_re = ODEProblem(mode_equations!, χ0, η_span, params)
    prob_im = ODEProblem(mode_equations!, χ0_im, η_span, params)

    sol_re = solve(prob_re, Tsit5(), reltol=1e-9, abstol=1e-9)
    sol_im = solve(prob_im, Tsit5(), reltol=1e-9, abstol=1e-9)

    return sol_re, sol_im
end

solve_mode(1.0, 1.0, 1.0)