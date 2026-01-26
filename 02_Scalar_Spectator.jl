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

println("Background solved. Creating fast interpolation tables...")

# 1. Create a dense grid in Float64
η_grid_fast = Float64.(range(η_SR, η_f, length=10000))

# 2. Pre-calculate a and R, converting to Float64 immediately
a_vals = Float64.(real.(a_η.(η_grid_fast)))
R_vals = Float64.(real.(R_η.(η_grid_fast)))

# 3. Create fast interpolators
const a_fast = LinearInterpolation(η_grid_fast, a_vals, extrapolation_bc=Flat())
const R_fast = LinearInterpolation(η_grid_fast, R_vals, extrapolation_bc=Flat())

println("Starting Scalar Spectator Field Production...")

#const mp = pyimport("mpmath")
#mp.dps = 50  # set decimal places

#function hankelh1_py(nu, z)
    # PyCall handles the Complex conversion automatically
#    return mp.hankel1(nu, z)
#end

# Let us define a Hankel frunction from ArbNumerics

setprecision(ArbFloat,64)

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

# Define a FAST frequency for the solver (no broadcasting, pure Float64)

function ω_fast(η, k, m, ξ)
    # Use the lookup table
    a = a_fast(η)
    R = R_fast(η)
    return sqrt(k^2 + a^2 * (m^2 + (ξ - 0.16666666666666666) * R) + 0im)
end

#function ω(η, k, m, ξ)
#    return sqrt(k.^2 .+ a_η.(η).^2 .* (m.^2 + (ξ - 1/6) * R_η.(η)) + 0im) # The frequency can become imaginary!
#end

function ω(η, k, m, ξ)
    return sqrt(k^2 + a_η(η)^2 * (m^2 + (ξ - 1/6) * R_η(η)) + 0im) # The frequency can become imaginary!
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

p_ω = plot(η_vals, real.(ω_vals),
    label=false, xlabel=L"\eta", ylabel=L"\omega(\eta)", lw=2, title="Mode Frequency vs Conformal Time")

display(p_ω)

# Also the derivatives

dω_vals = dω.(η_vals, k_sample, m_sample, ξ_sample)

p_dω = plot(η_vals, real.(dω_vals),
    label=false, xlabel=L"\eta", ylabel=L"\omega'(\eta)", lw=2, title="Mode Frequency Derivative vs Conformal Time")

display(p_dω)

d2ω_vals = d2ω.(η_vals, k_sample, m_sample, ξ_sample)

p_d2ω = plot(η_vals, real.(d2ω_vals),
    label=false, xlabel=L"\eta", ylabel=L"\omega''(\eta)", lw=2, title="Mode Frequency 2nd Derivative vs Conformal Time")

display(p_d2ω)

## Now we define the system of ODEs to solve for each mode

function mode_equations!(dχ, χ, params, η)
    k, m, ξ = params
    ω_val = ω_fast(η, k, m, ξ)

    dχ[1] = χ[2]
    dχ[2] = -ω_val^2* χ[1]
end

# 3. Solve with approximate SR initial conditions # ----------------------------------------------------------------

η_SR = -500

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

# Let us compute the derivative of the modes

dv_SR(η, k, m, ξ) = ForwardDiff.derivative(η -> v_SR(η, k, m, ξ), η)

# Let us compute the wronskian to check the normalization

function wronskian_SR(η, k, m, ξ)
    v = v_SR(η, k, m, ξ)
    dv = dv_SR(η, k, m, ξ)
    return v * conj(dv) - conj(v) * dv
end

wronskian_SR(η_SR,1,1,1)

wronskian_SR(-1,100,1,1)

wronskian_SR(η_SR,1,1e-4,1)

## Let us now solve the mode equations with these initial conditions

function solve_mode(k, m, ξ ;kwargs...)

    v0 = v_SR(η_SR, k, m, ξ)
    dv0 = dv_SR(η_SR, k, m, ξ)

    χ0 = ComplexF64[ComplexF64(v0), ComplexF64(dv0)]

    params = (Float64(k), Float64(m), Float64(ξ))
    η_span = (Float64(η_SR), Float64(η_f))

    #println("Solving mode k=$k, m=$m, ξ=$ξ from η=$η_SR to η=$η_f")
    #println("  -> Initial conditions: v0=$(v0), dv0=$(dv0)")

    prob = ODEProblem(mode_equations!, χ0, η_span, params)

    #sol = solve(prob, Rodas5P(autodiff=false), reltol=1e-10, abstol=1e-10, maxiters=1e7; kwargs...)
    sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-10, maxiters=1e7; kwargs...)

    return sol
end

mode_test=solve_mode(1e-4, 0.5, 1.0)

mode_test.t
v_test = mode_test[1,:]
dv_test = mode_test[2,:]

function wronskian(v,dv)
    return v * conj(dv) - conj(v) * dv
end

wronskian.(v_test, dv_test)

# This is perfect!

# 4. We obtain now the Bogoliubov coefficients # ----------------------------------------------------

## We first define the adiabatic modes

function u_ad(η, k, m, ξ)

    u = 1/sqrt(ω(η, k, m, ξ))

    return u
end

function du_ad(η, k, m, ξ)

    du = -1/sqrt(ω(η, k, m, ξ))*(1im*ω(η, k, m, ξ) + dω(η, k, m, ξ)/(2*ω(η, k, m, ξ)))

    return du
end

function bog_coeff(k, m, ξ)
    
    sol = solve_mode(k, m, ξ; save_everystep=false)

    v_f = sol[1, end]
    dv_f = sol[2, end]

    alpha = ( u_ad(η_f, k, m, ξ)*conj(dv_f) - du_ad(η_f, k, m, ξ)*conj(v_f) ) / (2im)

    beta = ( u_ad(η_f, k, m, ξ)*dv_f - du_ad(η_f, k, m, ξ)*v_f ) / (2im)

    return (alpha, beta)

end

bog_coeff(1,1,1)[2]

function beta2(k, m, ξ)
    
    beta = bog_coeff(k, m, ξ)[2]

    return beta*conj(beta)

end

beta2(1,1e-3,1)

## Let us try to plot a spectrum

a_f = Float64(a_η(η_f))

k_list = 10 .^ range(-4, 5, length=120)/a_f

function k2beta2(m, ξ)

    Sk = zeros(Float64, length(k_list))

    @threads for i in 1:length(k_list)

    #println("i = $i")

    Sk[i] = k_list[i]^2*beta2(k_list[i], m, ξ)

    end

    return Sk

end

Sk_test_1 = k2beta2(1e-4,1/6)

p_Sk_1 = plot(k_list, Sk_test_1,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

Sk_test_2 = k2beta2(1e-3,1/6)

p_Sk_2 = plot!(k_list, Sk_test_2,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

Sk_test_3 = k2beta2(1e-2,1/6)
p_Sk_3 = plot!(k_list, Sk_test_3,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

beta2(1000,1e-3,1)



Sk_test_1 = k2beta2(1e-4,0.51)

p_Sk_1 = plot(k_list, Sk_test_1,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

Sk_test_2 = k2beta2(1e-3,0.51)

p_Sk_2 = plot!(k_list, Sk_test_2,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

Sk_test_3 = k2beta2(1e-2,0.51)
p_Sk_3 = plot!(k_list, Sk_test_3,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")

Sk_test_4 = k2beta2(1e-1,0.51)
p_Sk_4 = plot!(k_list, Sk_test_4,
    scale=:log10,
    label=false, xlabel=L"k", ylabel=L"k^2|\beta_k^2|", lw=2, title="Produced particle spectrum")