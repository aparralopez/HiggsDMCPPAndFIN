# ==============================================================================
# 00_Physics.jl
# Defines the Universe: Constants and Potentials
# ==============================================================================

using LaTeXStrings

# --- 1. UNITS & PARAMETERS --- (from 2303.07359v2)

const rMP = 2.435e18 / 1.0e13 # Reduced Planck Mass in units of 10^13 GeV
#const rMP = 2.435e18         # Reduced Planck Mass in GeV          
const MP  = rMP * √(8π)       # Full Planck Mass
const ne  = 55.0              # e-folds

# Couplings

const λS = 2.5e-7 / ne^2
const λT = 6.2e-8 / ne^2      # Tanh potential coupling

# --- 2. POTENTIAL DEFINITIONS ---

# Starobinsky Potential (VS)

function V_S(ϕ)
    exponent_term = -√(2/3) * ϕ / rMP
    term = 1 - exp(exponent_term)
    return 3/2 * λS * rMP^4 * term^2
end

function dV_S(ϕ)
    exponent_term = -√(2/3) * ϕ / rMP
    exponential = exp(exponent_term)
    term = 1 - exponential
    return √6 * λS * rMP^3 * term * exponential
end

# T-Model Potential (VT)

function V_T(ϕ)
    arg = ϕ / (√6 * rMP)
    tanh_val = tanh(arg)
    return λT * rMP^4 * (√6 * tanh_val)^2
end

function dV_T(ϕ)
    arg = ϕ / (√6 * rMP)
    return 2 * √6 * λT * rMP^3 * sinh(arg) / (cosh(arg)^3)
end