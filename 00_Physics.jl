# ==============================================================================
# 00_Physics.jl
# Defines the Universe: Constants and Potentials
# ==============================================================================

# --- 1. UNITS & PARAMETERS ---
# rMP = 2.435*10^18 / (1.0*10^13)
const rMP = 243500.0          
const MP  = rMP * √(8π)       # Full Planck Mass
const ne  = 55.0              # e-folds

# Couplings
const λS = 2.5e-7 / ne^2
const λT = 6.2e-8 / ne^2      # Tanh potential coupling

# --- 2. POTENTIAL DEFINITIONS ---

# Starobinsky Potential (VS)
# Mathematica: VS[ϕ_] := 3/2*λS*rMP^4*(1 - Exp[-Sqrt[2/3]*ϕ/rMP])^2
function V_S(ϕ)
    # Breaking it down for readability
    exponent_term = -√(2/3) * ϕ / rMP
    term = 1 - exp(exponent_term)
    return 1.5 * λS * rMP^4 * term^2
end

# Derivative (dVS)
# Mathematica: dVS[ϕ_] := Sqrt[6]*λS*rMP^3 ...
function dV_S(ϕ)
    exponent_term = -√(2/3) * ϕ / rMP
    exponential = exp(exponent_term)
    term = 1 - exponential
    
    return √6 * λS * rMP^3 * term * exponential
end

# T-Model Potential (VT) - Optional, included since it was in your code
function V_T(ϕ)
    arg = ϕ / (√6 * rMP)
    tanh_val = tanh(arg)
    return λT * rMP^4 * (√6 * tanh_val)^2
end

function dV_T(ϕ)
    arg = ϕ / (√6 * rMP)
    return 2 * √6 * λT * rMP^3 * sinh(arg) / (cosh(arg)^3)
end