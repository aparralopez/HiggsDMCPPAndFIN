# ==============================================================================
# 02_Scalar_Spectator.jl
# Solves production for a scalar field without any couplings other than non-minimal to the curvature
# ==============================================================================

using DifferentialEquations
using Plots
using LaTeXStrings
using Interpolations

using SpecialFunctions
using Base.Threads # To check if threading is on

# 1. SETUP & INCLUDES # --------------------------------------------------------------------------

include("00_Physics.jl")
include("01_Solve_Background.jl")

println("Background solved. Starting Scalar Spectator Field Production...")

# 2. DEFINE SPECTATOR FIELD EQUATIONS # ---------------------------------------------------------

 