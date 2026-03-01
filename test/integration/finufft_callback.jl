using FINUFFT, Reactant, Test, CUDA

const RunningOnCPU = contains(string(Reactant.devices()[1]), "CPU")
const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function nufft1d1_wrapper(
    out::AbstractVector{<:Complex}, x::AbstractVector{<:Real}, c::AbstractVector{<:Complex}
)
    @show typeof(x), typeof(c), typeof(out)
    nufft1d1!(x, c, 1, 1e-9, out)
    return nothing
end

function vanilla_finufft(x, c)
    ms = 200      # output size (number of Fourier modes)
    out = similar(c, Complex{eltype(x)}, ms)
    nufft1d1!(x, c, 1, 1e-9, out)
    return out
end

function traced_finufft(x, c)
    ms = 200      # output size (number of Fourier modes)
    out = Reactant.Ops.julia_callback(
        nufft1d1_wrapper, ((Complex{Reactant.unwrapped_eltype(x)}, ms),), x, c
    )
    return out
end

if RunningOnCPU || RunningOnCUDA
    @testset "FINUFFT callback" begin
        nj = 100
        x = pi * (1.0 .- 2.0 * rand(nj))      # nonuniform points
        c = rand(nj) + 1im * rand(nj)         # their strengths

        x_ra = Reactant.to_rarray(x)
        c_ra = Reactant.to_rarray(c)

        res_ra = @jit traced_finufft(x_ra, c_ra)
        res_vanilla = vanilla_finufft(x, c)

        @test res_ra ≈ res_vanilla
    end
end
