using JuMP, Cbc
using Plots
using Random
using Distributions: Gumbel


# Holds basic information about an admissions market
Economy = NamedTuple{(:Y, :V, :W, :q, :r),
                     Tuple{Matrix{Float64},
                     Matrix{Float64},
                     Array{Float64, 3},
                     Vector{Int64},
                     Vector{Float64}}}

"""
    makedata(n=60, m=6)

Generate a random economy with the specified numbers of students and schools.
"""
function makedata(n=60::Int, # Students
                  m=6::Int   # Schools
                  )::Economy
    
    # n×m matrix of student utilities
    Y = repeat(randn(m)', n) + rand(Gumbel(), n, m)

    # School preferences

    # n×m matrix of additive utilities
    V = repeat(randn(m)', n) + rand(Gumbel(), n, m)
    
    # n×n×m tensor, where W[:, :, c] is school c's matrix
    # of complementarity/substitutability effects
    W = 0.1 * randn(n, n, m)

    # Soft capacity constraint: School pays additional
    # fixed cost if demand exceeds
    q = rand(n÷m .+ (-5:5), m)

    # Size of fixed cost
    r = randexp(m) * n / m

    return (;Y, V, W, q, r)
end


"""
    assignment(E, X)

Compute the assignment of students in economy `E` to schools, where `X`
is a matrix of admissions decisions. 
"""
function assignment(E::Economy, X::Matrix{Bool})::Matrix{Bool}
    (n, m) = size(E.Y)

    μ = zeros(Bool, n, m)

    for s in 1:n
        if true in X[s, :]
            μ[s, (1:m)[X[s, :]][argmax(E.Y[s, X[s, :]])]] = true
        end
    end

    return μ
end


"""
    bestresponse(E, X)

Compute each school's optimal admissions policy under the
assumption that other schools' admissions policies are unchanged.
"""
function bestresponse(E::Economy, X::Matrix{Bool})::Matrix{Bool}
    (n, m) = size(E.Y)

    μ = assignment(E, X)
    X_BR = zeros(Bool, n, m)

    for c in 1:m
        model = Model(Cbc.Optimizer)
        set_silent(model)

        # Demand vector
        @variable model μ_c[1:n] Bin

        # Complementarity indicator
        @variable model z[i in 1:n, j in 1:i-1]

        # Complementarity constraint
        for i in 1:n, j in 1:i-1
            @constraint model (z[i, j] ≥ μ_c[i] - μ_c[j])
            @constraint model (z[i, j] ≥ - μ_c[i] + μ_c[j])
            @constraint model (z[i, j] ≤ μ_c[i] + μ_c[j])
            @constraint model (z[i, j] ≤ 2 - μ_c[i] - μ_c[j])
        end

        # Exceeded capacity
        @variable model g Bin
        @constraint model (!g => {sum(μ_c) ≤ E.q[c]})

        for s in 1:n
            # If s is already going to a school she likes better
            # then changing our decision makes no difference.
            if E.Y[s, c] < E.Y[s, :]' * μ[s, :]
                @constraint model μ_c[s] == μ[s, c]
            end

            # If not, then we know that our decision matches hers.
            # This eliminates the variable X.
        end

        @objective model Max (E.V[:, c]' * μ_c +
                              sum(E.W[i, j, c] * z[i, j] for i in 1:n, j in 1:i-1) - 
                              g * E.r[c])

        optimize!(model)

        for s in 1:n
            if E.Y[s, c] < E.Y[s, :]' * μ[s, :]
                X_BR[s, c] = X[s, c]
            else
                X_BR[s, c] = value(μ_c[s])
            end

            # Or use this one line to simulate yield protection
            # Schools reject students who rejected them
            # X_BR[s, c] = value(μ_c[s])
        end
    end

    return X_BR
end


"""
    experiment(T, n, m)

Conduct an experiment to see if the best-response dynamics converge.
Returns the random economy `E`, the `gap` or number of students whose
assignments switched between iterations, and the demand vector `D` for
each school at each iteration.
"""
function experiment(T=5::Int, n=80::Int, m=8::Int)
    E = makedata(n, m)
    X = rand(Bool, size(E.Y))
    X_BR = copy(X)
    gap = Int[]
    D = Vector{Int}[]

    for t in 1:T
        @show t
        X_BR[:] = bestresponse(E, X)
        push!(gap, sum(abs.(X_BR - X)))
        println("BR gap: ", gap[end])
        push!(D, reshape(sum(assignment(E, X_BR), dims=1), :))
        X[:] = X_BR
    end

    return E, gap, D
end


E, gap, D = experiment()


function plots()
    pl = plot(gap, xlabel="iteration", label="number of students whose assignment changed")

    colors = theme_palette(:auto)
    pr = plot(reduce(hcat, D)', c = [colors[i] for i in 1:length(E.q)]', xlabel="iteration", ylabel="demand", legend=false)
    for (i, q) in enumerate(E.q)
        hline!(pr, [q], c = colors[i], ls=:dash)
    end

    return pl, pr
end

pl, pr = plots()

savefig(pl, "convergence.pdf")
savefig(pr, "demand.pdf")
