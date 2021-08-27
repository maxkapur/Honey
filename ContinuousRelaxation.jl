using JuMP, Ipopt
using Plots
using Random
using Distributions: Gumbel


# Holds basic information about an admissions market
Economy = NamedTuple{(:Y, :p, :V, :W, :q, :r),
                     Tuple{Matrix{Float64},
                     Vector{Float64},
                     Matrix{Float64},
                     Array{Float64, 3},
                     Vector{Float64},
                     Vector{Float64}}}

"""
    makedata(n=60, m=6)

Generate a random economy with the specified numbers of students and schools.
"""
function makedata(n=6::Int, # Number of student profiles
                  m=6::Int   # Schools
                  )::Economy
    
    # n×m matrix of student utilities
    Y = repeat(randn(m)', n) + rand(Gumbel(), n, m)

    # Density of students of each type
    p = rand(n)
    p ./= sum(p)

    # School preferences

    # n×m matrix of school additive utilities
    V = repeat(3 .+ randn(n)', m)' + rand(Gumbel(), n, m)
    
    # n×n×m tensor, where W[:, :, c] is school c's matrix
    # of complementarity/substitutability effects
    # Entries above diagonal are ignored
    W = randn(n, n, m)

    # Soft capacity constraint: School pays additional
    # fixed cost if demand exceeds
    q = 2 * rand(m) / n

    # Size of fixed cost
    r = randexp(m)

    return (;Y, p, V, W, q, r)
end


"""
    assignment(E, X)

Compute the assignment of students in economy `E` to schools, where `X`
is a matrix of admissions decisions.
"""
function assignment(E::Economy, X::Matrix{Float64})::Matrix{Float64}
    (n, m) = size(E.Y)

    omX = 1 .- X

    idx = zeros(Bool, m)
    freedemand = zeros(Float64, n, m)
    for s in 1:n, c in 1:m
        idx[:] = E.Y[s, :] .> E.Y[s, c]
        freedemand[s, c] = prod(omX[s, idx])
    end

    μ = zeros(Float64, n, m)
    for s in 1:n
        for c in 1:m
            μ[s, c] = X[s, c] * freedemand[s, c]
        end
    end

    return μ
end


"""
    bestresponse(E, X)

Compute each school's optimal admissions policy under the
assumption that other schools' admissions policies are unchanged.
"""
function bestresponse(E::Economy, X::Matrix{Float64})::Matrix{Float64}
    (n, m) = size(E.Y)

    omX = 1 .- X

    idx = zeros(Bool, m)
    freedemand = zeros(Float64, n, m)
    for s in 1:n, c in 1:m
        idx[:] = E.Y[s, :] .> E.Y[s, c]
        freedemand[s, c] = prod(omX[s, idx])
    end

    μ = zeros(Float64, n, m)
    for s in 1:n
        for c in 1:m
            μ[s, c] = X[s, c] * freedemand[s, c]
        end
    end

    X_BR = zeros(Float64, n, m)
    res_μ = zeros(Float64, n)

    for c in 1:m
        # println("  c = $c")
        model = Model(Ipopt.Optimizer)
        set_silent(model)

        # Demand vector
        @variable model (0 ≤ μ_c[1:n])

        for s in 1:n
            # The most demand we can get from students of type s
            # is the number of students who haven't gotten in 
            # somewhere better
            @constraint model (μ_c[s] ≤ freedemand[s, c])
        end

        # Unlike in discrete case, now it is possible to have
        # complementarity between two students of same type.
        @objective model Max (sum(E.V[s, c] * E.p[s] * μ_c[s] for s in 1:n) +
                              sum(E.W[i, j, c] * E.p[i] * μ_c[i] * E.p[j] * μ_c[j] for i in 1:n, j in 1:i))

        optimize!(model)

        # If the optimum here exceeded the capacity, try again
        # with a capacity constraint and see if it does better
        # The same can be accomplished with a binary variable
        if E.p' * value.(μ_c) > E.q[c]
            # println("    Opt exceeded capacity")
            res_μ[:], stash_obj = value.(μ_c), getobjectivevalue(model) - E.r[c]
            @constraint model (E.p' * value.(μ_c) ≤ E.q[c])
            optimize!(model)
            if getobjectivevalue(model) > stash_obj
                # println("    Found a better soln")
                res_μ[:] = value.(μ_c)
            else
                # println("    But it was still optimal")
            end
        else 
            res_μ[:] = value.(μ_c)
        end

        for s in 1:n
            # Admissions policy is the ratio of target 
            # number of students to number of students
            # willing to attend.
            if freedemand[s, c] > eps()
                X_BR[s, c] = min(1, res_μ[s] / freedemand[s, c])
            else
                X_BR[s, c] = X[s, c]
            end
        end
    end

    return X_BR
end



"""
    experiment(T, n, m)

Conduct an experiment to see if the best-response dynamics converge.
Returns the random economy `E`, the admissions policy `X`, the `gap` or
number of students whose assignments switched between iterations,
and the demand vector `D` for each school at each iteration.
"""
function experiment(T=10::Int, n=10::Int, m=8::Int)::NamedTuple
    E = makedata(n, m)

    X = rand(Float64, size(E.Y))
    X_BR = copy(X)

    μ = assignment(E, X)
    μ_BR = copy(μ)

    X_gap = Float64[]
    μ_gap = Float64[]

    D = Vector{Float64}[μ' * E.p]

    for t in 1:T
        @show t

        X_BR[:] = bestresponse(E, X)
        μ_BR[:] = assignment(E, X_BR)

        # Scale this norm by E.p
        push!(X_gap, sum(abs.((X_BR - X)' * E.p)))
        push!(μ_gap, sum(abs.((μ_BR - μ)' * E.p)))

        push!(D, μ_BR' * E.p)
        X[:] = X_BR
        μ[:] = μ_BR
    end

    return (;E, X, μ, X_gap, μ_gap, D)
end


@time res = experiment()


function plots(res)
    pl = plot(xlabel="iteration", yscale=:log10, legend=:bottomleft)
    plot!(pl, res.X_gap, label="number of changed admissions decisions")
    plot!(pl, res.μ_gap, label="number of changed assignments")

    colors = theme_palette(:auto)

    pr = plot(xlabel="iteration", ylabel="demand", legend=false, yscale=:log10)
        
    plot!(pr, reduce(hcat, res.D)',
          c = [colors[i] for i in 1:length(res.E.q)]',
          lw=2)
    for (i, q) in enumerate(res.E.q)
        hline!(pr, [q],
               c = colors[i],
               ls=:dash)
    end

    return pl, pr
end

pl, pr = plots(res)

# savefig(pl, "./continuousplots/convergence.pdf")
# savefig(pr, "./continuousplots/demand.pdf")
