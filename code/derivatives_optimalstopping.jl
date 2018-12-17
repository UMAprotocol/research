using LinearAlgebra
using Parameters


struct Deriv_OptStop{AV}
    β::Float64
    c::Float64
    d_tilde::Float64
    m_lb::Float64
    mgrid::AV
    ε_probs::Vector{Float64}
end


function Deriv_OptStop(
    ;β=0.99, c=0.01, d_tilde=0.10, m_lb=0.20,
    mgrid=range(0.0, stop=10*m_lb, length=500),
    n_ε=10, prob_decay=4, pos_mult=1.5
    )

    foo = collect(range(0.0, stop=1.0, length=n_ε)).^prob_decay
    ε_probs = vcat(foo, pos_mult*foo[end-1:-1:1])
    ε_probs = ε_probs ./ sum(ε_probs)

    return Deriv_OptStop(β, c, d_tilde, m_lb, mgrid, ε_probs)
end


function solve_optstop(model::Deriv_OptStop)

    @unpack β, c, d_tilde, m_lb = model
    @unpack mgrid, ε_probs = model

    n_m, n_ε = length(mgrid), length(ε_probs)
    Δ_ε = model.mgrid.step.hi

    # Allocate space for 3 vfs and policy
    V = ones(n_m) * -model.d_tilde
    V_upd = copy(V)
    V_a = zeros(n_m, n_ε)
    V_n = zeros(n_m, n_ε)
    adjust = fill(false, n_m)
    Δ_if_adjust = fill(0.0, n_m)

    iter, dist = 0, 1e3
    while (dist > 1e-4) & (iter < 500)
        # Update V
        copyto!(V, V_upd)

        for i_m in findfirst(x -> x >= m_lb, mgrid):n_m
            mtm1 = mgrid[i_m]

            for i_ε in -floor(Int, n_ε/2):1:floor(Int, n_ε/2)
                ε = Δ_ε * i_ε
                is_default_na = mtm1 < m_lb
                i_mtp1_n = clamp(i_m + i_ε, 1, n_m)

                V_n[i_m, i_ε+floor(Int, n_ε/2)+1] = ε + ifelse(is_default_na, -d_tilde, β*V[i_mtp1_n])

                _maxval = -1e8
                _idxm = -1
                for j_m in i_m:n_m
                    mt = mgrid[j_m]
                    Δm = mt - mtm1

                    is_default_a = mt < m_lb
                    i_mtp1_a = clamp(j_m + i_ε, 1, n_m)
                    _val =  ε - Δm - c + ifelse(is_default_a, -d_tilde, β*V[j_m])
                    if _val > _maxval
                        _maxval = _val
                        _idxm = j_m
                    end
                end

                V_a[i_m, i_ε+floor(Int, n_ε/2)+1] = _maxval
                Δ_if_adjust[i_m] = mgrid[_idxm] - mtm1
            end

            EV_a = dot(V_a[i_m, :], ε_probs)
            EV_n = dot(V_n[i_m, :], ε_probs)
            V_upd[i_m] = max(EV_a, EV_n)
            adjust[i_m] = ifelse(EV_a >= EV_n, true, false)
        end

        iter += 1
        dist = maximum(abs, V_upd .- V)
        println("Iteration $iter with distance $dist")
    end

    return V, V_a, V_n, adjust, Δ_if_adjust
end


model = Deriv_OptStop(;c=0.01)
V, V_a, V_n, adjust, Δ_if_adjust = solve_optstop(model)
