using QuantEcon


struct PortfolioChoiceProblem{AU<:AbstractUtility}
    # Basic parameters
    r::Float64  # Risk-free interest rate
    π::Float64  # The probability of making a mistake for voter
    u::AU

    # Oracle choice parameters
    d::Float64  # Dividend payment
    γ::Vector{Float64}  # For now make this 2x2... x is individual choice and X is Oracle output
end


"""
Computes wealth given the bonds and coins that an individual holds and the

Parameters
---------
b : Float64
    Number of bonds an individual holds
c : Float64
    Number of coins an individual holds. There is a total supply of 1.0 coins.
x : Int
    Individual's vote
X : Int
    Oracle's reported state

"""
function new_wealth(p::PortfolioChoiceProblem, b, c, x, X)
    risk_free_wealth = (1.0 + p.r) * b
    risky_wealth = c * (1 + p.γ[x, X]) * p.d

    return risk_free_wealth + risky_wealth
end


function agent_value(p::PortfolioChoiceProblem, b, c, X)
    return p.π*p.u(new_wealth(p, b, c, 1, 2)) + (1.0 - p.π)*p.u(new_wealth(p, b, c, 2, 2))
end


function find_price
