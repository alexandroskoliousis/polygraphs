"""
PolyGraph mathematical operations
"""

from collections import namedtuple
import torch


Evidence = namedtuple("Evidence", ["logits", "values", "trials"])


def _tologits(probabilities):
    """
    Computes logits from probabilities
    """
    epsilon = torch.finfo(probabilities.dtype).eps
    clamped = probabilities.clamp(min=epsilon, max=1 - epsilon)
    return torch.log(clamped) - torch.log1p(-clamped)


def probs(logits, values, trials):
    """
    Computes (normalised) probabilities based on torch.distributions.binomial:

        k log(p) + (n - k) log(1 - p) =
        k (log(p) - log(1 - p)) + n log(1 - p) =
        k logits - n max(logits, 0) - n log(1 + exp(-|logits|))

    Args:
        logits: log(p) - log(1 - p)
        values: Number of positive (or negative) samples observed
        trials: Total number of trials
    """
    norm = trials * logits.clamp(min=0) + trials * torch.log1p(
        torch.exp(-torch.abs(logits))
    )
    logp = values * logits - norm
    #
    # Given k successes and n - k failures out of n trials:
    #
    #   + log(n! / (k! (n - k)!)) =
    #   + log(n!) - log(k!) - log((n - k)!)
    #
    # lgamma(x + 1) = log x!
    logp += torch.lgamma(trials + 1)
    logp -= torch.lgamma(values + 1)
    logp -= torch.lgamma(trials - values + 1)

    return torch.exp(logp)


def likelihood(evidence, hypothesis=True):
    """
    An agent observes evidence that consists of $k$ out of $n$
    successful events with probability $p$.

    Since events are generated independently, we can multiply
    their probabilities.
    """
    result = probs(
        evidence.logits,
        evidence.values if hypothesis else evidence.trials - evidence.values,
        evidence.trials,
    )
    # assert torch.all(torch.ge(result, 0.)) and torch.all(torch.le(result, 1.))
    epsilon = torch.finfo(result.dtype).eps
    clamped = result.clamp(min=epsilon, max=1 - epsilon)
    return clamped


def marginal(prior, evidence):
    """
    Marginal likelihood, P(E) = P(H)P(E|H) + P(-H)P(E|-H)

    Args:
        prior, P(H)
        event likelihood, P(E|H)
    """
    result = prior * likelihood(evidence) + (1.0 - prior) * likelihood(
        evidence, hypothesis=False
    )
    # assert torch.all(torch.ge(result, 0.)) and torch.all(torch.le(result, 1.))
    epsilon = torch.finfo(result.dtype).eps
    clamped = result.clamp(min=epsilon, max=1 - epsilon)
    return clamped


def bayes(prior, evidence, occurred=True):
    """
    Updates prior with Bayes' rule.

    Args:
        prior: P(H)
        evidence: E
        occurred: Whether evidence E occured or not

    Returns:
        Posterior, P(H|E) ~ P(H)P(E|H) / P(E)
    """
    if occurred:
        result = prior * likelihood(evidence) / marginal(prior, evidence)
    else:
        result = (
            prior * (1.0 - likelihood(evidence)) / (1.0 - marginal(prior, evidence))
        )
    return result


def jeffrey(prior, evidence, certainty):
    """
    Updates prior with Jeffrey's rule.

    Args:
        prior: P(H)
        evidence: E
        certainty: Certainty of observed evidence, E
    """
    # Posterior belief, P(H|E), obtained via strict conditionalization on evidence, E
    belief = bayes(prior, evidence)
    # Posterior belief if evidence E did not occur, P(H|~E) = P(H)P(~E|H)/P(~E)
    misbelief = bayes(prior, evidence, occurred=False)
    # Update posterior belief, in light of uncertainty
    return belief * certainty + misbelief * (1.0 - certainty)
