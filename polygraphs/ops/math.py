"""
PolyGraph mathematical operations
"""

from collections import namedtuple
import torch


Evidence = namedtuple("Evidence", ["p", "k", "n"])


def _logprob(p, k, n):  # pylint: disable=invalid-name
    """
    Log unnormalised probability
    """
    return k * torch.log(p) + (n - k) * torch.log1p(-p)


def likelihood(evidence, hypothesis=True):
    """
    An agent observes evidence that consists of $k$ out of $n$
    successful events with probability $p$.

    Since events are generated independently, we can multiply
    their probabilities.
    """
    return torch.exp(
        _logprob(evidence.p if hypothesis else 1 - evidence.p, evidence.k, evidence.n)
    )


def marginal(prior, evidence):
    """
    Marginal likelihood, P(E) = P(H)P(E|H) + P(-H)P(E|-H)

    Args:
        prior, P(H)
        event likelihood, P(E|H)
    """
    return prior * likelihood(evidence) + (1.0 - prior) * likelihood(
        evidence, hypothesis=False
    )


def bayes(prior, evidence, occurred=True):
    """
    Updates prior with Bayes' rule.

    Args:
        prior, P(H)
        event likelihood, P(E|H)

    Returns:
        Posterior, P(H|E) = P(H)P(E|H)/P(E)
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
        prior, P(H)
        event likelyhood, P(E|H)
        certainty
    """
    # Posterior belief, P(H|E), obtained via strict conditionalization on evidence, E
    belief = bayes(prior, evidence)
    # Posterior belief if evidence E did not occur, P(H|~E) = P(H)P(~E|H)/P(~E)
    misbelief = bayes(prior, evidence, occurred=False)
    # Update posterior belief, in light of uncertainty
    return belief * certainty + misbelief * (1.0 - certainty)
