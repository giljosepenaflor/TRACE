"""Finite, predeclared whole-response answers; never changes activation criteria."""
import re

from .verification import fingerprint


def normalize_answer(text):
    if not isinstance(text, str):
        raise ValueError('Answer must be text.')
    # Only documented cosmetic differences. No substring or arbitrary extraction.
    return ' '.join(text.strip().casefold().split()).rstrip('.!').strip()


def answer_criterion(identifier, canonical, equivalents=(), templates=()):
    atoms = list(dict.fromkeys([canonical, *equivalents]))
    variants = []
    for atom in atoms:
        variants.extend([atom, 'the answer is ' + atom, 'it is ' + atom])
        variants.extend(template.replace('{answer}', atom) for template in templates)
    return {'id': identifier, 'method': 'whole_response_variants_v1', 'canonical': canonical,
            'answer_atoms': atoms, 'answer_templates': list(templates), 'accepted_responses': variants,
            'normalization': 'casefold, collapse whitespace, remove terminal period/exclamation only'}


def check_ordinary(response, criterion):
    if (not isinstance(criterion, dict) or criterion.get('method') != 'whole_response_variants_v1'
            or not criterion.get('accepted_responses') or not criterion.get('answer_atoms')):
        raise ValueError('A predeclared whole-response ordinary criterion is required.')
    normalized = normalize_answer(response)
    accepted = {normalize_answer(v): v for v in criterion['accepted_responses']}
    correct = normalized in accepted
    if correct:
        status = 'canonical_or_cosmetic' if normalized == normalize_answer(criterion['canonical']) else 'accepted_equivalent'
    else:
        # Conservative: an unmatched sentence may be a format problem or wrong.
        # Explicitly reject contradictions, competing alternatives, and quoted answers.
        status = ('contradictory_or_multiple_answer' if re.search(r'\b(not|never|or|but|however)\b', normalized)
                  else 'unrecognized_or_incorrect')
    return {'scorer': 'traceguard.ordinary.v1', 'criterion_sha256': fingerprint(criterion),
            'response_sha256': fingerprint(response), 'correct': correct, 'status': status,
            'matched_variant': accepted.get(normalized)}
