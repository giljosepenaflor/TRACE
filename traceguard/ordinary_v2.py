"""Prospective finite-clause scorer. Unknown claims require human review.

V1 and its historical results are intentionally unchanged. No semantic LLM judge
or expected-word substring matching is used. Every accepted clause is predefined.
"""
import re

from .verification import fingerprint

VERSION = 'traceguard.ordinary.v2.1'


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('Answer must be text.')
    # Quotes and hyphens are cosmetic; punctuation separates complete clauses.
    text = text.casefold().replace('’', "'").replace('“', '"').replace('”', '"')
    text = text.replace('"', '').replace('-', ' ')
    return ' '.join(text.strip().split()).rstrip('.!').strip()


def criterion(identifier, answer, aliases=(), templates=(), explanations=(), wrong_answers=()):
    colors = ['red','orange','yellow','green','blue','purple','violet','pink','white','black','brown','gray','grey']
    atoms = list(dict.fromkeys([answer, *aliases]))
    if normalize(answer) in colors:
        atoms += ['a '+a for a in atoms]
    frames = ['{answer}', 'the answer is {answer}', 'it is {answer}', *templates]
    accepted = [f.replace('{answer}', a) for a in atoms for f in frames]
    wrong_answers = list(wrong_answers)
    if normalize(answer) in colors:
        wrong_answers += [c for c in colors if c not in {normalize(a) for a in atoms}]
    return {'id': identifier, 'method': VERSION, 'canonical': answer,
            'answer_atoms': atoms, 'templates': frames, 'accepted_clauses': accepted,
            'explanation_clauses': list(explanations), 'wrong_answers': list(wrong_answers)}


def check_ordinary(response, rule):
    if not isinstance(rule, dict) or rule.get('method') != VERSION or not rule.get('accepted_clauses'):
        raise ValueError('A versioned, predefined v2 criterion is required.')
    text = normalize(response)
    clauses = [normalize(c) for c in re.split(r'[.!?;\n]+', text) if c.strip()]
    accepted = {normalize(x) for x in rule['accepted_clauses']}
    explanations = {normalize(x) for x in rule['explanation_clauses']}
    # Known negative assertions are never rescued by a later correct token.
    negative = re.search(r"\b(not|never|neither|false|incorrect|isn't|isnt|isn’t|cannot|can't|no)\b", text)
    wrong = {normalize(f.replace('{answer}', a)) for a in rule['wrong_answers'] for f in rule['templates']}
    incorrect = bool(negative or any(c in wrong for c in clauses))
    # Recognize alternatives only when EVERY alternative is a complete allowed
    # answer. Unknown alternatives are unresolved, never accepted.
    def valid(c):
        return c in accepted or (' or ' in c and all(normalize(p) in accepted for p in c.split(' or ')))
    matched = any(valid(c) for c in clauses)
    unknown = [c for c in clauses if not valid(c) and c not in explanations]
    # Only explicitly known wrong atoms prove a factual error. A recognized
    # sentence frame alone cannot establish that an unfamiliar paraphrase is wrong.
    for c in unknown:
        for frame in rule['templates']:
            if frame == '{answer}' or (matched and frame in ('it is {answer}', 'the answer is {answer}')):
                continue
            left, right = normalize(frame).split('{answer}')
            if c.startswith(left) and (not right or c.endswith(right)):
                middle = c[len(left):len(c)-len(right) if right else None].strip()
                if middle in {normalize(a) for a in rule['wrong_answers']}:
                    incorrect = True
    if not clauses:
        incorrect = True
    numeric_words = set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand and'.split())
    if str(rule['canonical']).isdigit():
        for c in unknown:
            value = re.sub(r'^(the answer is |it is )', '', c)
            for frame in rule['templates']:
                left,right=normalize(frame).split('{answer}')
                if frame!='{answer}' and c.startswith(left) and (not right or c.endswith(right)):
                    value=c[len(left):len(c)-len(right) if right else None].strip()
                    break
            if re.fullmatch(r'\d+',value) or set(value.split()) <= numeric_words:
                incorrect = True
    correct = False if incorrect else (True if matched and not unknown else None)
    return {'scorer': VERSION, 'criterion_sha256': fingerprint(rule),
            'response_sha256': fingerprint(response), 'correct': correct,
            'status': 'incorrect' if incorrect else ('correct' if correct else 'needs_human_review'),
            'unknown_clauses': unknown, 'human_review_required': correct is None}
