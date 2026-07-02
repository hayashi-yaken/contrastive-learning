"""WordNet 3.0 noun hypernymy loading. Nickel-Kiela setup: synsets are nodes,
transitive closure of hypernymy are the edges. Glosses feed the encoder track."""
import re
from dataclasses import dataclass, field


def ensure_wordnet():
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


_WS = re.compile(r"\s+")


def gloss_text(synset, max_lemmas=3):
    """Fixed preprocessing: '<lemma1 lemma2 ...> : <definition>', lowercased,
    whitespace-collapsed. Controls length/lemma bias (ESimCSE caveat)."""
    lemmas = [l.replace("_", " ") for l in synset.lemma_names()[:max_lemmas]]
    head = " ".join(lemmas)
    body = synset.definition() or ""
    text = f"{head} : {body}".lower()
    return _WS.sub(" ", text).strip()


@dataclass
class WordNetData:
    synsets: list
    index: dict
    direct_edges: list
    closure_edges: list
    glosses: list
    roots: list = field(default_factory=list)


def load_noun_hypernymy(max_synsets=None):
    from nltk.corpus import wordnet as wn
    all_syn = list(wn.all_synsets(pos="n"))
    if max_synsets is not None:
        all_syn = all_syn[:max_synsets]
    names = [s.name() for s in all_syn]
    index = {n: i for i, n in enumerate(names)}
    present = set(index)

    direct, closure, roots = [], [], []
    for s in all_syn:
        i = index[s.name()]
        parents = s.hypernyms() + s.instance_hypernyms()
        parents_in = [p for p in parents if p.name() in present]
        if not parents_in:
            roots.append(i)
        for p in parents_in:
            direct.append((i, index[p.name()]))
        # transitive closure: all ancestors reachable via hypernym paths
        seen = set()
        stack = list(parents)
        while stack:
            p = stack.pop()
            if p.name() in seen:
                continue
            seen.add(p.name())
            if p.name() in present:
                closure.append((i, index[p.name()]))
            stack.extend(p.hypernyms() + p.instance_hypernyms())

    glosses = [gloss_text(s) for s in all_syn]
    return WordNetData(names, index, direct, closure, glosses, roots)
