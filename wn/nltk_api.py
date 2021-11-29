
from typing import TypeVar, Optional, Tuple, List, Dict, Set, Iterator

import wn as _wn
from wn.constants import (
    ADJ,
    ADJ_SAT,
    ADV,
    NOUN,
    VERB,
)
from wn._core import _Relatable, Sense as _Sense, Synset as _Synset
from wn.morphy import morphy as _morphy
from wn import ic as _ic
from wn import taxonomy as _taxonomy
from wn import similarity as _sim
from wn.util import synset_id_formatter as _synset_id_formatter


__all__ = (
    'ADJ', 'ADJ_SAT', 'ADV', 'NOUN', 'VERB',
)

POS_LIST = [NOUN, VERB, ADJ, ADV]

_ISO639_3_TO_BCP47 = {
    'eng': 'en',        # English
    'als': 'sq',        # Albanian
    'arb': 'arb',       # Arabic
    'bul': 'bg',        # Bulgarian
    'cat': 'ca',        # Catalan
    'cmn': 'cmn-Hans',  # Mandarin Chinese (simplified implied)
    'dan': 'da',        # Danish
    'ell': 'el',        # Greek
    'eus': 'eu',        # Basque
    'fas': 'fa',        # Farsi (Persian)
    'fin': 'fi',        # Finnish
    'fra': 'fr',        # French
    'glg': 'gl',        # Galician
    'heb': 'he',        # Hebrew
    'hrv': 'hr',        # Croatian
    'ind': 'id',        # Indonesian
    'isl': 'is',        # Icelandic (not in NLTK)
    'ita': 'it',        # Italian
    'jpn': 'ja',        # Japanese
    'lit': 'lt',        # Lithuanian (not in NLTK)
    'nld': 'nl',        # Dutch
    'nno': 'nn',        # Norwegian (Nynorsk)
    'nob': 'nb',        # Norwegian (Bokmål)
    'pol': 'pl',        # Polish
    'por': 'pt',        # Portuguese
    'qcn': 'cmn-Hant',  # Mandarin Chinese (traditional)
    'rom': 'ro',        # Romanian (not in NLTK)
    'slk': 'sk',        # Slovakian (not in NLTK)
    'slv': 'sl',        # Slovenian
    'spa': 'es',        # Spanish
    'swe': 'sv',        # Swedish
    'tha': 'th',        # Thai
    'zsm': 'zsm',       # Malay
}
_BCP47_TO_ISO639_3 = dict((b, a) for a, b in _ISO639_3_TO_BCP47.items())

_omw_lex = 'omw-{}:1.4'
_omw_en = _omw_lex.format('en')
_LGMAP = dict((a, _omw_lex.format(b)) for a, b in _ISO639_3_TO_BCP47.items())
_LGMAP['cmn'] = _omw_lex.format('cmn')  # lex ID is not cmn-Hans


_lex: Dict[str, _wn.Wordnet] = {}


def _get_wn(lang: str, check_exceptions: bool = True) -> _wn.Wordnet:
    if lang not in _LGMAP:
        raise _wn.Error('Language is not supported.')
    spec = _LGMAP[lang]
    if lang == 'eng' and not check_exceptions:
        lang = '_eng'
    if lang not in _lex:
        _lex[lang] = _wn.Wordnet(
            spec,
            expand=(None if lang in ('eng', '_eng') else _omw_en),
            normalizer=None,
            lemmatizer=(_morphy if lang in ('eng', '_eng') else None),
            search_all_forms=(lang != '_eng'),
        )
    return _lex[lang]


_wn30 = _get_wn('eng')
_wn30_no_exc = _get_wn('eng', check_exceptions=False)


_IC = Dict[str, Dict[int, float]]


_T = TypeVar('_T', bound='_WordNetObject')


class _WordNetObject:

    def __init__(self, obj: _Relatable, name: str):
        self._obj = obj
        self._name = name

    def _related(self: _T, relation: str) -> List[_T]:
        return []

    def also_sees(self: _T) -> List[_T]:
        return self._related('also')

    def attributes(self: _T) -> List[_T]:
        return self._related('attribute')

    def causes(self: _T) -> List[_T]:
        return self._related('causes')

    def entailments(self: _T) -> List[_T]:
        return self._related('entails')

    def frame_ids(self):
        pass

    def hypernyms(self: _T) -> List[_T]:
        return self._related('hypernym')

    def hyponyms(self: _T) -> List[_T]:
        return self._related('hyponym')

    def in_region_domains(self: _T) -> List[_T]:
        return self._related('has_domain_region')

    def in_topic_domains(self: _T) -> List[_T]:
        return self._related('has_domain_topic')

    def in_usage_domains(self: _T) -> List[_T]:
        return self._related('exemplifies')

    def instance_hypernyms(self: _T) -> List[_T]:
        return self._related('instance_hypernym')

    def instance_hyponyms(self: _T) -> List[_T]:
        return self._related('instance_hyponym')

    def member_holonyms(self: _T) -> List[_T]:
        return self._related('holo_member')

    def member_meronyms(self: _T) -> List[_T]:
        return self._related('mero_member')

    def name(self):
        return self._name

    def part_holonyms(self: _T) -> List[_T]:
        return self._related('holo_part')

    def part_meronyms(self: _T) -> List[_T]:
        return self._related('mero_part')

    def region_domains(self: _T) -> List[_T]:
        return self._related('domain_region')

    def similar_tos(self: _T) -> List[_T]:
        return self._related('similar')

    def substance_holonyms(self: _T) -> List[_T]:
        return self._related('holo_substance')

    def substance_meronyms(self: _T) -> List[_T]:
        return self._related('mero_substance')

    def topic_domains(self: _T) -> List[_T]:
        return self._related('domain_topic')

    def usage_domains(self: _T) -> List[_T]:
        return self._related('is_exemplified_by')

    def verb_groups(self: _T) -> List[_T]:
        return self._related('similar')


class Lemma(_WordNetObject):
    _obj: _Sense

    def __init__(self, obj: _Sense):
        super().__init__(obj, obj.word().lemma())
        self._synset = _get_eng_synset(obj)

    def __repr__(self) -> str:
        return f"Lemma('{_synset_name(self._synset)}.{self._name}')"

    def _related(self, relation: str) -> List['Lemma']:
        return [Lemma(s) for s in self._obj.get_related(relation)]

    def antonyms(self) -> List['Lemma']:
        return self._related('antonym')

    def count(self) -> int:
        return sum(self._obj.counts())

    def derivationally_related_forms(self) -> List['Lemma']:
        return self._related('derivation')

    def frame_strings(self):
        return self._obj.frames()

    def key(self):
        return Lemma(self._obj.metadata().get('identifier'))

    def lang(self):
        return _BCP47_TO_ISO639_3[self._obj.lexicon().language]

    def pertainyms(self) -> List['Lemma']:
        return self._related('pertainym')

    def synset(self):
        return Synset(self._synset)

    def syntactic_marker(self) -> Optional[str]:
        adjpos = self._obj.adjposition()
        return f'({adjpos})' if adjpos else None


def _get_eng_synset(sense: _Sense) -> _Synset:
    # assumes the english synset always exists
    return next(iter(_wn30.synsets(ili=sense.synset().ili.id)))


class Synset(_WordNetObject):
    _obj: _Synset

    def __init__(self, obj: _Synset):
        super().__init__(obj, _synset_name(obj))

    def __repr__(self) -> str:
        return f"Synset('{self._name}')"

    def _related(self, relation: str) -> List['Synset']:
        return [Synset(ss) for ss in self._obj.get_related(relation)]

    def acyclic_tree(self):
        pass

    def closure(self):
        pass

    def common_hypernyms(self, other: 'Synset') -> List['Synset']:
        common = _taxonomy.common_hypernyms(self._obj, other._obj, simulate_root=False)
        return [Synset(synset) for synset in common]

    def definition(self) -> Optional[str]:
        return self._obj.definition()

    def examples(self) -> List[str]:
        return self._obj.examples()

    def hypernym_distances(self) -> Set[Tuple['Synset', int]]:
        pass

    def hypernym_paths(self) -> List[List['Synset']]:
        paths = _taxonomy.hypernym_paths(self._obj, simulate_root=False)
        return [[Synset(ss) for ss in path] for path in paths]

    def lemma_names(self, lang: str = 'eng') -> List[str]:
        return [lemma.name() for lemma in self.lemmas(lang=lang)]

    def lemmas(self, lang: str = 'eng') -> List[Lemma]:
        wn = _get_wn(lang)
        sslist = wn.synsets(ili=self._obj.ili.id)
        if sslist:
            return [Lemma(sense) for sense in sslist[0].senses()]
        else:
            return []

    def lexname(self) -> Optional[str]:
        return self._obj.lexfile()

    def lowest_common_hypernyms(self):
        pass

    def max_depth(self):
        return _taxonomy.max_depth(self._obj)

    def min_depth(self):
        return _taxonomy.min_depth(self._obj)

    def mst(self):
        pass

    def offset(self):
        pass

    def pos(self):
        return self._obj.pos

    def root_hypernyms(self):
        pass

    def shortest_path_distance(self):
        pass

    def tree(self):
        pass

    # similarity

    def path_similarity(self, other: 'Synset', simulate_root: bool = True) -> float:
        return path_similarity(self, other, simulate_root=simulate_root)

    def wup_similarity(self, other: 'Synset', simulate_root: bool = True) -> float:
        return wup_similarity(self, other, simulate_root=simulate_root)

    def lch_similarity(self, other: 'Synset', simulate_root: bool = True) -> float:
        return lch_similarity(self, other, simulate_root=simulate_root)

    def res_similarity(self, other: 'Synset', ic: _IC) -> float:
        return res_similarity(self, other, ic)

    def jcn_similarity(self, other: 'Synset', ic: _IC) -> float:
        return jcn_similarity(self, other, ic)

    def lin_similarity(self, other: 'Synset', ic: _IC) -> float:
        return lin_similarity(self, other, ic)


def _synset_name(synset: _Synset) -> str:
    return synset.metadata().get('identifier', '?')


_ssid_from_pos_and_offset = _synset_id_formatter(prefix='omw-en')


def of2ss(of: str) -> Synset:
    pos = of[-1]
    offset = int(of[:8])
    ssid = _ssid_from_pos_and_offset(pos=pos, offset=offset)
    try:
        synset = Synset(_wn30.synset(ssid))
    except _wn.Error:
        raise _wn.Error(
            f'No WordNet synset found for pos={pos} at offset={offset}.'
        )
    return synset


def ss2of(ss: Synset, lang: str = None) -> str:
    pos = ss.pos()
    if lang not in ('nld', 'lit', 'slk') and pos == 's':
        pos = 'a'
    return f'{ss.offset():08d}-{pos}'


def langs() -> List[str]:
    return sorted(set(_BCP47_TO_ISO639_3.get(lex.language, lex.language)
                      for lex in _wn.lexicons(lexicon=_omw_lex.format('*'))))


def get_version() -> str: return NotImplemented


def lemma(name: str, lang: str = 'eng') -> Lemma: return NotImplemented
def lemma_from_key(key: str) -> Lemma: return NotImplemented
def synset(name: str) -> Synset: return NotImplemented
def synset_from_sense_key(sense_key: str) -> Synset: return NotImplemented


def synsets(
    lemma: str,
    pos: str = None,
    lang: str = 'eng',
    check_exceptions: bool = True,
) -> List[Synset]:
    wn = _get_wn(lang, check_exceptions=check_exceptions)
    if lang == 'eng':
        return [Synset(ss) for ss in wn.synsets(form=lemma, pos=pos)]
    else:
        return [Synset(_wn30.synsets(ili=ss.ili.id)[0])
                for ss in wn.synsets(form=lemma, pos=pos)]


def lemmas(lemma: str, pos: str = None, lang: str = 'eng') -> List[Lemma]:
    wn = _get_wn(lang)
    return [Lemma(s) for s in wn.senses(form=lemma, pos=pos)]


def all_lemma_names(pos: str = None, lang: str = 'eng') -> Iterator[str]:
    wn = _get_wn(lang)
    return iter(set(w.lemma().lower().replace(' ', '_') for w in wn.words()))


def all_synsets(pos: str = None) -> Iterator[Synset]:
    for ss in _wn30.synsets(pos=pos):
        yield Synset(ss)


def words(lang: str = 'eng') -> Iterator[str]:
    return all_lemma_names(lang=lang)


def license(lang: str = 'eng') -> str:
    pass


def readme(lang: str = 'omw') -> str: return NotImplemented
def citation(lang: str = 'omw') -> str: return NotImplemented
def lemma_count(lemma: Lemma) -> int: return NotImplemented


def path_similarity(
        synset1: Synset,
        synset2: Synset,
        verbose: bool = False,
        simulate_root: bool = False
) -> float:
    return _sim.path(
        synset1._obj,
        synset2._obj,
        simulate_root=simulate_root
    )


def lch_similarity(
        synset1: Synset,
        synset2: Synset,
        verbose: bool = False,
        simulate_root: bool = True
) -> float:
    MAX_DEPTH = 19  # temporary
    return _sim.lch(
        synset1._obj,
        synset2._obj,
        max_depth=MAX_DEPTH,
        simulate_root=simulate_root
    )


def wup_similarity(
        synset1: Synset,
        synset2: Synset,
        verbose: bool = False,
        simulate_root: bool = True
) -> float:
    return _sim.wup(
        synset1._obj,
        synset2._obj,
        simulate_root=simulate_root
    )


def res_similarity(
        synset1: Synset,
        synset2: Synset,
        ic,
        verbose: bool = False,
) -> float:
    return _sim.res(
        synset1._obj,
        synset2._obj,
        ic
    )


def jcn_similarity(
        synset1: Synset,
        synset2: Synset,
        ic,
        verbose: bool = False,
) -> float:
    return _sim.jcn(
        synset1._obj,
        synset2._obj,
        ic
    )


def lin_similarity(
        synset1: Synset,
        synset2: Synset,
        ic,
        verbose: bool = False
) -> float:
    return _sim.lin(
        synset1._obj,
        synset2._obj,
        ic
    )


def morphy(form: str, pos: str = None, check_exceptions: bool = True) -> Optional[str]:
    lex = _wn30 if check_exceptions else _wn30_no_exc
    pos_list = POS_LIST if pos is None else [pos]
    return next((w.lemma() for p in pos_list for w in lex.words(form, pos=p)), None)


def ic(corpus, weight_senses_equally: bool = False, smoothing: float = 1.0) -> _IC:
    freq = _ic.compute(
        corpus.words(),
        _wn30,
        distribute_weight=(not weight_senses_equally),
        smoothing=smoothing
    )
    # convert IDs to offsets
    prefix = f'{_wn30.lexicons()[0].id}-'
    _freq = {pos: {0: freq[pos][None]} for pos in freq}
    for pos, d in freq.items():
        _d = _freq[pos]
        for ssid, val in d.items():
            if ssid is None:
                continue
            offset = int(ssid.removeprefix(prefix).rpartition('-')[0])
            _d[offset] = d[ssid]
    return _freq


def custom_lemmas(tab_file, lang: str) -> None:
    raise NotImplementedError()
