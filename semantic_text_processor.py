"""
Semantic Text Processor (Single File, Offline)
- No LLM/API usage
- Uses scikit-learn TF-IDF, cosine similarity, and lightweight classifiers
- Provides: intent classification, pest detection, symptom detection,
  severity/urgency estimation, context analysis, and recommendations hooks
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

import re
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


class QueryIntent(Enum):
    PEST_IDENTIFICATION = "pest_identification"
    SYMPTOM_DESCRIPTION = "symptom_description"
    TREATMENT_REQUEST = "treatment_request"
    SEVERITY_ASSESSMENT = "severity_assessment"
    GENERAL_INQUIRY = "general_inquiry"


@dataclass
class ProcessedQuery:
    intent: QueryIntent
    detected_pests: List[str]
    symptoms: List[str]
    severity: str
    urgency: str
    confidence: float
    context: Dict[str, Any]


class SemanticTextProcessor:
    """
    Offline semantic text analyzer built around TF-IDF similarity and
    a small logistic regression classifier trained on curated templates.
    """

    def __init__(self) -> None:
        # Knowledge bases
        self.pest_database: Dict[str, Dict[str, List[str]]] = self._load_pest_database()
        self.symptom_database: Dict[str, Dict[str, List[str]]] = self._load_symptom_database()
        self.plant_synonyms: Dict[str, List[str]] = self._load_plant_synonyms()
        self.normalization_map: Dict[str, str] = self._load_normalization_map()

        # Build TF-IDF semantic space using all phrases we care about
        self.reference_terms: List[str] = self._build_reference_terms()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        # Fit vectorizer on references so cosine similarity works immediately
        self.reference_matrix = self.vectorizer.fit_transform(self.reference_terms)

        # LSA projection for deeper semantics (offline)
        self._init_lsa()

        # Train a tiny intent classifier on synthetic templates (offline)
        self.intent_clf: Pipeline = self._train_intent_classifier()

    def process_query(self, query: str) -> ProcessedQuery:
        query_norm = self._normalize(query)

        # Intent
        intent = self._classify_intent(query_norm)

        # Semantic detections
        detected_pests = self._semantic_detect(query_norm, self._all_pest_aliases(), min_sim=0.32)
        detected_pests = self._canonicalize_pests(detected_pests)

        detected_symptoms = self._semantic_detect(query_norm, self._all_symptom_aliases(), min_sim=0.32)
        detected_symptoms = self._canonicalize_symptoms(detected_symptoms)

        # Severity & urgency
        severity = self._assess_severity(query_norm, detected_pests, detected_symptoms)
        urgency = self._detect_urgency(query_norm)

        # Confidence
        confidence = self._calculate_confidence(query_norm, detected_pests, detected_symptoms)

        # Context
        context = self._basic_context(query_norm, detected_pests, detected_symptoms)

        return ProcessedQuery(
            intent=intent,
            detected_pests=detected_pests,
            symptoms=detected_symptoms,
            severity=severity,
            urgency=urgency,
            confidence=confidence,
            context=context,
        )

    def analyze_context(self, query: str, detected_pests: List[str], symptoms: List[str]) -> Dict[str, Any]:
        q = self._normalize(query)
        plant_context = self._detect_plant_context(q)
        environmental_factors = self._detect_environmental_factors(q)

        # Simple risk heuristic
        risk = "low"
        high_pests = {"caterpillar", "grasshopper", "beetle"}
        if any(p in high_pests for p in detected_pests) or len(symptoms) >= 2:
            risk = "medium"
        if len(detected_pests) >= 2 or "defoliation" in symptoms or "wilting" in symptoms:
            risk = "high"

        # Recommended focus
        if risk == "high":
            focus = "immediate_pest_control"
        elif plant_context != "unknown":
            focus = "targeted_treatment"
        else:
            focus = "preventive_measures"

        # Temporal/spatial heuristics
        temporal = (
            "recent"
            if any(w in q for w in ["today", "now", "lately", "recently", "just", "this week"]) else
            ("ongoing" if any(w in q for w in ["for days", "for weeks", "keeps happening", "continuously"]) else
             ("sudden" if any(w in q for w in ["suddenly", "overnight", "quickly"]) else "none"))
        )
        spatial = (
            "widespread" if any(w in q for w in ["everywhere", "entire", "whole", "throughout"]) else
            ("localized" if any(w in q for w in ["corner", "edge", "one area", "specific spot", "patches"]) else "none")
        )

        return {
            "primary_concern": self._primary_concern(detected_pests, symptoms, plant_context),
            "secondary_concerns": [],
            "temporal_context": temporal,
            "spatial_context": spatial,
            "plant_context": plant_context,
            "environmental_factors": environmental_factors,
            "confidence_level": min(1.0, 0.5 + 0.1 * len(detected_pests) + 0.1 * len(symptoms)),
            "recommended_focus": focus,
            "context": {"risk_level": risk},
        }

    # ----------------------- internals -----------------------

    def _normalize(self, text: str) -> str:
        s = re.sub(r"\s+", " ", text.strip().lower())
        # Apply simple normalization for common misspellings and variants
        for wrong, right in self.normalization_map.items():
            if wrong in s:
                s = s.replace(wrong, right)
        return s

    def _build_reference_terms(self) -> List[str]:
        terms: List[str] = []
        for pest, data in self.pest_database.items():
            terms.append(pest)
            terms.extend(data["synonyms"])  # type: ignore
            terms.extend(data["characteristics"])  # type: ignore
        for sym, data in self.symptom_database.items():
            terms.append(sym)
            terms.extend(data["keywords"])  # type: ignore
            terms.extend(data["synonyms"])  # type: ignore
        # Plants and environment terms for better space coverage
        terms.extend(["tomato", "cabbage", "corn", "lettuce", "pepper", "bean"])
        terms.extend(["rain", "wet", "humid", "dry", "hot", "cold", "soil", "watering"])
        return sorted(set(terms))

    def _train_intent_classifier(self) -> Pipeline:
        # Tiny synthetic dataset (offline, no downloads)
        X = [
            # pest identification
            "what pest is this", "identify the insect", "which bug is causing this",
            # symptom description
            "leaves have holes", "plants are wilting", "yellow spots on leaves",
            # treatment request
            "how to treat pests", "need pesticide recommendation", "how to control caterpillars",
            # severity assessment
            "how bad is this infestation", "is this severe damage", "many pests destroying everything",
            # general inquiry
            "need help with my garden", "advice about plant care", "information about pests",
        ]
        y = [
            "pest_identification", "pest_identification", "pest_identification",
            "symptom_description", "symptom_description", "symptom_description",
            "treatment_request", "treatment_request", "treatment_request",
            "severity_assessment", "severity_assessment", "severity_assessment",
            "general_inquiry", "general_inquiry", "general_inquiry",
        ]
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(X, y)
        return pipe

    def _classify_intent(self, text: str) -> QueryIntent:
        try:
            label = self.intent_clf.predict([text])[0]
            return QueryIntent(label)
        except Exception:
            return QueryIntent.GENERAL_INQUIRY

    def _vectorize(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def _semantic_detect(self, query: str, candidates: List[str], min_sim: float = 0.32) -> List[str]:
        # TF-IDF cosine
        q_vec = self._vectorize([query])
        cand_vecs = self._vectorize(candidates)
        sims_tfidf = cosine_similarity(q_vec, cand_vecs)[0]

        # LSA cosine (latent semantics)
        q_lsa = self.lsa_transformer.transform(q_vec)
        cand_lsa = self.lsa_transformer.transform(cand_vecs)
        sims_lsa = cosine_similarity(q_lsa, cand_lsa)[0]

        # Blend scores
        sims = 0.6 * sims_tfidf + 0.4 * sims_lsa

        # Keep top-k (avoid over-detection)
        top = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)[:5]
        return [c for c, s in top if s >= min_sim]

    def _init_lsa(self) -> None:
        # Project TF-IDF space into lower-dimensional latent space
        # Keep modest components to stay light-weight
        self.lsa_svd = TruncatedSVD(n_components=64, random_state=42)
        self.lsa_norm = Normalizer(copy=False)
        # Fit on reference space
        X = self.reference_matrix
        X_svd = self.lsa_svd.fit_transform(X)
        X_lsa = self.lsa_norm.fit_transform(X_svd)
        # Build transformer that maps new TF-IDF vectors into LSA space
        class _LSATransformer:
            def __init__(self, svd, norm):
                self._svd = svd
                self._norm = norm
            def transform(self, X_):
                return self._norm.transform(self._svd.transform(X_))
        self.lsa_transformer = _LSATransformer(self.lsa_svd, self.lsa_norm)

    def _assess_severity(self, query: str, pests: List[str], symptoms: List[str]) -> str:
        score = 0.0
        if any(w in query for w in ["many", "lots", "severe", "infestation", "destroying", "completely"]):
            score += 1.0
        if len(pests) >= 2:
            score += 0.5
        if any(s in ["defoliation", "wilting"] for s in symptoms):
            score += 0.7
        if score >= 1.2:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def _detect_urgency(self, query: str) -> str:
        return "high" if any(w in query for w in ["urgent", "immediately", "asap", "emergency"]) else "normal"

    def _calculate_confidence(self, query: str, pests: List[str], symptoms: List[str]) -> float:
        confidence = 0.4 if pests else 0.2
        confidence += 0.2 if symptoms else 0.0
        confidence += 0.1 if len(query.split()) > 8 else 0.0
        return min(1.0, round(confidence, 2))

    def _basic_context(self, query: str, pests: List[str], symptoms: List[str]) -> Dict[str, Any]:
        return {
            "query_type": "descriptive" if len(query.split()) > 6 else "brief",
            "pest_symptom_correlation": [],
            "recommended_actions": [],
            "risk_level": "high" if "defoliation" in symptoms or len(pests) >= 2 else ("medium" if pests or symptoms else "low"),
        }

    def _primary_concern(self, pests: List[str], symptoms: List[str], plant: str) -> str:
        if not pests and not symptoms:
            return "general_inquiry"
        if any(p in {"caterpillar", "grasshopper", "beetle"} for p in pests):
            return "severe_pest_damage"
        if plant != "unknown" and pests:
            return "plant_specific_pest_issue"
        if any(s in {"defoliation", "wilting"} for s in symptoms):
            return "severe_symptom_damage"
        return "moderate_pest_issue"

    def _detect_plant_context(self, query: str) -> str:
        # direct plant names
        for p in self.plant_synonyms.keys():
            if p in query or (p + "s") in query:
                return p
        # synonym match
        for plant, syns in self.plant_synonyms.items():
            for s in syns:
                if s in query:
                    return plant
        # phrase patterns
        m = re.search(r"my ([a-z\s]+?) (plants|crop|leaves)", query)
        if m:
            phrase = m.group(1).strip()
            for plant, syns in self.plant_synonyms.items():
                if plant in phrase or any(s in phrase for s in syns):
                    return plant
        return "none"

    def _detect_environmental_factors(self, query: str) -> List[str]:
        mapping = {
            "weather": [
                "rain", "rainy", "wet", "humid", "dry", "hot", "cold",
                "monsoon", "storm", "windy", "heat", "drought", "flood"
            ],
            "soil": ["soil", "drainage", "fertilizer", "nutrients"],
            "water": ["watering", "irrigation", "moisture"],
        }
        found: List[str] = []
        for k, words in mapping.items():
            if any(w in query for w in words):
                found.append(k)
        return found or ["none"]

    def _all_pest_aliases(self) -> List[str]:
        aliases: List[str] = []
        for name, data in self.pest_database.items():
            aliases.append(name)
            aliases.extend(data["synonyms"])  # type: ignore
        return sorted(set(aliases))

    def _all_symptom_aliases(self) -> List[str]:
        aliases: List[str] = []
        for name, data in self.symptom_database.items():
            aliases.append(name)
            aliases.extend(data["keywords"])  # type: ignore
            aliases.extend(data["synonyms"])  # type: ignore
        return sorted(set(aliases))

    def _canonicalize_pests(self, detected: List[str]) -> List[str]:
        result: List[str] = []
        for name, data in self.pest_database.items():
            if name in detected or any(a in detected for a in data["synonyms"]):  # type: ignore
                result.append(name)
        return sorted(set(result))

    def _canonicalize_symptoms(self, detected: List[str]) -> List[str]:
        result: List[str] = []
        for name, data in self.symptom_database.items():
            if name in detected or any(a in detected for a in (data["keywords"] + data["synonyms"])):  # type: ignore
                result.append(name)
        return sorted(set(result))

    def _load_pest_database(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "ants": {"synonyms": ["ant", "ants"], "characteristics": ["colony", "small", "black", "brown"]},
            "bees": {"synonyms": ["bee", "bees"], "characteristics": ["yellow", "black", "flying"]},
            "beetle": {"synonyms": ["beetle", "beetles", "bug", "bugs"], "characteristics": ["hard shell", "spots"]},
            "caterpillar": {"synonyms": ["caterpillar", "caterpillars", "worm", "larva", "larvae", "inchworm"], "characteristics": ["green", "hairy", "soft body"]},
            "earthworms": {"synonyms": ["earthworm", "earthworms", "worm", "worms"], "characteristics": ["underground", "soil"]},
            "earwig": {"synonyms": ["earwig", "earwigs"], "characteristics": ["pincers", "nocturnal"]},
            "grasshopper": {"synonyms": ["grasshopper", "grasshoppers", "locust"], "characteristics": ["jumping", "green", "brown"]},
            "moth": {"synonyms": ["moth", "moths"], "characteristics": ["flying", "nocturnal"]},
            "slug": {"synonyms": ["slug", "slugs"], "characteristics": ["slimy", "trail"]},
            "snail": {"synonyms": ["snail", "snails"], "characteristics": ["shell", "slow"]},
            "wasp": {"synonyms": ["wasp", "wasps"], "characteristics": ["stinging", "aggressive"]},
            "weevil": {"synonyms": ["weevil", "weevils"], "characteristics": ["snout", "small"]},
        }

    def _load_symptom_database(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "holes_in_leaves": {
                "keywords": ["holes", "chewed", "eaten", "damaged leaves", "perforated"],
                "synonyms": ["leaf damage", "chewing damage", "feeding damage"],
            },
            "yellowing": {
                "keywords": ["yellow", "yellowing", "discolored", "chlorosis"],
                "synonyms": ["pale leaves", "leaf yellowing"],
            },
            "wilting": {
                "keywords": ["wilt", "wilting", "drooping", "limp"],
                "synonyms": ["droopy leaves", "loss of turgor"],
            },
            "spots": {
                "keywords": ["spots", "patches", "marks", "lesions"],
                "synonyms": ["leaf spots", "blemishes"],
            },
            "stunted_growth": {
                "keywords": ["stunted", "small", "not growing", "slow growth"],
                "synonyms": ["retarded growth", "dwarfing"],
            },
            "defoliation": {
                "keywords": ["defoliation", "leaf loss", "stripped"],
                "synonyms": ["complete leaf loss", "denuded"],
            },
        }

    def _load_plant_synonyms(self) -> Dict[str, List[str]]:
        return {
            "tomato": ["tomatoes", "tomato plant", "tomato crop", "tomato plants"],
            "cabbage": ["cabbage plant", "cabbage crop", "brassica", "cabbages"],
            "corn": ["maize", "corn plant", "corn crop", "corns"],
            "lettuce": ["lettuce plant", "lettuce crop", "leafy greens"],
            "pepper": ["pepper plant", "pepper crop", "bell pepper", "chili", "chilli", "capsicum", "peppers"],
            "bean": ["bean plant", "bean crop", "beans", "legumes"],
            "rose": ["roses", "rose plant"],
        }

    def _load_normalization_map(self) -> Dict[str, str]:
        return {
            # common pest misspellings
            "catterpillar": "caterpillar",
            "catapillar": "caterpillar",
            "catepillar": "caterpillar",
            "gras hopper": "grasshopper",
            # morphological variants we want unified
            "larvae": "larva",
            "insects": "insect",
        }


