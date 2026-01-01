"""
Credibility Scoring Engine (CSE) for ATLAS v2.0

This module implements weighted truth scoring combining:
- Source trust ratings
- Cross-time agreement
- Semantic alignment (using sentence transformers)
- Evidence diversity
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

# Enhanced semantic matching with sentence-transformers
# Lazy import to avoid slow startup
SEMANTIC_AVAILABLE = False
_sentence_transformer = None
_numpy = None

def _get_semantic_libs():
    """Lazy load sentence transformers and numpy"""
    global SEMANTIC_AVAILABLE, _sentence_transformer, _numpy
    if not SEMANTIC_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _sentence_transformer = SentenceTransformer
            _numpy = np
            SEMANTIC_AVAILABLE = True
        except ImportError:
            logging.warning("sentence-transformers not installed. Using basic keyword matching. Install with: pip install sentence-transformers")
    return SEMANTIC_AVAILABLE


@dataclass
class Source:
    """Represents an evidence source with trust metrics."""
    url: str
    domain: str
    content: str
    timestamp: datetime
    trust_score: float = 0.5  # Default neutral trust
    bias_flags: List[str] = None
    
    def __post_init__(self):
        if self.bias_flags is None:
            self.bias_flags = []


@dataclass
class CredibilityScore:
    """Complete credibility assessment for a claim."""
    overall_score: float  # 0.0 to 1.0
    source_trust: float
    semantic_alignment: float
    temporal_consistency: float
    evidence_diversity: float
    confidence_level: str  # "High", "Medium", "Low"
    explanation: str
    warnings: List[str]


class CredibilityEngine:
    """
    Weighted truth scoring engine that combines multiple signals:
    
    ATLAS v2 (4-axis):
    1. Source Trust: Domain reputation and historical accuracy
    2. Semantic Alignment: Advanced similarity using sentence transformers
    3. Temporal Consistency: Time-based validation
    4. Evidence Diversity: Multiple independent sources
    
    ATLAS v3 (7-axis) - Enhanced Multi-Axis System:
    1. Domain Reliability (dynamic)
    2. Ecosystem Cross-Verification (local, global, wire, regional, citizen, OSINT)
    3. Temporal Consistency (dates/times/metadata alignment)
    4. Source Independence (from SIG Module B)
    5. Sentiment & Framing Analysis (PR-speak, emotional manipulation)
    6. Evidence Diversity (varied sources)
    7. Fact-checker / RAG Hit (strong boost/penalty)
    """
    
    # Weights for v2 scoring components (4-axis - must sum to 1.0)
    WEIGHTS_V2 = {
        'source_trust': 0.30,
        'semantic_alignment': 0.35,
        'temporal_consistency': 0.15,
        'evidence_diversity': 0.20
    }
    
    # Weights for v3 scoring components (7-axis - must sum to 1.0)
    WEIGHTS_V3 = {
        'domain_reliability': 0.20,
        'ecosystem_verification': 0.20,
        'temporal_consistency': 0.10,
        'source_independence': 0.20,
        'sentiment_framing': 0.10,
        'evidence_diversity': 0.10,
        'factchecker_rag': 0.10
    }
    
    # Trusted domain list (can be expanded)
    TRUSTED_DOMAINS = {
        'reuters.com': 0.9,
        'apnews.com': 0.9,
        'bbc.com': 0.85,
        'npr.org': 0.85,
        'economist.com': 0.8,
        'nytimes.com': 0.75,
        'theguardian.com': 0.75,
        'wsj.com': 0.75,
    }
    
    def __init__(self, use_v3_scoring: bool = False):
        """
        Initialize Credibility Engine
        
        Args:
            use_v3_scoring: If True, use 7-axis v3 scoring. If False, use 4-axis v2 scoring.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.semantic_model = None
        self.use_v3_scoring = use_v3_scoring
        
        # Initialize semantic similarity model if available
        if _get_semantic_libs():
            try:
                # Use lightweight but accurate model
                self.semantic_model = _sentence_transformer('all-MiniLM-L6-v2')
                self.logger.info("✅ Semantic similarity model loaded (sentence-transformers)")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}. Falling back to keyword matching.")
                self.semantic_model = None
        else:
            self.logger.info("Using basic keyword matching for semantic alignment")
        
        scoring_version = "v3 (7-axis)" if use_v3_scoring else "v2 (4-axis)"
        self.logger.info(f"Credibility Engine initialized with {scoring_version} scoring")
        
    def calculate_credibility(
        self,
        claim: str,
        sources: List[Source],
        evidence_texts: List[str],
        **kwargs  # Additional context for v3 scoring
    ) -> CredibilityScore:
        """
        Calculate comprehensive credibility score for a claim.
        
        Routes to v2 (4-axis) or v3 (7-axis) scoring based on initialization.
        
        Args:
            claim: The statement being verified
            sources: List of Source objects with metadata
            evidence_texts: List of extracted evidence snippets
            **kwargs: Additional context (sig_result, pr_results, rag_hits) for v3 scoring
            
        Returns:
            CredibilityScore object with detailed assessment
        """
        if self.use_v3_scoring:
            return self._calculate_credibility_v3(claim, sources, evidence_texts, **kwargs)
        else:
            return self._calculate_credibility_v2(claim, sources, evidence_texts)
    
    def _calculate_credibility_v2(
        self,
        claim: str,
        sources: List[Source],
        evidence_texts: List[str]
    ) -> CredibilityScore:
        """
        Original v2 scoring with 4 axes
        """
        warnings = []
        
        # 1. Source Trust Score
        source_trust = self._calculate_source_trust(sources)
        if source_trust < 0.3:
            warnings.append("Low source credibility detected")
        
        # 2. Semantic Alignment (simplified - can be enhanced with embeddings)
        semantic_alignment = self._calculate_semantic_alignment(claim, evidence_texts)
        if semantic_alignment < 0.4:
            warnings.append("Weak evidence-claim alignment")
        
        # 3. Temporal Consistency
        temporal_consistency = self._calculate_temporal_consistency(sources)
        if temporal_consistency < 0.5:
            warnings.append("Temporal inconsistencies detected")
        
        # 4. Evidence Diversity
        evidence_diversity = self._calculate_evidence_diversity(sources)
        if evidence_diversity < 0.3:
            warnings.append("Limited source diversity")
        
        # Calculate weighted overall score
        overall_score = (
            self.WEIGHTS_V2['source_trust'] * source_trust +
            self.WEIGHTS_V2['semantic_alignment'] * semantic_alignment +
            self.WEIGHTS_V2['temporal_consistency'] * temporal_consistency +
            self.WEIGHTS_V2['evidence_diversity'] * evidence_diversity
        )
        
        # Determine confidence level
        if overall_score >= 0.75:
            confidence = "High"
        elif overall_score >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        explanation = self._generate_explanation(
            overall_score, source_trust, semantic_alignment,
            temporal_consistency, evidence_diversity
        )
        
        self.logger.info(f"Credibility Score: {overall_score:.2f} ({confidence} confidence)")
        
        return CredibilityScore(
            overall_score=overall_score,
            source_trust=source_trust,
            semantic_alignment=semantic_alignment,
            temporal_consistency=temporal_consistency,
            evidence_diversity=evidence_diversity,
            confidence_level=confidence,
            explanation=explanation,
            warnings=warnings
        )
    
    def _calculate_source_trust(self, sources: List[Source]) -> float:
        """Calculate average trust score across all sources."""
        if not sources:
            self.logger.warning("No sources provided for trust calculation")
            # Return neutral score instead of 0 when no sources available
            return 0.5
        
        trust_scores = []
        
        # Phase 3 Integration: Use dynamic outlet reliability if available
        try:
            from agents.outlet_reliability import get_outlet_reliability_tracker
            outlet_tracker = get_outlet_reliability_tracker()
            use_dynamic_reliability = True
        except Exception:
            use_dynamic_reliability = False
        
        for source in sources:
            # Phase 3: Get dynamic authority score if available
            if use_dynamic_reliability:
                domain_trust = outlet_tracker.get_outlet_authority(source.domain) / 100.0
                self.logger.debug(f"Using dynamic authority for {source.domain}: {domain_trust}")
            else:
                # Fallback to static trusted domains list
                domain_trust = self.TRUSTED_DOMAINS.get(source.domain, 0.5)
            
            # Apply bias penalties
            bias_penalty = len(source.bias_flags) * 0.1
            adjusted_trust = max(0.0, domain_trust - bias_penalty)
            
            trust_scores.append(adjusted_trust)
        
        return sum(trust_scores) / len(trust_scores)
    
    def _calculate_semantic_alignment(self, claim: str, evidence_texts: List[str]) -> float:
        """
        Calculate how well evidence supports the claim using semantic similarity.
        
        Uses sentence-transformers for advanced semantic matching if available,
        otherwise falls back to keyword-based matching.
        
        Args:
            claim: The statement being verified
            evidence_texts: List of evidence snippets from sources
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not evidence_texts:
            # Return minimal score instead of 0.0 for no evidence
            # This prevents complete score collapse when evidence is unavailable
            self.logger.warning("No evidence texts provided for semantic alignment")
            return 0.3  # Baseline score when no evidence available
        
        # Filter out empty or whitespace-only evidence
        valid_evidence = [e.strip() for e in evidence_texts if e and e.strip()]
        if not valid_evidence:
            self.logger.warning("All evidence texts are empty")
            return 0.3  # Baseline score
        
        # Use advanced semantic matching if model is loaded
        if self.semantic_model is not None:
            return self._semantic_similarity_transformer(claim, valid_evidence)
        else:
            return self._semantic_similarity_keyword(claim, valid_evidence)
    
    def _semantic_similarity_transformer(self, claim: str, evidence_texts: List[str]) -> float:
        """
        Advanced semantic matching using sentence transformers.
        Computes cosine similarity between claim and evidence embeddings.
        """
        try:
            # Encode claim
            claim_embedding = self.semantic_model.encode([claim], convert_to_tensor=False)[0]
            
            # Encode all evidence texts
            evidence_embeddings = self.semantic_model.encode(evidence_texts, convert_to_tensor=False)
            
            # Calculate cosine similarity for each evidence
            similarities = []
            for evidence_embedding in evidence_embeddings:
                # Cosine similarity
                similarity = _numpy.dot(claim_embedding, evidence_embedding) / (
                    _numpy.linalg.norm(claim_embedding) * _numpy.linalg.norm(evidence_embedding)
                )
                # Convert from [-1, 1] to [0, 1]
                similarity = (similarity + 1) / 2
                similarities.append(similarity)
            
            # Return average similarity
            avg_similarity = sum(similarities) / len(similarities)
            
            self.logger.debug(f"Semantic similarity (transformer): {avg_similarity:.3f}")
            return float(avg_similarity)
            
        except Exception as e:
            self.logger.error(f"Error in semantic similarity: {e}")
            return self._semantic_similarity_keyword(claim, evidence_texts)
    
    def _semantic_similarity_keyword(self, claim: str, evidence_texts: List[str]) -> float:
        """
        Fallback keyword-based semantic matching.
        Uses Jaccard similarity with term frequency weighting.
        """
        claim_words = set(claim.lower().split())
        alignments = []
        
        for evidence in evidence_texts:
            evidence_words = set(evidence.lower().split())
            
            # Jaccard similarity
            intersection = len(claim_words.intersection(evidence_words))
            union = len(claim_words.union(evidence_words))
            
            if union > 0:
                similarity = intersection / union
                alignments.append(similarity)
        
        avg_alignment = sum(alignments) / len(alignments) if alignments else 0.0
        self.logger.debug(f"Semantic alignment (keyword): {avg_alignment:.3f}")
        return avg_alignment
    
    def _calculate_temporal_consistency(self, sources: List[Source]) -> float:
        """
        Check if sources are recent and consistent across time.
        """
        if not sources:
            self.logger.warning("No sources provided for temporal consistency")
            # Return moderate score instead of 0 when no temporal data
            return 0.6
        
        # Check recency (sources within last 30 days get higher score)
        now = datetime.now()
        recency_scores = []
        
        for source in sources:
            if source.timestamp:
                days_old = (now - source.timestamp).days
                if days_old <= 30:
                    recency_score = 1.0
                elif days_old <= 90:
                    recency_score = 0.7
                elif days_old <= 365:
                    recency_score = 0.5
                else:
                    recency_score = 0.3
                recency_scores.append(recency_score)
        
        return sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
    
    def _calculate_evidence_diversity(self, sources: List[Source]) -> float:
        """
        Measure diversity of sources (different domains, perspectives).
        """
        if not sources:
            self.logger.warning("No sources provided for diversity calculation")
            # Return low-moderate score instead of 0 when no sources
            return 0.4
        
        # Count unique domains
        unique_domains = len(set(source.domain for source in sources if source.domain))
        
        # If no valid domains, return baseline
        if unique_domains == 0:
            return 0.4
        
        # Normalize by number of sources (diminishing returns)
        diversity_score = min(1.0, unique_domains / max(3, len(sources) * 0.6))
        
        return diversity_score
    
    def _generate_explanation(
        self,
        overall: float,
        trust: float,
        semantic: float,
        temporal: float,
        diversity: float
    ) -> str:
        """Generate human-readable explanation of the score."""
        
        parts = []
        parts.append(f"Overall credibility: {overall:.1%}")
        parts.append(f"Source trustworthiness: {trust:.1%}")
        parts.append(f"Evidence alignment: {semantic:.1%}")
        parts.append(f"Temporal consistency: {temporal:.1%}")
        parts.append(f"Source diversity: {diversity:.1%}")
        
        # Add interpretation
        if overall >= 0.75:
            parts.append("\n✅ This claim is strongly supported by credible evidence.")
        elif overall >= 0.5:
            parts.append("\n⚠️ This claim has moderate support but may need additional verification.")
        else:
            parts.append("\n❌ This claim lacks sufficient credible evidence.")
        
        return " | ".join(parts)


    # =========================================================================
    # ATLAS v3 ENHANCED 7-AXIS SCORING SYSTEM
    # =========================================================================
    
    def _calculate_credibility_v3(
        self,
        claim: str,
        sources: List[Source],
        evidence_texts: List[str],
        **kwargs
    ) -> CredibilityScore:
        """
        ATLAS v3 enhanced scoring with 7 axes
        
        Additional kwargs:
        - sig_result: Source Independence Graph result
        - pr_results: List of PR detection results
        - rag_hits: Dict with factchecker/RAG matches
        """
        warnings = []
        
        # 1. Domain Reliability (dynamic - based on historical performance)
        domain_reliability = self._calculate_domain_reliability(sources)
        if domain_reliability < 0.3:
            warnings.append("Low domain reliability")
        
        # 2. Ecosystem Cross-Verification
        ecosystem_score = self._calculate_ecosystem_verification(sources)
        if ecosystem_score < 0.4:
            warnings.append("Limited ecosystem coverage")
        
        # 3. Temporal Consistency (same as v2)
        temporal_consistency = self._calculate_temporal_consistency(sources)
        if temporal_consistency < 0.5:
            warnings.append("Temporal inconsistencies detected")
        
        # 4. Source Independence (from SIG Module B)
        sig_result = kwargs.get('sig_result')
        source_independence = self._extract_independence_score(sig_result)
        if source_independence < 0.4:
            warnings.append("Low source independence - possible syndication")
        
        # 5. Sentiment & Framing Analysis
        pr_results = kwargs.get('pr_results', [])
        sentiment_framing = self._calculate_sentiment_framing(evidence_texts, pr_results)
        if sentiment_framing < 0.5:
            warnings.append("Manipulative framing or PR-speak detected")
        
        # 6. Evidence Diversity (enhanced)
        evidence_diversity = self._calculate_evidence_diversity(sources)
        if evidence_diversity < 0.3:
            warnings.append("Limited source diversity")
        
        # 7. Fact-checker / RAG Hit
        rag_hits = kwargs.get('rag_hits', {})
        factchecker_score = self._calculate_factchecker_score(rag_hits)
        
        # Calculate weighted overall score (v3)
        overall_score = (
            self.WEIGHTS_V3['domain_reliability'] * domain_reliability +
            self.WEIGHTS_V3['ecosystem_verification'] * ecosystem_score +
            self.WEIGHTS_V3['temporal_consistency'] * temporal_consistency +
            self.WEIGHTS_V3['source_independence'] * source_independence +
            self.WEIGHTS_V3['sentiment_framing'] * sentiment_framing +
            self.WEIGHTS_V3['evidence_diversity'] * evidence_diversity +
            self.WEIGHTS_V3['factchecker_rag'] * factchecker_score
        )
        
        # Determine confidence level
        if overall_score >= 0.75:
            confidence = "High"
        elif overall_score >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        explanation = self._generate_explanation_v3(
            overall_score, domain_reliability, ecosystem_score,
            temporal_consistency, source_independence, sentiment_framing,
            evidence_diversity, factchecker_score
        )
        
        self.logger.info(f"Credibility Score (v3): {overall_score:.2f} ({confidence} confidence)")
        
        return CredibilityScore(
            overall_score=overall_score,
            source_trust=domain_reliability,  # Map to existing field
            semantic_alignment=ecosystem_score,  # Map to existing field
            temporal_consistency=temporal_consistency,
            evidence_diversity=evidence_diversity,
            confidence_level=confidence,
            explanation=explanation,
            warnings=warnings
        )
    
    def _calculate_domain_reliability(self, sources: List[Source]) -> float:
        """
        Calculate dynamic domain reliability (can track historical performance)
        For now, uses trusted domain list
        """
        return self._calculate_source_trust(sources)
    
    def _calculate_ecosystem_verification(self, sources: List[Source]) -> float:
        """
        Check coverage across different media ecosystems:
        - Local reporters
        - Global outlets
        - Wire services
        - Regional languages
        - Citizen evidence
        - OSINT data
        """
        if not sources:
            return 0.3
        
        ecosystems_covered = set()
        
        for source in sources:
            domain = source.domain.lower()
            
            # Classify by ecosystem
            if any(wire in domain for wire in ['reuters', 'ap', 'afp', 'upi']):
                ecosystems_covered.add('wire')
            elif any(global_outlet in domain for global_outlet in ['bbc', 'cnn', 'aljazeera']):
                ecosystems_covered.add('global')
            elif any(local in domain for local in ['.local', 'patch', 'city']):
                ecosystems_covered.add('local')
            elif domain.endswith(('.in', '.pk', '.bd', '.lk')):  # Regional TLDs
                ecosystems_covered.add('regional')
            else:
                ecosystems_covered.add('other')
        
        # Score based on ecosystem diversity (max 5 types)
        score = len(ecosystems_covered) / 5.0
        return min(1.0, score)
    
    def _extract_independence_score(self, sig_result) -> float:
        """Extract independence index from SIG result"""
        if not sig_result:
            return 0.5  # Neutral when SIG not available
        
        # If sig_result is dict
        if isinstance(sig_result, dict):
            return sig_result.get('independence_index', 0.5)
        
        # If sig_result is SIGResult object
        if hasattr(sig_result, 'independence_index'):
            return sig_result.independence_index
        
        return 0.5
    
    def _calculate_sentiment_framing(
        self,
        evidence_texts: List[str],
        pr_results: List[Dict]
    ) -> float:
        """
        Detect manipulative framing, sensationalism, PR-speak
        
        Uses PR detection results to identify problematic framing
        """
        if not evidence_texts:
            return 0.5
        
        # Start with neutral score
        score = 0.7
        
        # Penalize if PR detection flagged issues
        if pr_results:
            avg_pr_score = sum(pr.get('pr_score', 0) for pr in pr_results) / len(pr_results)
            # Normalize (0-100 to 0-1)
            pr_penalty = (avg_pr_score / 100) * 0.4
            score -= pr_penalty
        
        # Check for sensational language
        sensational_patterns = [
            r'\b(?:shocking|devastating|catastrophic|unbelievable)\b',
            r'\b(?:must see|you won\'t believe|breaking)\b',
            r'!!!|!!!!'
        ]
        
        combined_text = ' '.join(evidence_texts).lower()
        sensational_count = sum(1 for pattern in sensational_patterns if re.search(pattern, combined_text))
        
        if sensational_count > 2:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_factchecker_score(self, rag_hits: Dict) -> float:
        """
        Strong boost/penalty based on factchecker and RAG hits
        
        rag_hits format:
        {
            'verified': bool,
            'debunked': bool,
            'confidence': float
        }
        """
        if not rag_hits:
            return 0.5  # Neutral when no RAG data
        
        if rag_hits.get('verified'):
            return 0.9  # Strong positive signal
        elif rag_hits.get('debunked'):
            return 0.1  # Strong negative signal
        else:
            return 0.5  # Neutral
    
    def _generate_explanation_v3(
        self,
        overall: float,
        domain: float,
        ecosystem: float,
        temporal: float,
        independence: float,
        framing: float,
        diversity: float,
        factcheck: float
    ) -> str:
        """Generate human-readable explanation for v3 scoring"""
        
        parts = []
        parts.append(f"Overall credibility (v3): {overall:.1%}")
        parts.append(f"Domain reliability: {domain:.1%}")
        parts.append(f"Ecosystem coverage: {ecosystem:.1%}")
        parts.append(f"Temporal consistency: {temporal:.1%}")
        parts.append(f"Source independence: {independence:.1%}")
        parts.append(f"Framing analysis: {framing:.1%}")
        parts.append(f"Evidence diversity: {diversity:.1%}")
        parts.append(f"Fact-checker hits: {factcheck:.1%}")
        
        # Add interpretation
        if overall >= 0.75:
            parts.append("\n✅ This claim is strongly supported by independent, credible evidence.")
        elif overall >= 0.5:
            parts.append("\n⚠️ This claim has moderate support but may need additional verification.")
        else:
            parts.append("\n❌ This claim lacks sufficient independent, credible evidence.")
        
        return " | ".join(parts)


# Standalone function for quick scoring
def score_claim_credibility(
    claim: str,
    sources: List[Dict],
    evidence_texts: List[str]
) -> Dict:
    """
    Convenience function for scoring a claim.
    
    Args:
        claim: Statement to verify
        sources: List of dicts with keys: url, domain, content, timestamp
        evidence_texts: List of evidence snippets
        
    Returns:
        Dict with credibility metrics
    """
    engine = CredibilityEngine()
    
    # Convert dicts to Source objects
    source_objects = [
        Source(
            url=s.get('url', ''),
            domain=s.get('domain', ''),
            content=s.get('content', ''),
            timestamp=s.get('timestamp', datetime.now()),
            trust_score=s.get('trust_score', 0.5),
            bias_flags=s.get('bias_flags', [])
        )
        for s in sources
    ]
    
    score = engine.calculate_credibility(claim, source_objects, evidence_texts)
    
    return {
        'overall_score': score.overall_score,
        'confidence_level': score.confidence_level,
        'source_trust': score.source_trust,
        'semantic_alignment': score.semantic_alignment,
        'temporal_consistency': score.temporal_consistency,
        'evidence_diversity': score.evidence_diversity,
        'explanation': score.explanation,
        'warnings': score.warnings
    }
