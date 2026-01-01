"""
PR & Propaganda Detection Engine (Module A) for ATLAS v3

Detects:
- Press release origins
- Syndicated/copy-paste content
- Narrative alignment with talking points
- Missing journalism markers
- Benefit analysis (cui bono)
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PRDetectionResult:
    """Results from PR/propaganda detection analysis"""
    pr_score: float  # 0-100 (higher = more likely PR/propaganda)
    pr_flags: List[str]
    narrative_origin_actor: Optional[str]
    confidence: float
    explanation: str
    syndication_cluster_id: Optional[str]
    journalism_markers_missing: List[str]
    benefit_actors: List[str]


class PRDetectionEngine:
    """
    Detects press releases and propaganda disguised as journalism.
    
    Key Detection Methods:
    1. Press Release Origin Detection - vocabulary/boilerplate patterns
    2. Syndication Detection - identical content across outlets
    3. Narrative Alignment - matches political/corporate talking points
    4. Missing Journalism Markers - lacks real reporting indicators
    5. Benefit Analysis - identifies who benefits from the narrative
    """
    
    # Press release vocabulary patterns
    PR_BOILERPLATE_PATTERNS = [
        r'\b(?:announced today|announced yesterday|pleased to announce)\b',
        r'\b(?:industry-leading|market-disrupting|groundbreaking solution)\b',
        r'\b(?:according to a press release|in a statement|company spokesman)\b',
        r'\b(?:for more information|visit our website|contact:|media contact)\b',
        r'\b(?:forward-looking statements|safe harbor)\b',
        r'\b(?:about [A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd))\b',
        r'\b(?:headquartered in|founded in \d{4})\b',
    ]
    
    # PR news wire services
    PR_WIRE_DOMAINS = {
        'prnewswire.com', 'businesswire.com', 'globenewswire.com',
        'marketwired.com', 'prlog.org', 'pressrelease-distribution.com',
        'ani.in', 'pti.in', 'ians.in'  # Indian wire services
    }
    
    # Corporate/political framing patterns
    NARRATIVE_PATTERNS = {
        'corporate': [
            r'\b(?:innovation|disruptive|synergy|leverage|ecosystem)\b',
            r'\b(?:stakeholders|shareholders|value proposition)\b',
            r'\b(?:sustainable growth|market opportunity)\b'
        ],
        'political': [
            r'\b(?:radical left|far-right|socialist agenda)\b',
            r'\b(?:threat to democracy|protecting freedom)\b',
            r'\b(?:war on [a-z]+|crisis at the border)\b'
        ],
        'military': [
            r'\b(?:surgical strike|neutralized threat|defensive measures)\b',
            r'\b(?:peacekeeping operation|strategic deterrence)\b'
        ]
    }
    
    # Real journalism markers that should be present
    JOURNALISM_MARKERS = {
        'direct_quotes': r'"[^"]+"(?:\s+(?:said|stated|told|explained|argued)\s+[A-Z])',
        'attribution': r'according to|sources say|told [A-Z]|confirmed by',
        'multi_source': r'(?:sources|officials|experts)\s+(?:said|told|confirmed)',
        'location_timestamp': r'[A-Z][a-z]+,\s+[A-Z][a-z]+\s+\d{1,2}',
        'counter_position': r'(?:however|critics|opponents|meanwhile)',
        'background_context': r'(?:background|context|historically|previously)'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_pr_propaganda(
        self,
        text: str,
        title: str,
        domain: str,
        url: str,
        additional_context: Optional[Dict] = None
    ) -> PRDetectionResult:
        """
        Main detection method - analyzes article for PR/propaganda markers
        
        Args:
            text: Article text content
            title: Article title
            domain: Source domain
            url: Article URL
            additional_context: Optional metadata (author, publish_date, etc.)
            
        Returns:
            PRDetectionResult with detailed analysis
        """
        if not text or len(text) < 100:
            return self._create_empty_result("Insufficient content")
        
        pr_flags = []
        narrative_actors = []
        missing_markers = []
        benefit_actors = []
        
        # 1. Check for press release origin
        pr_origin_score = self._detect_pr_origin(text, title, domain, url)
        if pr_origin_score > 0.5:
            pr_flags.append("press_release_like")
        
        # 2. Check for corporate/political framing
        narrative_score, detected_narrative = self._detect_narrative_framing(text)
        if narrative_score > 0.3:
            pr_flags.append(f"{detected_narrative}_framing")
            if detected_narrative == 'corporate':
                narrative_actors.append("Corporate PR")
            elif detected_narrative == 'political':
                narrative_actors.append("Political Actor")
        
        # 3. Check for missing journalism markers
        journalism_score, missing = self._check_journalism_markers(text)
        missing_markers = missing
        if journalism_score < 0.4:
            pr_flags.append("missing_journalism_markers")
        
        # 4. Benefit analysis
        benefits = self._analyze_benefit(text, title, additional_context)
        benefit_actors = benefits
        if benefits:
            pr_flags.append("clear_beneficiary")
        
        # Calculate overall PR score (0-100)
        pr_score = self._calculate_pr_score(
            pr_origin_score,
            narrative_score,
            journalism_score,
            len(benefit_actors)
        )
        
        # Determine confidence
        confidence = self._calculate_confidence(pr_flags, text)
        
        # Generate explanation
        explanation = self._generate_explanation(
            pr_score, pr_flags, narrative_actors, missing_markers, benefit_actors
        )
        
        # Syndication cluster ID (for grouping identical content)
        cluster_id = self._generate_cluster_id(text, title)
        
        return PRDetectionResult(
            pr_score=pr_score,
            pr_flags=pr_flags,
            narrative_origin_actor=narrative_actors[0] if narrative_actors else None,
            confidence=confidence,
            explanation=explanation,
            syndication_cluster_id=cluster_id,
            journalism_markers_missing=missing_markers,
            benefit_actors=benefit_actors
        )
    
    def analyze_content(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Public API: Analyze content for PR characteristics
        
        Args:
            content: Text content to analyze
            sources: List of source dictionaries with metadata OR list of URLs
            
        Returns:
            Dict with PR analysis results
        """
        if not sources:
            return self._create_empty_result()
        
        # Handle both dict and string inputs
        if isinstance(sources[0], str):
            # Convert URLs to dict format
            primary_source = {
                'url': sources[0],
                'domain': sources[0].split('/')[2] if '/' in sources[0] else '',
                'title': '',
                'text': content
            }
        else:
            primary_source = sources[0]
        
        title = primary_source.get('title', '')
        domain = primary_source.get('domain', '')
        url = primary_source.get('url', '')
        
        # Run all detection methods
        pr_score = self._detect_pr_origin(content, title, domain, url)
        narrative_score, narrative_type = self._detect_narrative_framing(content)
        journalism_score, journalism_missing = self._check_journalism_markers(content)
        
        # Build narrative analysis dict
        narrative_analysis = {
            'framing_score': narrative_score,
            'narrative_type': narrative_type,
            'actors': [narrative_type] if narrative_type != 'none' else [],
            'beneficiaries': []
        }
        
        # Build journalism check dict
        journalism_check = {
            'score': journalism_score,
            'missing_markers': journalism_missing,
            'missing_count': len(journalism_missing)
        }
        
        # Build PR flags
        pr_flags = []
        if pr_score > 0.5:
            pr_flags.append("Press Release Origin Detected")
        if narrative_score > 0.6:
            pr_flags.append("Narrative Framing Detected")
        if len(journalism_missing) >= 2:
            pr_flags.append("Journalistic Standards Missing")
        
        # Calculate confidence
        confidence = self._calculate_confidence(pr_flags, content)
        
        # Extract narrative actors
        narrative_actors = narrative_analysis.get('actors', [])
        benefit_actors = narrative_analysis.get('beneficiaries', [])
        
        # Generate syndication cluster
        cluster_id = self._generate_cluster_id(title, content)
        
        # Build explanation
        explanation = f"PR Score: {pr_score:.2f}, Narrative Framing: {narrative_analysis.get('framing_score', 0):.2f}"
        
        return {
            'is_pr': pr_score > 0.5,
            'pr_score': pr_score,
            'pr_flags': pr_flags,
            'indicators': pr_flags,
            'narrative_origin_actor': narrative_actors[0] if narrative_actors else None,
            'confidence': confidence,
            'explanation': explanation,
            'syndication_cluster_id': cluster_id,
            'journalism_markers_missing': journalism_check.get('missing_markers', []),
            'benefit_actors': benefit_actors
        }
    
    def _detect_pr_origin(self, text: str, title: str, domain: str, url: str) -> float:
        """
        Detect if content originates from a press release
        
        Returns:
            Score between 0.0 and 1.0 (higher = more PR-like)
        """
        score = 0.0
        
        # Check if domain is a known PR wire service
        if domain in self.PR_WIRE_DOMAINS:
            self.logger.info(f"PR wire service detected: {domain}")
            return 0.95
        
        # Check for boilerplate patterns
        boilerplate_matches = 0
        for pattern in self.PR_BOILERPLATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                boilerplate_matches += 1
        
        # More matches = higher PR score
        if boilerplate_matches >= 3:
            score += 0.6
        elif boilerplate_matches >= 2:
            score += 0.4
        elif boilerplate_matches >= 1:
            score += 0.2
        
        # Check for corporate tone in title
        corporate_title_patterns = [
            r'announces', r'launches', r'unveils', r'introduces',
            r'reports.*results', r'completes.*transaction'
        ]
        if any(re.search(p, title, re.IGNORECASE) for p in corporate_title_patterns):
            score += 0.2
        
        # Check for "About Company" boilerplate section
        if re.search(r'\n+About [A-Z][a-z]+(?:\s+Inc|\s+Corp|\s+LLC)?', text):
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_narrative_framing(self, text: str) -> tuple[float, str]:
        """
        Detect if text uses specific narrative framing
        
        Returns:
            (score, narrative_type) where score is 0-1 and type is the detected narrative
        """
        max_score = 0.0
        detected_narrative = "none"
        
        for narrative_type, patterns in self.NARRATIVE_PATTERNS.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            score = min(1.0, matches * 0.15)
            
            if score > max_score:
                max_score = score
                detected_narrative = narrative_type
        
        return max_score, detected_narrative
    
    def _check_journalism_markers(self, text: str) -> tuple[float, List[str]]:
        """
        Check for presence of real journalism markers
        
        Returns:
            (score, missing_markers) where score is 0-1 and missing_markers is list of absent elements
        """
        present_markers = []
        missing_markers = []
        
        for marker_name, pattern in self.JOURNALISM_MARKERS.items():
            if re.search(pattern, text, re.IGNORECASE):
                present_markers.append(marker_name)
            else:
                missing_markers.append(marker_name)
        
        # Score = percentage of markers present
        score = len(present_markers) / len(self.JOURNALISM_MARKERS)
        
        return score, missing_markers
    
    def _analyze_benefit(
        self,
        text: str,
        title: str,
        additional_context: Optional[Dict]
    ) -> List[str]:
        """
        Identify who benefits if the story is believed
        
        Returns:
            List of potential beneficiary actors
        """
        beneficiaries = []
        
        # Extract company names (simple heuristic)
        company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Inc|Corp|LLC|Ltd|Company)\b'
        companies = re.findall(company_pattern, text)
        if companies:
            beneficiaries.extend([f"{comp} (Corporate)" for comp in set(companies[:3])])
        
        # Political actors
        political_entities = [
            'government', 'administration', 'party', 'campaign',
            'senator', 'congressman', 'president', 'prime minister'
        ]
        for entity in political_entities:
            if entity in text.lower():
                beneficiaries.append(f"{entity.title()} (Political)")
                break
        
        # Military/defense
        military_terms = ['military', 'defense', 'armed forces', 'pentagon']
        if any(term in text.lower() for term in military_terms):
            beneficiaries.append("Military/Defense (Institutional)")
        
        return list(set(beneficiaries))[:5]  # Return top 5 unique
    
    def _calculate_pr_score(
        self,
        pr_origin_score: float,
        narrative_score: float,
        journalism_score: float,
        num_beneficiaries: int
    ) -> float:
        """
        Calculate overall PR/propaganda score (0-100)
        
        Formula weighs different factors:
        - PR origin: 40%
        - Narrative framing: 20%
        - Missing journalism: 25%
        - Clear beneficiaries: 15%
        """
        # Invert journalism score (missing markers increases PR score)
        journalism_penalty = 1.0 - journalism_score
        
        # Beneficiary score (normalize to 0-1)
        beneficiary_score = min(1.0, num_beneficiaries * 0.25)
        
        # Weighted combination
        raw_score = (
            pr_origin_score * 0.40 +
            narrative_score * 0.20 +
            journalism_penalty * 0.25 +
            beneficiary_score * 0.15
        )
        
        # Convert to 0-100 scale
        return round(raw_score * 100, 1)
    
    def _calculate_confidence(self, pr_flags: List[str], text: str) -> float:
        """
        Calculate confidence in the detection
        
        Returns:
            Score between 0.0 and 1.0
        """
        # More flags = higher confidence
        flag_confidence = min(1.0, len(pr_flags) * 0.2)
        
        # Longer text = higher confidence in analysis
        length_confidence = min(1.0, len(text) / 2000)
        
        # Average the two
        return round((flag_confidence + length_confidence) / 2, 2)
    
    def _generate_explanation(
        self,
        pr_score: float,
        pr_flags: List[str],
        narrative_actors: List[str],
        missing_markers: List[str],
        benefit_actors: List[str]
    ) -> str:
        """Generate human-readable explanation of the detection"""
        
        parts = [f"PR/Propaganda Score: {pr_score:.1f}/100"]
        
        if pr_score >= 70:
            parts.append("⚠️ HIGH likelihood of PR/propaganda content")
        elif pr_score >= 40:
            parts.append("⚠️ MODERATE indicators of PR influence")
        else:
            parts.append("✅ LOW PR/propaganda indicators")
        
        if pr_flags:
            parts.append(f"Detected patterns: {', '.join(pr_flags)}")
        
        if narrative_actors:
            parts.append(f"Narrative origin: {', '.join(narrative_actors)}")
        
        if missing_markers:
            parts.append(f"Missing journalism markers: {', '.join(missing_markers[:3])}")
        
        if benefit_actors:
            parts.append(f"Potential beneficiaries: {', '.join(benefit_actors[:3])}")
        
        return " | ".join(parts)
    
    def _generate_cluster_id(self, text: str, title: str) -> str:
        """
        Generate cluster ID for syndication detection
        Uses hash of normalized text for deduplication
        """
        # Normalize: lowercase, remove extra whitespace
        normalized = re.sub(r'\s+', ' ', f"{title} {text[:500]}".lower()).strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _create_empty_result(self, reason: str) -> PRDetectionResult:
        """Create empty result when detection cannot be performed"""
        return PRDetectionResult(
            pr_score=0.0,
            pr_flags=[],
            narrative_origin_actor=None,
            confidence=0.0,
            explanation=f"Detection not performed: {reason}",
            syndication_cluster_id=None,
            journalism_markers_missing=[],
            benefit_actors=[]
        )


def detect_pr_content(
    text: str,
    title: str = "",
    domain: str = "",
    url: str = "",
    **kwargs
) -> Dict:
    """
    Convenience function for quick PR detection
    
    Returns:
        Dict with PR detection results
    """
    engine = PRDetectionEngine()
    result = engine.detect_pr_propaganda(text, title, domain, url, kwargs)
    
    return {
        'pr_score': result.pr_score,
        'pr_flags': result.pr_flags,
        'narrative_origin': result.narrative_origin_actor,
        'confidence': result.confidence,
        'explanation': result.explanation,
        'cluster_id': result.syndication_cluster_id,
        'missing_journalism_markers': result.journalism_markers_missing,
        'beneficiaries': result.benefit_actors
    }
