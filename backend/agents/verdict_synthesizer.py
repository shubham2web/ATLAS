"""
ATLAS v3 Module D: Reasoning-Rich Verdict Synthesizer
Generates structured 4-part verdicts with transparent reasoning.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class VerdictSynthesizer:
    """
    Synthesizes comprehensive verdicts from all analysis modules.
    
    Output Structure:
    1. Verdict: Clear determination (True/False/Misleading/Unverifiable)
    2. Evidence Reasoning: Why this verdict was reached
    3. PR/Bias Explanations: Detection of propaganda and biases
    4. Confidence Score: Overall confidence level
    """
    
    def __init__(self):
        self.verdict_types = {
            "TRUE": "The claim is supported by credible evidence",
            "FALSE": "The claim is contradicted by credible evidence",
            "MISLEADING": "The claim contains partial truths but misleads",
            "UNVERIFIABLE": "Insufficient evidence to determine truth",
            "PR_CONTENT": "Content appears to be press release or propaganda"
        }
        
        self.confidence_levels = {
            "HIGH": (0.8, 1.0),
            "MEDIUM": (0.5, 0.8),
            "LOW": (0.0, 0.5)
        }
    
    def synthesize_verdict(
        self,
        claim: str,
        evidence_data: List[Dict[str, Any]],
        credibility_scores: Dict[str, Any],
        pr_analysis: Dict[str, Any],
        sig_analysis: Dict[str, Any],
        bias_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive reasoning-rich verdict.
        
        Args:
            claim: The claim being analyzed
            evidence_data: Gathered evidence from sources
            credibility_scores: 7-axis credibility scores
            pr_analysis: PR detection results
            sig_analysis: Source independence graph results
            bias_analysis: Bias detection results
            
        Returns:
            Structured verdict with 4 components
        """
        logger.info(f"Synthesizing verdict for claim: {claim[:100]}...")
        
        # Part 1: Determine Primary Verdict
        primary_verdict = self._determine_verdict(
            evidence_data, 
            credibility_scores, 
            pr_analysis
        )
        
        # Part 2: Evidence Reasoning
        evidence_reasoning = self._generate_evidence_reasoning(
            evidence_data,
            credibility_scores,
            sig_analysis
        )
        
        # Part 3: PR/Bias Explanations
        pr_bias_explanations = self._generate_pr_bias_explanations(
            pr_analysis,
            sig_analysis,
            bias_analysis
        )
        
        # Part 4: Confidence Score
        confidence = self._calculate_confidence(
            evidence_data,
            credibility_scores,
            sig_analysis
        )
        
        return {
            "claim": claim,
            "verdict": {
                "determination": primary_verdict["type"],
                "summary": primary_verdict["summary"],
                "confidence_level": confidence["level"],
                "confidence_score": confidence["score"]
            },
            "evidence_reasoning": evidence_reasoning,
            "pr_bias_analysis": pr_bias_explanations,
            "confidence_breakdown": confidence["breakdown"],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sources_analyzed": len(evidence_data),
                "independent_sources": sig_analysis.get("independent_source_count", 0),
                "pr_detected": pr_analysis.get("is_pr_content", False)
            }
        }
    
    def _determine_verdict(
        self,
        evidence_data: List[Dict[str, Any]],
        credibility_scores: Dict[str, Any],
        pr_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Determine the primary verdict type."""
        
        # Check if content is primarily PR/propaganda
        if pr_analysis.get("is_pr_content") and pr_analysis.get("confidence", 0) > 0.7:
            return {
                "type": "PR_CONTENT",
                "summary": self.verdict_types["PR_CONTENT"]
            }
        
        # Get overall credibility score
        # credibility_scores is a CredibilityScore object, not a dict
        overall_score = credibility_scores.overall_score if hasattr(credibility_scores, 'overall_score') else 0
        
        # Analyze evidence consensus
        if not evidence_data or len(evidence_data) == 0:
            return {
                "type": "UNVERIFIABLE",
                "summary": self.verdict_types["UNVERIFIABLE"]
            }
        
        # Count supporting vs contradicting evidence
        supporting = sum(1 for e in evidence_data if e.get("supports_claim", False))
        contradicting = sum(1 for e in evidence_data if e.get("contradicts_claim", False))
        total = len(evidence_data)
        
        support_ratio = supporting / total if total > 0 else 0
        contradict_ratio = contradicting / total if total > 0 else 0
        
        # Determine verdict based on evidence and credibility
        if overall_score >= 0.7 and support_ratio >= 0.6:
            return {
                "type": "TRUE",
                "summary": self.verdict_types["TRUE"]
            }
        elif overall_score < 0.4 or contradict_ratio >= 0.6:
            return {
                "type": "FALSE",
                "summary": self.verdict_types["FALSE"]
            }
        elif support_ratio > 0.3 and contradict_ratio > 0.3:
            return {
                "type": "MISLEADING",
                "summary": self.verdict_types["MISLEADING"]
            }
        else:
            return {
                "type": "UNVERIFIABLE",
                "summary": self.verdict_types["UNVERIFIABLE"]
            }
    
    def _generate_evidence_reasoning(
        self,
        evidence_data: List[Dict[str, Any]],
        credibility_scores: Dict[str, Any],
        sig_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed reasoning from evidence with bullet points."""
        
        # Extract key evidence points
        supporting_evidence = [
            e for e in evidence_data 
            if e.get("supports_claim", False)
        ]
        
        # Build explanation bullet points
        explanation_points = []
        
        # Evidence quantity
        num_sources = len(evidence_data)
        if num_sources == 0:
            explanation_points.append("No evidence sources found; claim cannot be verified.")
        elif num_sources == 1:
            explanation_points.append(f"Only 1 supporting article found; evidence quantity low.")
        elif num_sources < 3:
            explanation_points.append(f"Limited evidence found ({num_sources} sources); low diversity.")
        else:
            explanation_points.append(f"Multiple sources analyzed ({num_sources} articles).")
        
        # Credibility assessment
        overall_score = credibility_scores.overall_score if hasattr(credibility_scores, 'overall_score') else 0
        if overall_score >= 0.7:
            explanation_points.append("Source credibility high; reputable outlets detected.")
        elif overall_score >= 0.5:
            explanation_points.append(f"Source credibility moderate (~{int(overall_score*100)}%).")
        else:
            explanation_points.append(f"Source credibility low; questionable reliability.")
        
        # Independence analysis
        independence_score = sig_analysis.get("independence_score", 0)
        if independence_score >= 0.8:
            explanation_points.append("Independence score high due to unique sources.")
        elif independence_score < 0.5:
            explanation_points.append("Sources appear to share common origin; independence low.")
        
        # Consensus analysis
        if supporting_evidence:
            explanation_points.append(f"{len(supporting_evidence)} source(s) support the claim.")
        else:
            explanation_points.append("No contradictory reporting detected.")
        
        # Cross-verification
        if num_sources < 2:
            explanation_points.append("No cross-verification from independent outlets.")
        
        
        contradicting_evidence = [
            e for e in evidence_data 
            if e.get("contradicts_claim", False)
        ]
        
        # Analyze source quality
        high_quality_sources = [
            e for e in evidence_data 
            if e.get("credibility_score", 0) >= 0.7
        ]
        
        # Check source independence
        independent_sources = sig_analysis.get("independent_source_count", 0)
        total_sources = len(evidence_data)
        
        return {
            "summary": self._create_reasoning_summary(
                supporting_evidence,
                contradicting_evidence,
                high_quality_sources,
                independent_sources
            ),
            "explanation_points": explanation_points,
            "sources_analyzed": [{
                "url": e.get("url", ""),
                "domain": e.get("domain", ""),
                "title": e.get("title", "")
            } for e in evidence_data],
            "supporting_points": [
                {
                    "source": e.get("source", "Unknown"),
                    "excerpt": e.get("content", "")[:200],
                    "credibility": e.get("credibility_score", 0)
                }
                for e in supporting_evidence[:3]
            ],
            "contradicting_points": [
                {
                    "source": e.get("source", "Unknown"),
                    "excerpt": e.get("content", "")[:200],
                    "credibility": e.get("credibility_score", 0)
                }
                for e in contradicting_evidence[:3]
            ],
            "source_quality": {
                "high_quality_count": len(high_quality_sources),
                "independent_count": independent_sources,
                "total_count": total_sources,
                "independence_ratio": independent_sources / total_sources if total_sources > 0 else 0
            }
        }
    
    def _create_reasoning_summary(
        self,
        supporting: List[Dict],
        contradicting: List[Dict],
        high_quality: List[Dict],
        independent_count: int
    ) -> str:
        """Create human-readable reasoning summary."""
        
        parts = []
        
        # Evidence distribution
        if len(supporting) > len(contradicting):
            parts.append(
                f"Analysis found {len(supporting)} sources supporting the claim "
                f"versus {len(contradicting)} contradicting it."
            )
        elif len(contradicting) > len(supporting):
            parts.append(
                f"Analysis found {len(contradicting)} sources contradicting the claim "
                f"versus {len(supporting)} supporting it."
            )
        else:
            parts.append(
                f"Analysis found mixed evidence with {len(supporting)} sources on each side."
            )
        
        # Source quality
        if len(high_quality) >= 3:
            parts.append(
                f"{len(high_quality)} high-quality sources were identified, "
                "lending weight to the analysis."
            )
        
        # Source independence
        if independent_count >= 3:
            parts.append(
                f"{independent_count} truly independent sources were verified, "
                "indicating genuine cross-verification."
            )
        elif independent_count <= 1:
            parts.append(
                "Limited source independence detected - most sources may originate "
                "from the same root source or press release."
            )
        
        return " ".join(parts)
    
    def _generate_pr_bias_explanations(
        self,
        pr_analysis: Dict[str, Any],
        sig_analysis: Dict[str, Any],
        bias_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate PR and bias detection explanations."""
        
        explanations = {
            "pr_detection": {
                "is_pr": pr_analysis.get("is_pr_content", False),
                "confidence": pr_analysis.get("confidence", 0),
                "indicators": pr_analysis.get("indicators", []),
                "press_release_origin": pr_analysis.get("press_release_origin", False),
                "syndication_pattern": pr_analysis.get("syndication_detected", False)
            },
            "source_independence": {
                "independence_score": sig_analysis.get("independence_score", 0),
                "shared_origin_detected": sig_analysis.get("shared_origin_detected", False),
                "source_clusters": sig_analysis.get("source_clusters", []),
                "narrative_coordination": sig_analysis.get("coordinated_narrative", False)
            },
            "bias_detection": self._extract_bias_info(bias_analysis),
            "transparency_notes": self._generate_transparency_notes(
                pr_analysis,
                sig_analysis,
                bias_analysis
            )
        }
        
        return explanations
    
    def _extract_bias_info(self, bias_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant bias information."""
        if not bias_analysis:
            return {
                "biases_detected": [],
                "bias_count": 0,
                "bias_summary": "No bias analysis available"
            }
        
        return {
            "biases_detected": bias_analysis.get("detected_biases", []),
            "bias_count": len(bias_analysis.get("detected_biases", [])),
            "bias_summary": bias_analysis.get("summary", ""),
            "severity": bias_analysis.get("overall_severity", "low")
        }
    
    def _generate_transparency_notes(
        self,
        pr_analysis: Dict[str, Any],
        sig_analysis: Dict[str, Any],
        bias_analysis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate transparency notes for users."""
        notes = []
        
        # PR transparency
        if pr_analysis.get("is_pr_content"):
            notes.append(
                "This content shows characteristics of press release or promotional material. "
                "Claims may be biased toward positive representation."
            )
        
        # Source independence transparency
        if sig_analysis.get("independence_score", 1.0) < 0.5:
            notes.append(
                "Multiple sources appear to derive from the same origin. "
                "This reduces the reliability of cross-verification."
            )
        
        # Syndication transparency
        if pr_analysis.get("syndication_detected"):
            notes.append(
                "Syndication pattern detected - the same content appears across multiple outlets. "
                "This may indicate coordinated messaging rather than independent reporting."
            )
        
        # Bias transparency
        if bias_analysis and len(bias_analysis.get("detected_biases", [])) > 2:
            notes.append(
                f"{len(bias_analysis['detected_biases'])} cognitive biases detected in the content or sources. "
                "Exercise caution when evaluating claims."
            )
        
        return notes
    
    def _calculate_confidence(
        self,
        evidence_data: List[Dict[str, Any]],
        credibility_scores: Dict[str, Any],
        sig_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall confidence in the verdict."""
        
        # Factor 1: Evidence quantity (0-1)
        evidence_score = min(len(evidence_data) / 10, 1.0)
        
        # Factor 2: Source credibility (0-1)
        
        # credibility_scores is a CredibilityScore object, not a dict
        credibility_score = credibility_scores.overall_score if hasattr(credibility_scores, 'overall_score') else 0
                # Factor 3: Source independence (0-1)
        independence_score = sig_analysis.get("independence_score", 0)
        
        # Factor 4: Evidence consensus (0-1)
        # Fixed: No contradictory evidence = HIGH consensus, not zero
        supporting = sum(1 for e in evidence_data if e.get("supports_claim", False))
        contradicting = sum(1 for e in evidence_data if e.get("contradicts_claim", False))
        total = len(evidence_data)
        
        if total > 0:
            # If no contradictions found, consensus is high (supporting/total)
            # If contradictions exist, consensus is low
            if contradicting == 0 and supporting > 0:
                # No contradictions = strong consensus
                consensus_score = min(1.0, supporting / total + 0.3)  # Boost for unanimity
            elif contradicting == 0 and supporting == 0:
                # Neutral: no supporting or contradicting
                consensus_score = 0.5  # Neutral instead of 0
            else:
                # Mixed evidence: ratio of majority
                max_consensus = max(supporting, contradicting)
                consensus_score = max_consensus / total
        else:
            consensus_score = 0
        
        # Weighted combination
        overall_confidence = (
            evidence_score * 0.2 +
            credibility_score * 0.3 +
            independence_score * 0.25 +
            consensus_score * 0.25
        )
        
        # Determine confidence level
        confidence_level = "LOW"
        for level, (min_score, max_score) in self.confidence_levels.items():
            if min_score <= overall_confidence < max_score:
                confidence_level = level
                break
        
        return {
            "score": round(overall_confidence, 2),
            "level": confidence_level,
            "breakdown": {
                "evidence_quantity": round(evidence_score, 2),
                "source_credibility": round(credibility_score, 2),
                "source_independence": round(independence_score, 2),
                "evidence_consensus": round(consensus_score, 2)
            }
        }
    
    def format_for_display(self, verdict_data: Dict[str, Any]) -> str:
        """Format verdict for human-readable display."""
        
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("ATLAS v3 TRUTH INTELLIGENCE VERDICT")
        output.append("=" * 80)
        output.append("")
        
        # Verdict
        v = verdict_data["verdict"]
        output.append(f"VERDICT: {v['determination']}")
        output.append(f"Summary: {v['summary']}")
        output.append(f"Confidence: {v['confidence_level']} ({v['confidence_score']})")
        output.append("")
        
        # Evidence Reasoning
        output.append("-" * 80)
        output.append("EVIDENCE REASONING")
        output.append("-" * 80)
        er = verdict_data["evidence_reasoning"]
        output.append(er["summary"])
        output.append("")
        
        if er["supporting_points"]:
            output.append("Supporting Evidence:")
            for i, point in enumerate(er["supporting_points"], 1):
                output.append(f"  {i}. {point['source']} (credibility: {point['credibility']})")
                output.append(f"     {point['excerpt']}...")
        
        if er["contradicting_points"]:
            output.append("")
            output.append("Contradicting Evidence:")
            for i, point in enumerate(er["contradicting_points"], 1):
                output.append(f"  {i}. {point['source']} (credibility: {point['credibility']})")
                output.append(f"     {point['excerpt']}...")
        
        output.append("")
        
        # PR/Bias Analysis
        output.append("-" * 80)
        output.append("PR & BIAS ANALYSIS")
        output.append("-" * 80)
        pba = verdict_data["pr_bias_analysis"]
        
        pr = pba["pr_detection"]
        if pr["is_pr"]:
            output.append(f"PR Content Detected (confidence: {pr['confidence']})")
            output.append(f"Indicators: {', '.join(pr['indicators'][:5])}")
        else:
            output.append("No significant PR characteristics detected")
        
        output.append("")
        si = pba["source_independence"]
        output.append(f"Source Independence Score: {si['independence_score']}")
        if si["shared_origin_detected"]:
            output.append("  Warning: Multiple sources share common origin")
        
        output.append("")
        
        # Transparency Notes
        if pba["transparency_notes"]:
            output.append("-" * 80)
            output.append("TRANSPARENCY NOTES")
            output.append("-" * 80)
            for note in pba["transparency_notes"]:
                output.append(f"  - {note}")
        
        output.append("")
        output.append("=" * 80)
        
        return "\n".join(output)


# Standalone utility function
def synthesize_quick_verdict(
    claim: str,
    evidence: List[Dict],
    scores: Dict
) -> Dict[str, Any]:
    """Quick verdict synthesis for simple cases."""
    synthesizer = VerdictSynthesizer()
    
    # Create minimal inputs for modules not provided
    pr_analysis = {"is_pr_content": False, "confidence": 0}
    sig_analysis = {"independence_score": 0.7, "independent_source_count": len(evidence)}
    
    return synthesizer.synthesize_verdict(
        claim,
        evidence,
        scores,
        pr_analysis,
        sig_analysis
    )
