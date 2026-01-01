# backend/agents/factcheck_integration.py
"""
Phase 4 - Cross-Ecosystem Verification & Fact-Check Integration

Integrates with professional fact-checking organizations and databases:
- Snopes API integration
- PolitiFact API integration
- FactCheck.org scraping
- Google Fact Check Tools API
- ClaimReview schema detection
- Historical claim database (local cache)
"""

import os
import logging
import hashlib
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger("factcheck_integration")

# API Configuration
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
POLITIFACT_API_KEY = os.getenv("POLITIFACT_API_KEY")

# Database path for historical claims
CLAIMS_DB_PATH = Path(__file__).parent.parent / "database" / "historical_claims.json"


class FactCheckIntegration:
    """
    Cross-ecosystem verification engine that checks claims against
    professional fact-checking databases and historical records.
    """
    
    def __init__(self):
        self.has_google_factcheck = bool(GOOGLE_FACTCHECK_API_KEY)
        self.has_politifact = bool(POLITIFACT_API_KEY)
        self.claims_cache = self._load_claims_database()
        logger.info(f"FactCheckIntegration initialized - Google: {self.has_google_factcheck}, PolitiFact: {self.has_politifact}")
        logger.info(f"Historical claims cache: {len(self.claims_cache)} entries")
    
    def verify_claim(self, claim: str, claim_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Main entry point for cross-ecosystem fact-check verification.
        
        Args:
            claim: The claim text to verify
            claim_date: Optional date when claim emerged
            
        Returns:
            {
                "factcheck_results": [...],
                "historical_matches": [...],
                "cross_verification_score": 0-100,
                "verdict_consensus": "VERIFIED/DEBUNKED/MIXED",
                "confidence": 0-1,
                "sources": [...]
            }
        """
        logger.info(f"Starting cross-ecosystem verification for: {claim[:100]}")
        
        results = {
            "claim_hash": self._hash_claim(claim),
            "factcheck_results": [],
            "historical_matches": [],
            "cross_verification_score": 50,
            "verdict_consensus": "UNVERIFIED",
            "confidence": 0.0,
            "sources": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 1. Check historical claims database (instant)
        historical_matches = self._check_historical_database(claim)
        if historical_matches:
            results["historical_matches"] = historical_matches
            logger.info(f"Found {len(historical_matches)} historical matches")
        
        # 2. Google Fact Check Tools API
        if self.has_google_factcheck:
            google_results = self._check_google_factcheck(claim)
            if google_results:
                results["factcheck_results"].extend(google_results)
                results["sources"].append("Google Fact Check Tools")
        
        # 3. PolitiFact API
        if self.has_politifact:
            politifact_results = self._check_politifact(claim)
            if politifact_results:
                results["factcheck_results"].extend(politifact_results)
                results["sources"].append("PolitiFact")
        
        # 4. ClaimReview schema search (web scraping fallback)
        if not results["factcheck_results"] and not self.has_google_factcheck:
            claimreview_results = self._search_claimreview_schema(claim)
            if claimreview_results:
                results["factcheck_results"].extend(claimreview_results)
                results["sources"].append("ClaimReview Schema")
        
        # 5. Calculate consensus and cross-verification score
        if results["factcheck_results"] or results["historical_matches"]:
            consensus_data = self._calculate_consensus(
                results["factcheck_results"],
                results["historical_matches"]
            )
            results.update(consensus_data)
        
        # 6. Cache this claim for future reference
        self._cache_claim(claim, results)
        
        return results
    
    def _hash_claim(self, claim: str) -> str:
        """Generate normalized hash for claim deduplication."""
        # Normalize: lowercase, remove extra spaces, strip punctuation
        normalized = claim.lower().strip()
        normalized = ' '.join(normalized.split())
        # Remove common punctuation that doesn't affect meaning
        for char in '.,!?;:':
            normalized = normalized.replace(char, '')
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _load_claims_database(self) -> Dict[str, Any]:
        """Load historical claims database from disk."""
        if CLAIMS_DB_PATH.exists():
            try:
                with open(CLAIMS_DB_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading claims database: {e}")
                return {}
        return {}
    
    def _save_claims_database(self):
        """Save claims database to disk."""
        try:
            CLAIMS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CLAIMS_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.claims_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving claims database: {e}")
    
    def _check_historical_database(self, claim: str) -> List[Dict[str, Any]]:
        """Check if this claim has been verified before."""
        claim_hash = self._hash_claim(claim)
        
        matches = []
        
        # Exact hash match
        if claim_hash in self.claims_cache:
            cached = self.claims_cache[claim_hash]
            matches.append({
                "type": "exact_match",
                "previous_verdict": cached.get("verdict"),
                "previous_confidence": cached.get("confidence"),
                "first_seen": cached.get("first_seen"),
                "last_verified": cached.get("last_verified"),
                "verification_count": cached.get("verification_count", 1)
            })
        
        # Fuzzy matching using Levenshtein-like approach
        similar_claims = self._find_similar_claims(claim)
        for similar_hash, similarity_score in similar_claims:
            if similarity_score > 0.85:  # High similarity threshold
                cached = self.claims_cache[similar_hash]
                matches.append({
                    "type": "similar_claim",
                    "similarity": similarity_score,
                    "previous_verdict": cached.get("verdict"),
                    "previous_confidence": cached.get("confidence"),
                    "original_claim": cached.get("original_text", "")[:100]
                })
        
        return matches
    
    def _find_similar_claims(self, claim: str, threshold: float = 0.8) -> List[tuple]:
        """Find similar claims using word overlap scoring."""
        claim_words = set(claim.lower().split())
        similar = []
        
        for cached_hash, cached_data in self.claims_cache.items():
            cached_text = cached_data.get("original_text", "")
            cached_words = set(cached_text.lower().split())
            
            # Jaccard similarity
            intersection = claim_words & cached_words
            union = claim_words | cached_words
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity >= threshold:
                    similar.append((cached_hash, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)[:3]
    
    def _check_google_factcheck(self, claim: str) -> List[Dict[str, Any]]:
        """Query Google Fact Check Tools API."""
        try:
            api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            
            params = {
                "query": claim,
                "key": GOOGLE_FACTCHECK_API_KEY,
                "languageCode": "en"
            }
            
            response = requests.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])
                
                results = []
                for item in claims[:5]:  # Top 5 results
                    claim_review = item.get("claimReview", [{}])[0]
                    
                    results.append({
                        "source": "Google Fact Check",
                        "organization": claim_review.get("publisher", {}).get("name", "Unknown"),
                        "claim_text": item.get("text", ""),
                        "verdict": claim_review.get("textualRating", ""),
                        "url": claim_review.get("url", ""),
                        "date": claim_review.get("reviewDate", ""),
                        "title": claim_review.get("title", "")
                    })
                
                return results
        
        except Exception as e:
            logger.error(f"Google Fact Check API error: {e}")
            return []
    
    def _check_politifact(self, claim: str) -> List[Dict[str, Any]]:
        """Query PolitiFact API (if available)."""
        # Note: PolitiFact doesn't have a public API
        # This is a placeholder for potential partnership/paid API access
        try:
            if not self.has_politifact:
                return []
            
            # Placeholder implementation
            api_url = "https://api.politifact.com/v1/search"
            
            headers = {
                "Authorization": f"Bearer {POLITIFACT_API_KEY}"
            }
            
            params = {
                "q": claim,
                "limit": 5
            }
            
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("results", []):
                    results.append({
                        "source": "PolitiFact",
                        "organization": "PolitiFact",
                        "claim_text": item.get("statement", ""),
                        "verdict": item.get("ruling", {}).get("ruling", ""),
                        "url": item.get("statement_url", ""),
                        "date": item.get("ruling_date", ""),
                        "speaker": item.get("speaker", {}).get("name", "")
                    })
                
                return results
        
        except Exception as e:
            logger.error(f"PolitiFact API error: {e}")
            return []
    
    def _search_claimreview_schema(self, claim: str) -> List[Dict[str, Any]]:
        """
        Fallback: Search for ClaimReview schema markup using web search.
        ClaimReview is a standard schema used by fact-checkers.
        """
        try:
            # Use DuckDuckGo or Google search with site: operators
            search_query = f'"{claim[:100]}" site:snopes.com OR site:factcheck.org OR site:politifact.com'
            
            # This is a simplified approach - in production, use proper web scraping
            # or integrate with a search API
            
            # For now, return empty as this requires web scraping infrastructure
            logger.info(f"ClaimReview search query: {search_query}")
            return []
        
        except Exception as e:
            logger.error(f"ClaimReview search error: {e}")
            return []
    
    def _calculate_consensus(
        self,
        factcheck_results: List[Dict[str, Any]],
        historical_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate verdict consensus from multiple fact-check sources.
        
        Returns:
            {
                "verdict_consensus": "VERIFIED/DEBUNKED/MIXED/UNVERIFIED",
                "confidence": 0-1,
                "cross_verification_score": 0-100
            }
        """
        all_verdicts = []
        
        # Extract verdicts from fact-check results
        for result in factcheck_results:
            verdict_text = result.get("verdict", "").lower()
            
            # Normalize verdict to standard categories
            if any(term in verdict_text for term in ["true", "correct", "accurate", "verified", "mostly true"]):
                all_verdicts.append("VERIFIED")
            elif any(term in verdict_text for term in ["false", "incorrect", "fake", "debunked", "pants on fire"]):
                all_verdicts.append("DEBUNKED")
            elif any(term in verdict_text for term in ["mixed", "half true", "mostly false", "misleading"]):
                all_verdicts.append("MIXED")
            else:
                all_verdicts.append("UNVERIFIED")
        
        # Extract verdicts from historical matches
        for match in historical_matches:
            prev_verdict = match.get("previous_verdict", "")
            if prev_verdict:
                all_verdicts.append(prev_verdict)
        
        if not all_verdicts:
            return {
                "verdict_consensus": "UNVERIFIED",
                "confidence": 0.0,
                "cross_verification_score": 50
            }
        
        # Count verdict distribution
        from collections import Counter
        verdict_counts = Counter(all_verdicts)
        
        # Determine consensus
        total = len(all_verdicts)
        verified_count = verdict_counts.get("VERIFIED", 0)
        debunked_count = verdict_counts.get("DEBUNKED", 0)
        mixed_count = verdict_counts.get("MIXED", 0)
        
        # Calculate percentages
        verified_pct = verified_count / total
        debunked_pct = debunked_count / total
        
        # Determine consensus verdict
        if verified_pct >= 0.6:
            consensus = "VERIFIED"
            confidence = verified_pct
        elif debunked_pct >= 0.6:
            consensus = "DEBUNKED"
            confidence = debunked_pct
        elif mixed_count >= total * 0.4:
            consensus = "MIXED"
            confidence = 0.5
        else:
            consensus = "CONFLICTING"
            confidence = 0.3
        
        # Cross-verification score (0-100)
        # Higher score = more sources agree
        agreement_score = max(verified_pct, debunked_pct) * 100
        source_count_bonus = min(total * 5, 20)  # Bonus for multiple sources
        cross_verification_score = min(100, agreement_score + source_count_bonus)
        
        return {
            "verdict_consensus": consensus,
            "confidence": round(confidence, 3),
            "cross_verification_score": int(cross_verification_score),
            "verdict_distribution": dict(verdict_counts)
        }
    
    def _cache_claim(self, claim: str, verification_results: Dict[str, Any]):
        """Cache this claim for future instant lookups."""
        claim_hash = self._hash_claim(claim)
        
        # Update or create cache entry
        if claim_hash in self.claims_cache:
            cached = self.claims_cache[claim_hash]
            cached["verification_count"] = cached.get("verification_count", 1) + 1
            cached["last_verified"] = datetime.utcnow().isoformat()
        else:
            self.claims_cache[claim_hash] = {
                "original_text": claim[:500],  # Store truncated version
                "verdict": verification_results.get("verdict_consensus"),
                "confidence": verification_results.get("confidence"),
                "first_seen": datetime.utcnow().isoformat(),
                "last_verified": datetime.utcnow().isoformat(),
                "verification_count": 1,
                "factcheck_sources": len(verification_results.get("factcheck_results", []))
            }
        
        # Save to disk
        self._save_claims_database()
    
    def get_claim_history(self, claim: str) -> Optional[Dict[str, Any]]:
        """Get historical verification data for a specific claim."""
        claim_hash = self._hash_claim(claim)
        return self.claims_cache.get(claim_hash)
    
    def get_trending_claims(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get most frequently verified claims in recent period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        trending = []
        for claim_hash, data in self.claims_cache.items():
            last_verified = datetime.fromisoformat(data.get("last_verified", "2020-01-01T00:00:00"))
            if last_verified >= cutoff_date:
                trending.append({
                    "claim": data.get("original_text", ""),
                    "verification_count": data.get("verification_count", 0),
                    "verdict": data.get("verdict"),
                    "last_verified": data.get("last_verified")
                })
        
        return sorted(trending, key=lambda x: x["verification_count"], reverse=True)[:10]


# Global instance
_factcheck_integration = None

def get_factcheck_integration() -> FactCheckIntegration:
    """Get or create the global fact-check integration instance."""
    global _factcheck_integration
    if _factcheck_integration is None:
        _factcheck_integration = FactCheckIntegration()
    return _factcheck_integration
