# backend/agents/outlet_reliability.py
"""
Phase 3 - Dynamic Outlet Reliability Tracker for ATLAS v3

Tracks historical accuracy of news outlets and adjusts credibility scores dynamically.
Learns from past debunks and fact-checks to build outlet reputation profiles.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("outlet_reliability")

# Database path
DB_PATH = Path(__file__).parent.parent / "database" / "outlet_reliability.json"


class OutletReliabilityTracker:
    """
    Tracks outlet performance over time and adjusts authority scores dynamically.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.reliability_db = self._load_database()
        logger.info(f"OutletReliabilityTracker initialized with {len(self.reliability_db)} outlets tracked")
    
    def _load_database(self) -> Dict[str, Any]:
        """Load outlet reliability database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading reliability DB: {e}")
                return {}
        else:
            # Initialize with baseline
            return self._initialize_baseline_db()
    
    def _save_database(self):
        """Save outlet reliability database to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.reliability_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving reliability DB: {e}")
    
    def _initialize_baseline_db(self) -> Dict[str, Any]:
        """
        Initialize with baseline authority scores for known outlets.
        These will be dynamically adjusted based on performance.
        """
        baseline = {
            # Tier 1: High Authority (Start at 85-95)
            "reuters.com": {
                "base_authority": 95,
                "current_authority": 95,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "apnews.com": {
                "base_authority": 94,
                "current_authority": 94,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "bbc.com": {
                "base_authority": 90,
                "current_authority": 90,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "nytimes.com": {
                "base_authority": 88,
                "current_authority": 88,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "theguardian.com": {
                "base_authority": 87,
                "current_authority": 87,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "washingtonpost.com": {
                "base_authority": 86,
                "current_authority": 86,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            
            # Tier 2: Medium-High Authority (Start at 70-85)
            "timesofindia.com": {
                "base_authority": 75,
                "current_authority": 75,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "cnn.com": {
                "base_authority": 75,
                "current_authority": 75,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "thehindu.com": {
                "base_authority": 78,
                "current_authority": 78,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            
            # Tier 3: Medium Authority (Start at 50-70)
            "indianexpress.com": {
                "base_authority": 70,
                "current_authority": 70,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "hindustantimes.com": {
                "base_authority": 68,
                "current_authority": 68,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            
            # Tier 4: Low-Medium Authority (Start at 40-50)
            "medium.com": {
                "base_authority": 45,
                "current_authority": 45,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            },
            "substack.com": {
                "base_authority": 40,
                "current_authority": 40,
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        return baseline
    
    def get_outlet_authority(self, domain: str) -> float:
        """
        Get current authority score for an outlet (0-100).
        Returns dynamic score if tracked, otherwise returns heuristic baseline.
        """
        # Normalize domain
        domain = self._normalize_domain(domain)
        
        if domain in self.reliability_db:
            return self.reliability_db[domain]["current_authority"]
        else:
            # Fallback to heuristic for unknown outlets
            return self._heuristic_authority(domain)
    
    def record_verdict_result(
        self,
        domain: str,
        verdict: str,
        claim: str,
        confidence: float
    ):
        """
        Record the outcome of a fact-check involving this outlet.
        Updates outlet's track record and adjusts authority.
        
        Args:
            domain: The outlet domain
            verdict: VERIFIED, DEBUNKED, MISLEADING, etc.
            claim: The claim that was checked
            confidence: Confidence in the verdict (0-1)
        """
        domain = self._normalize_domain(domain)
        
        # Initialize if not tracked
        if domain not in self.reliability_db:
            self.reliability_db[domain] = {
                "base_authority": self._heuristic_authority(domain),
                "current_authority": self._heuristic_authority(domain),
                "track_record": {"verified": 0, "debunked": 0, "mixed": 0},
                "last_updated": datetime.utcnow().isoformat(),
                "first_seen": datetime.utcnow().isoformat()
            }
        
        outlet_data = self.reliability_db[domain]
        
        # Update track record
        verdict_upper = verdict.upper()
        if verdict_upper in ["VERIFIED", "TRUE"]:
            outlet_data["track_record"]["verified"] += 1
        elif verdict_upper in ["DEBUNKED", "FALSE"]:
            outlet_data["track_record"]["debunked"] += 1
        else:
            outlet_data["track_record"]["mixed"] += 1
        
        # Recalculate authority
        outlet_data["current_authority"] = self._calculate_dynamic_authority(outlet_data)
        outlet_data["last_updated"] = datetime.utcnow().isoformat()
        
        # Save to disk
        self._save_database()
        
        logger.info(f"Updated {domain} - new authority: {outlet_data['current_authority']}")
    
    def _calculate_dynamic_authority(self, outlet_data: Dict[str, Any]) -> float:
        """
        Calculate dynamic authority based on track record.
        
        Formula:
        - Start with base authority
        - Increase for verified articles (+2 per verified, max +20)
        - Decrease for debunked articles (-5 per debunked, max -30)
        - Slight decrease for mixed/uncertain (-1 per mixed, max -10)
        """
        base = outlet_data["base_authority"]
        track = outlet_data["track_record"]
        
        verified = track["verified"]
        debunked = track["debunked"]
        mixed = track["mixed"]
        
        # Boost for verified content
        verified_boost = min(verified * 2, 20)
        
        # Penalty for debunked content
        debunked_penalty = min(debunked * 5, 30)
        
        # Small penalty for mixed/uncertain
        mixed_penalty = min(mixed * 1, 10)
        
        # Calculate new authority
        new_authority = base + verified_boost - debunked_penalty - mixed_penalty
        
        # Clamp between 0 and 100
        return max(0, min(100, new_authority))
    
    def get_outlet_stats(self, domain: str) -> Dict[str, Any]:
        """Get detailed statistics for an outlet."""
        domain = self._normalize_domain(domain)
        
        if domain not in self.reliability_db:
            return {
                "domain": domain,
                "tracked": False,
                "current_authority": self._heuristic_authority(domain),
                "note": "Not yet tracked in reliability database"
            }
        
        data = self.reliability_db[domain]
        track = data["track_record"]
        total = track["verified"] + track["debunked"] + track["mixed"]
        
        return {
            "domain": domain,
            "tracked": True,
            "base_authority": data["base_authority"],
            "current_authority": data["current_authority"],
            "track_record": track,
            "total_analyzed": total,
            "accuracy_rate": (track["verified"] / total * 100) if total > 0 else 0,
            "last_updated": data.get("last_updated"),
            "first_seen": data.get("first_seen")
        }
    
    def get_top_reliable_outlets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N most reliable outlets by current authority."""
        outlets = [
            {
                "domain": domain,
                "authority": data["current_authority"],
                "track_record": data["track_record"]
            }
            for domain, data in self.reliability_db.items()
        ]
        
        outlets.sort(key=lambda x: x["authority"], reverse=True)
        return outlets[:limit]
    
    def get_declining_outlets(self, threshold: float = -10) -> List[Dict[str, Any]]:
        """Get outlets whose authority has declined significantly."""
        declining = []
        
        for domain, data in self.reliability_db.items():
            decline = data["current_authority"] - data["base_authority"]
            if decline < threshold:
                declining.append({
                    "domain": domain,
                    "base_authority": data["base_authority"],
                    "current_authority": data["current_authority"],
                    "decline": decline,
                    "track_record": data["track_record"]
                })
        
        declining.sort(key=lambda x: x["decline"])
        return declining
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain to consistent format."""
        domain = domain.lower().strip()
        
        # Remove protocol
        domain = domain.replace("http://", "").replace("https://", "")
        
        # Remove www
        domain = domain.replace("www.", "")
        
        # Remove path
        if "/" in domain:
            domain = domain.split("/")[0]
        
        return domain
    
    def _heuristic_authority(self, domain: str) -> float:
        """
        Fallback heuristic for unknown outlets.
        Based on TLD and domain patterns.
        """
        domain_lower = domain.lower()
        
        # Known high-authority TLDs and patterns
        if any(pattern in domain_lower for pattern in [".gov", ".edu", ".ac."]):
            return 85
        
        # News agency patterns
        if any(pattern in domain_lower for pattern in ["news", "times", "post", "herald"]):
            return 65
        
        # Blog platforms
        if any(pattern in domain_lower for pattern in ["medium.com", "substack.com", "blogger", "wordpress"]):
            return 40
        
        # Social media
        if any(pattern in domain_lower for pattern in ["twitter.com", "facebook.com", "instagram.com"]):
            return 30
        
        # Default for unknown
        return 50


# Global instance
_reliability_tracker = None

def get_outlet_reliability_tracker() -> OutletReliabilityTracker:
    """Get or create the global outlet reliability tracker instance."""
    global _reliability_tracker
    if _reliability_tracker is None:
        _reliability_tracker = OutletReliabilityTracker()
    return _reliability_tracker
