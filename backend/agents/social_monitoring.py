# backend/agents/social_monitoring.py
"""
Phase 4 - Social Media Trend Analysis

Monitors social media platforms for viral claim tracking:
- Twitter/X API integration for trending topics
- Reddit monitoring for claim spread
- Viral velocity calculation
- Bot detection indicators
- Coordinated inauthentic behavior (CIB) detection
"""

import os
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

logger = logging.getLogger("social_monitoring")

# API Configuration
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")


class SocialMonitoring:
    """
    Tracks claim propagation across social media platforms.
    Detects viral spread patterns and potential bot activity.
    """
    
    def __init__(self):
        self.has_twitter = bool(TWITTER_BEARER_TOKEN)
        self.has_reddit = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
        logger.info(f"SocialMonitoring initialized - Twitter: {self.has_twitter}, Reddit: {self.has_reddit}")
    
    def analyze_social_spread(self, claim: str, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Analyze how a claim is spreading on social media.
        
        Args:
            claim: The claim text
            keywords: Optional list of keywords to track
            
        Returns:
            {
                "viral_velocity": 0-100,
                "platform_presence": {...},
                "bot_indicators": [...],
                "coordinated_behavior": {...},
                "trending_status": "viral/emerging/dormant"
            }
        """
        logger.info(f"Analyzing social spread for: {claim[:100]}")
        
        # Extract keywords if not provided
        if not keywords:
            keywords = self._extract_keywords(claim)
        
        results = {
            "viral_velocity": 0,
            "platform_presence": {},
            "bot_indicators": [],
            "coordinated_behavior": {},
            "trending_status": "dormant",
            "engagement_metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Twitter/X analysis
        if self.has_twitter:
            twitter_data = self._analyze_twitter(claim, keywords)
            results["platform_presence"]["twitter"] = twitter_data
            results["engagement_metrics"]["twitter"] = twitter_data.get("engagement", {})
        
        # Reddit analysis
        if self.has_reddit:
            reddit_data = self._analyze_reddit(claim, keywords)
            results["platform_presence"]["reddit"] = reddit_data
            results["engagement_metrics"]["reddit"] = reddit_data.get("engagement", {})
        
        # Calculate viral velocity
        results["viral_velocity"] = self._calculate_viral_velocity(
            results["platform_presence"]
        )
        
        # Detect bot activity
        results["bot_indicators"] = self._detect_bots(
            results["platform_presence"]
        )
        
        # Detect coordinated behavior
        results["coordinated_behavior"] = self._detect_coordinated_behavior(
            results["platform_presence"]
        )
        
        # Determine trending status
        results["trending_status"] = self._determine_trending_status(
            results["viral_velocity"],
            results["engagement_metrics"]
        )
        
        return results
    
    def _extract_keywords(self, claim: str, max_keywords: int = 5) -> List[str]:
        """Extract key terms from claim for social monitoring."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'be', 'been', 'being'}
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', claim.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Return unique keywords sorted by length (longer = more specific)
        unique_keywords = sorted(set(keywords), key=len, reverse=True)
        return unique_keywords[:max_keywords]
    
    def _analyze_twitter(self, claim: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze Twitter/X for claim mentions."""
        try:
            # Twitter API v2 - Recent Search
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            
            # Build search query
            query = ' OR '.join([f'"{kw}"' for kw in keywords[:3]])
            
            params = {
                "query": query,
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,author_id,lang",
                "expansions": "author_id",
                "user.fields": "created_at,public_metrics,verified"
            }
            
            headers = {
                "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get("data", [])
                users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
                
                # Analyze metrics
                total_tweets = len(tweets)
                total_likes = sum(t.get("public_metrics", {}).get("like_count", 0) for t in tweets)
                total_retweets = sum(t.get("public_metrics", {}).get("retweet_count", 0) for t in tweets)
                total_replies = sum(t.get("public_metrics", {}).get("reply_count", 0) for t in tweets)
                
                # Check for verified accounts
                verified_tweets = sum(1 for t in tweets if users.get(t.get("author_id"), {}).get("verified", False))
                
                # Calculate account age distribution (bot indicator)
                new_accounts = 0
                for tweet in tweets:
                    user = users.get(tweet.get("author_id"), {})
                    created_at = user.get("created_at")
                    if created_at:
                        account_age = datetime.utcnow() - datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if account_age < timedelta(days=90):  # New account
                            new_accounts += 1
                
                return {
                    "present": total_tweets > 0,
                    "tweet_count": total_tweets,
                    "verified_accounts": verified_tweets,
                    "new_accounts": new_accounts,
                    "engagement": {
                        "likes": total_likes,
                        "retweets": total_retweets,
                        "replies": total_replies
                    },
                    "engagement_rate": (total_likes + total_retweets) / max(total_tweets, 1)
                }
            else:
                logger.warning(f"Twitter API returned status {response.status_code}")
                return {"present": False, "error": "API error"}
        
        except Exception as e:
            logger.error(f"Twitter analysis error: {e}")
            return {"present": False, "error": str(e)}
    
    def _analyze_reddit(self, claim: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze Reddit for claim discussions."""
        try:
            # Reddit API - Search posts
            # First, get access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            auth = (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
            headers_auth = {"User-Agent": "ATLAS v3 Fact Checker"}
            
            token_response = requests.post(auth_url, auth=auth, data=auth_data, headers=headers_auth, timeout=10)
            
            if token_response.status_code != 200:
                return {"present": False, "error": "Auth failed"}
            
            access_token = token_response.json().get("access_token")
            
            # Search Reddit
            search_url = "https://oauth.reddit.com/search"
            query = ' '.join(keywords[:3])
            
            params = {
                "q": query,
                "limit": 100,
                "sort": "new",
                "t": "week"
            }
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "ATLAS v3 Fact Checker"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                total_posts = len(posts)
                total_upvotes = sum(p.get("data", {}).get("ups", 0) for p in posts)
                total_comments = sum(p.get("data", {}).get("num_comments", 0) for p in posts)
                
                # Subreddit distribution
                subreddits = defaultdict(int)
                for post in posts:
                    subreddit = post.get("data", {}).get("subreddit", "unknown")
                    subreddits[subreddit] += 1
                
                return {
                    "present": total_posts > 0,
                    "post_count": total_posts,
                    "top_subreddits": dict(sorted(subreddits.items(), key=lambda x: x[1], reverse=True)[:5]),
                    "engagement": {
                        "upvotes": total_upvotes,
                        "comments": total_comments
                    },
                    "engagement_rate": (total_upvotes + total_comments) / max(total_posts, 1)
                }
            else:
                return {"present": False, "error": "API error"}
        
        except Exception as e:
            logger.error(f"Reddit analysis error: {e}")
            return {"present": False, "error": str(e)}
    
    def _calculate_viral_velocity(self, platform_presence: Dict[str, Any]) -> int:
        """
        Calculate viral velocity score (0-100).
        Higher score = claim is spreading rapidly.
        """
        velocity = 0
        
        # Twitter contribution (max 60 points)
        twitter = platform_presence.get("twitter", {})
        if twitter.get("present"):
            tweets = twitter.get("tweet_count", 0)
            engagement_rate = twitter.get("engagement_rate", 0)
            
            velocity += min(30, tweets / 10)  # Up to 30 points for volume
            velocity += min(30, engagement_rate * 100)  # Up to 30 points for engagement
        
        # Reddit contribution (max 40 points)
        reddit = platform_presence.get("reddit", {})
        if reddit.get("present"):
            posts = reddit.get("post_count", 0)
            engagement_rate = reddit.get("engagement_rate", 0)
            
            velocity += min(20, posts / 5)  # Up to 20 points for volume
            velocity += min(20, engagement_rate / 10)  # Up to 20 points for engagement
        
        return int(min(100, velocity))
    
    def _detect_bots(self, platform_presence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect indicators of bot activity."""
        indicators = []
        
        # Twitter bot indicators
        twitter = platform_presence.get("twitter", {})
        if twitter.get("present"):
            new_accounts = twitter.get("new_accounts", 0)
            total = twitter.get("tweet_count", 1)
            new_account_ratio = new_accounts / total
            
            if new_account_ratio > 0.3:  # More than 30% from new accounts
                indicators.append({
                    "platform": "twitter",
                    "indicator": "high_new_account_ratio",
                    "severity": "medium",
                    "value": f"{new_account_ratio:.1%}",
                    "description": "Unusually high proportion of tweets from new accounts"
                })
        
        return indicators
    
    def _detect_coordinated_behavior(self, platform_presence: Dict[str, Any]) -> Dict[str, Any]:
        """Detect signs of coordinated inauthentic behavior (CIB)."""
        coordinated = {
            "detected": False,
            "confidence": 0.0,
            "signals": []
        }
        
        # Check for burst activity patterns
        twitter = platform_presence.get("twitter", {})
        if twitter.get("present"):
            tweet_count = twitter.get("tweet_count", 0)
            engagement_rate = twitter.get("engagement_rate", 0)
            
            # Unusual burst: high volume but low organic engagement
            if tweet_count > 50 and engagement_rate < 1.0:
                coordinated["signals"].append({
                    "signal": "low_engagement_high_volume",
                    "description": "High tweet volume with disproportionately low engagement"
                })
                coordinated["detected"] = True
                coordinated["confidence"] = 0.6
        
        return coordinated
    
    def _determine_trending_status(
        self,
        viral_velocity: int,
        engagement_metrics: Dict[str, Any]
    ) -> str:
        """Determine if claim is trending, emerging, or dormant."""
        if viral_velocity >= 70:
            return "viral"
        elif viral_velocity >= 40:
            return "emerging"
        elif viral_velocity >= 15:
            return "spreading"
        else:
            return "dormant"


# Global instance
_social_monitoring = None

def get_social_monitoring() -> SocialMonitoring:
    """Get or create the global social monitoring instance."""
    global _social_monitoring
    if _social_monitoring is None:
        _social_monitoring = SocialMonitoring()
    return _social_monitoring
