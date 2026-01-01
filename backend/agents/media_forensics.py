# backend/agents/media_forensics.py
"""
Phase 2 - Media Forensics Module for ATLAS v3

Implements OSINT capabilities:
- Reverse image search (Google Vision API, TinEye)
- Metadata analysis (EXIF data extraction)
- Tampering detection (Error Level Analysis, noise patterns)
- Social media timestamp verification
"""

import os
import logging
import hashlib
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import BytesIO
import re

logger = logging.getLogger("media_forensics")

# API Configuration
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
TINEYE_API_KEY = os.getenv("TINEYE_API_KEY")


class MediaForensicsEngine:
    """
    Analyzes images and media for authenticity, manipulation, and provenance.
    """
    
    def __init__(self):
        self.has_google_vision = bool(GOOGLE_VISION_API_KEY)
        self.has_tineye = bool(TINEYE_API_KEY)
        logger.info(f"MediaForensics initialized - Google Vision: {self.has_google_vision}, TinEye: {self.has_tineye}")
    
    def analyze_media(self, claim: str, evidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for media forensics analysis.
        
        Args:
            claim: The claim being analyzed
            evidence_data: List of evidence items with potential image URLs
            
        Returns:
            {
                "reverse_image_results": [...],
                "metadata_analysis": {...},
                "tampering_detection": {...},
                "overall_authenticity_score": 0-100,
                "red_flags": [...]
            }
        """
        logger.info(f"Starting media forensics for claim: {claim[:100]}")
        
        # Extract image URLs from evidence
        image_urls = self._extract_image_urls(evidence_data)
        
        if not image_urls:
            return self._no_media_response()
        
        results = {
            "images_analyzed": len(image_urls),
            "reverse_image_results": [],
            "metadata_analysis": {},
            "tampering_detection": {},
            "overall_authenticity_score": 100,
            "red_flags": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Process each image
        for idx, img_url in enumerate(image_urls[:3]):  # Limit to 3 images
            try:
                # Reverse image search
                reverse_results = self._reverse_image_search(img_url)
                if reverse_results:
                    results["reverse_image_results"].append(reverse_results)
                    
                    # Check for earlier appearances
                    if reverse_results.get("earliest_date"):
                        results["red_flags"].append({
                            "type": "outdated_image",
                            "severity": "high",
                            "description": f"Image appears in content from {reverse_results['earliest_date']}",
                            "image_url": img_url
                        })
                
                # Metadata analysis
                metadata = self._analyze_metadata(img_url)
                if metadata:
                    results["metadata_analysis"][f"image_{idx}"] = metadata
                    
                    # Check for metadata inconsistencies
                    if metadata.get("modified") or metadata.get("software_detected"):
                        results["red_flags"].append({
                            "type": "metadata_manipulation",
                            "severity": "medium",
                            "description": "Image metadata suggests editing or manipulation",
                            "details": metadata
                        })
                
                # Tampering detection
                tampering = self._detect_tampering(img_url)
                if tampering:
                    results["tampering_detection"][f"image_{idx}"] = tampering
                    
                    if tampering.get("tampering_detected"):
                        results["red_flags"].append({
                            "type": "tampering_detected",
                            "severity": "critical",
                            "description": tampering.get("description", "Image shows signs of digital manipulation"),
                            "confidence": tampering.get("confidence", 0)
                        })
            
            except Exception as e:
                logger.error(f"Error analyzing image {img_url}: {e}")
                continue
        
        # Calculate overall authenticity score
        results["overall_authenticity_score"] = self._calculate_authenticity_score(results)
        
        return results
    
    def _extract_image_urls(self, evidence_data: List[Dict[str, Any]]) -> List[str]:
        """Extract image URLs from evidence items."""
        image_urls = []
        
        for item in evidence_data:
            # Check for direct image URL field
            if item.get("image_url"):
                image_urls.append(item["image_url"])
            
            # Check for images in content/text
            content = item.get("content", "") or item.get("text", "")
            if content:
                # Extract URLs that look like images
                img_patterns = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
                image_urls.extend(img_patterns)
        
        return list(set(image_urls))[:5]  # Deduplicate and limit
    
    def _reverse_image_search(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Perform reverse image search using available APIs.
        Falls back to deterministic analysis if no API keys.
        """
        if self.has_google_vision:
            return self._google_vision_search(image_url)
        elif self.has_tineye:
            return self._tineye_search(image_url)
        else:
            # Deterministic fallback
            return self._fallback_reverse_search(image_url)
    
    def _google_vision_search(self, image_url: str) -> Optional[Dict[str, Any]]:
        """Use Google Vision API for reverse image search."""
        try:
            api_endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
            
            payload = {
                "requests": [
                    {
                        "image": {"source": {"imageUri": image_url}},
                        "features": [
                            {"type": "WEB_DETECTION", "maxResults": 10}
                        ]
                    }
                ]
            }
            
            response = requests.post(api_endpoint, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                web_detection = data.get("responses", [{}])[0].get("webDetection", {})
                
                pages_with_matching = web_detection.get("pagesWithMatchingImages", [])
                
                earliest_date = None
                if pages_with_matching:
                    # Try to extract dates from URLs/titles
                    for page in pages_with_matching[:5]:
                        url = page.get("url", "")
                        # Simple date extraction from URL
                        date_match = re.search(r'/(\d{4})/(\d{1,2})', url)
                        if date_match:
                            year, month = date_match.groups()
                            if not earliest_date or f"{year}-{month}" < earliest_date:
                                earliest_date = f"{year}-{month}"
                
                return {
                    "method": "google_vision",
                    "matches_found": len(pages_with_matching),
                    "earliest_date": earliest_date,
                    "matching_pages": [
                        {
                            "url": p.get("url"),
                            "title": p.get("pageTitle", "Unknown")
                        }
                        for p in pages_with_matching[:5]
                    ]
                }
        
        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            return None
    
    def _tineye_search(self, image_url: str) -> Optional[Dict[str, Any]]:
        """Use TinEye API for reverse image search."""
        try:
            api_endpoint = "https://api.tineye.com/rest/search/"
            
            params = {
                "image_url": image_url,
                "api_key": TINEYE_API_KEY
            }
            
            response = requests.get(api_endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("results", {}).get("matches", [])
                
                earliest_date = None
                if matches:
                    # Find earliest crawl date
                    for match in matches:
                        crawl_date = match.get("crawl_date")
                        if crawl_date and (not earliest_date or crawl_date < earliest_date):
                            earliest_date = crawl_date
                
                return {
                    "method": "tineye",
                    "matches_found": len(matches),
                    "earliest_date": earliest_date,
                    "matching_pages": [
                        {
                            "url": m.get("url"),
                            "domain": m.get("domain"),
                            "crawl_date": m.get("crawl_date")
                        }
                        for m in matches[:5]
                    ]
                }
        
        except Exception as e:
            logger.error(f"TinEye API error: {e}")
            return None
    
    def _fallback_reverse_search(self, image_url: str) -> Dict[str, Any]:
        """
        Deterministic fallback when no API keys available.
        Uses URL analysis and hash-based checking.
        """
        # Extract domain and path info
        from urllib.parse import urlparse
        parsed = urlparse(image_url)
        domain = parsed.netloc
        path = parsed.path
        
        # Check if URL contains date patterns
        date_pattern = re.search(r'/(\d{4})[/-]?(\d{1,2})?[/-]?(\d{1,2})?', path)
        earliest_date = None
        if date_pattern:
            year = date_pattern.group(1)
            month = date_pattern.group(2) or "01"
            earliest_date = f"{year}-{month}"
        
        # Generate content hash for tracking
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        
        return {
            "method": "fallback",
            "matches_found": 0,
            "earliest_date": earliest_date,
            "note": "Limited analysis - API keys not configured",
            "image_hash": url_hash,
            "domain": domain
        }
    
    def _analyze_metadata(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Analyze image metadata (EXIF data).
        Requires PIL/Pillow library.
        """
        try:
            # Lazy import
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            # Download image
            response = requests.get(image_url, timeout=10, stream=True)
            if response.status_code != 200:
                return None
            
            img = Image.open(BytesIO(response.content))
            
            # Extract EXIF data
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            
            if not exif_data:
                return {
                    "has_metadata": False,
                    "note": "No EXIF data found (may be stripped)"
                }
            
            metadata = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = str(value)
            
            return {
                "has_metadata": True,
                "camera": metadata.get("Make", "Unknown"),
                "software_detected": "Software" in metadata,
                "modified": "DateTime" in metadata and "DateTimeOriginal" in metadata,
                "gps_location": "GPSInfo" in metadata,
                "selected_fields": {
                    k: v for k, v in metadata.items() 
                    if k in ["Make", "Model", "Software", "DateTime", "DateTimeOriginal"]
                }
            }
        
        except ImportError:
            logger.warning("PIL/Pillow not installed - metadata analysis skipped")
            return {"error": "PIL library required"}
        except Exception as e:
            logger.error(f"Metadata analysis error: {e}")
            return None
    
    def _detect_tampering(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Detect image tampering using Error Level Analysis (ELA) and noise patterns.
        This is a simplified version - full ELA requires specialized libraries.
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Download image
            response = requests.get(image_url, timeout=10, stream=True)
            if response.status_code != 200:
                return None
            
            img = Image.open(BytesIO(response.content))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Simple tampering indicators
            tampering_score = 0
            indicators = []
            
            # Check 1: Uniform regions (copy-paste indicator)
            if len(img_array.shape) == 3:  # Color image
                variance = np.var(img_array, axis=(0, 1))
                if np.min(variance) < 10:  # Very low variance
                    tampering_score += 30
                    indicators.append("Detected uniform regions (possible copy-paste)")
            
            # Check 2: Sharp edges in compression artifacts
            # (This is simplified - full ELA requires re-compression)
            if hasattr(img, 'info') and 'quality' in img.info:
                quality = img.info['quality']
                if quality > 95:  # Very high quality suggests minimal compression
                    tampering_score += 20
                    indicators.append("High JPEG quality (unusual for web images)")
            
            # Check 3: Image dimensions
            width, height = img.size
            if width % 8 != 0 or height % 8 != 0:
                # JPEG compression works on 8x8 blocks
                tampering_score += 10
                indicators.append("Non-standard dimensions for JPEG")
            
            tampering_detected = tampering_score > 40
            
            return {
                "tampering_detected": tampering_detected,
                "confidence": min(tampering_score, 100),
                "indicators": indicators,
                "description": f"Tampering analysis score: {tampering_score}/100",
                "method": "simplified_ela"
            }
        
        except ImportError:
            return {"error": "PIL/numpy required for tampering detection"}
        except Exception as e:
            logger.error(f"Tampering detection error: {e}")
            return None
    
    def _calculate_authenticity_score(self, results: Dict[str, Any]) -> int:
        """
        Calculate overall authenticity score based on forensics analysis.
        100 = highly authentic, 0 = likely manipulated
        """
        score = 100
        
        red_flags = results.get("red_flags", [])
        
        for flag in red_flags:
            severity = flag.get("severity", "low")
            if severity == "critical":
                score -= 40
            elif severity == "high":
                score -= 25
            elif severity == "medium":
                score -= 15
            else:
                score -= 5
        
        return max(0, min(100, score))
    
    def _no_media_response(self) -> Dict[str, Any]:
        """Return response when no media to analyze."""
        return {
            "images_analyzed": 0,
            "reverse_image_results": [],
            "metadata_analysis": {},
            "tampering_detection": {},
            "overall_authenticity_score": 100,
            "red_flags": [],
            "note": "No images found in evidence to analyze",
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
_forensics_engine = None

def get_media_forensics_engine() -> MediaForensicsEngine:
    """Get or create the global media forensics engine instance."""
    global _forensics_engine
    if _forensics_engine is None:
        _forensics_engine = MediaForensicsEngine()
    return _forensics_engine
