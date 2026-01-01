# ATLAS v3 - Phase 2 & 3 Quick Start Guide

## What's New

**Phase 2: OSINT Integration (Media Forensics)**
- Reverse image search to detect reused/manipulated images
- EXIF metadata analysis for tampering detection
- Authenticity scoring (0-100)
- Red flag system for suspicious media

**Phase 3: Dynamic Outlet Reliability**
- Learning system that tracks outlet accuracy
- Authority scores adjust based on verdict outcomes
- Persistent database of outlet performance
- Replaces static credibility scores with dynamic ones

## Server Status

✅ **Running on http://127.0.0.1:8000**

Server loaded:
- Phase 1: PR Detection, SIG, Credibility, Verdict Synthesizer
- Phase 2: Media Forensics Engine
- Phase 3: Outlet Reliability Tracker

## How to Use

### 1. Analyze a Claim (With Images)

Submit any claim through the ATLAS interface. If evidence contains images:

**Media Forensics will automatically:**
- Extract image URLs from evidence
- Perform reverse image searches
- Analyze metadata (EXIF data)
- Detect tampering indicators
- Calculate authenticity score
- Generate red flags if issues found

**Example:**
```
Claim: "Photo shows recent flood in Mumbai"
```

If the image is from 2019, you'll see:
```
🚨 Red Flag: Image appears in content from 2019-08
Authenticity Score: 40/100
```

### 2. Dynamic Outlet Reliability

**Automatic Learning:**
Every time a verdict is reached, the system:
1. Records which outlets provided evidence
2. Updates their track record (verified/debunked/mixed)
3. Adjusts authority scores accordingly
4. Saves to persistent database

**Example Flow:**
```
Reuters article → Verdict: VERIFIED
→ Reuters verified count +1
→ Authority: 95 → 97 (+2 boost)

Unknown blog → Verdict: DEBUNKED  
→ Blog debunked count +1
→ Authority: 50 → 45 (-5 penalty)
```

### 3. Check UI for Results

**v3 Intelligence Panel includes:**

1. **Media Integrity Status** (Phase 2)
   - Authenticity Score badge
   - Reverse image search results
   - Red flags section
   - Forensics checks status

2. **Confidence Breakdown** (Enhanced with Phase 3)
   - Evidence Quantity
   - Source Credibility (now using dynamic scores)
   - Source Independence
   - Evidence Consensus

## API Keys (Optional)

For enhanced reverse image search:

```bash
# Add to backend/.env
GOOGLE_VISION_API_KEY=your_key_here
TINEYE_API_KEY=your_key_here
```

**Without API keys:** System uses deterministic URL-based analysis (still functional).

## Database Location

Phase 3 creates a learning database:
```
backend/database/outlet_reliability.json
```

This file grows over time as the system analyzes more claims.

## Monitoring

### Check Current Outlet Scores

```python
from agents.outlet_reliability import get_outlet_reliability_tracker

tracker = get_outlet_reliability_tracker()
stats = tracker.get_outlet_stats("reuters.com")
print(stats)
```

**Output:**
```json
{
  "domain": "reuters.com",
  "tracked": true,
  "base_authority": 95,
  "current_authority": 97,
  "track_record": {"verified": 5, "debunked": 0, "mixed": 1},
  "accuracy_rate": 83.3
}
```

### View Top Reliable Outlets

```python
top_outlets = tracker.get_top_reliable_outlets(limit=5)
for outlet in top_outlets:
    print(f"{outlet['domain']}: {outlet['authority']}")
```

### Check Declining Outlets

```python
declining = tracker.get_declining_outlets(threshold=-10)
for outlet in declining:
    print(f"{outlet['domain']}: {outlet['base_authority']} → {outlet['current_authority']}")
```

## Testing Checklist

- [ ] Analyze claim with images → Check Media Integrity Status appears
- [ ] Verify authenticity score shown (0-100)
- [ ] Check if red flags appear for suspicious images
- [ ] Submit multiple claims from same outlet
- [ ] Check `outlet_reliability.json` updates after each verdict
- [ ] Verify authority scores change over time
- [ ] Check logs show dynamic authority values

## Troubleshooting

**Media forensics not showing results?**
- Check if evidence contains image URLs
- Verify images are accessible (not 404)
- Check logs for PIL/numpy errors

**Outlet reliability not updating?**
- Check `backend/database/` directory exists
- Verify write permissions on directory
- Check server logs for JSON save errors

**Images not being analyzed?**
- Limit is 3 images per claim (configurable)
- Check image URL patterns match regex in code
- Verify image formats are supported (jpg/png/gif/webp)

## Performance Notes

- **Media forensics adds:** ~1-3 seconds per image (with APIs)
- **Outlet tracking adds:** < 10ms per verdict
- **Memory overhead:** Minimal (~1-2MB for tracking database)
- **No blocking:** All operations async-compatible

## Next Steps

1. **Monitor Performance:** Watch how outlet scores evolve over time
2. **Configure APIs:** Add Google Vision/TinEye keys for better results
3. **Custom Thresholds:** Adjust tampering detection sensitivity
4. **Export Data:** Generate reports from outlet_reliability.json

## Documentation

Full details: `PRD/PHASE_2_3_IMPLEMENTATION.md`

PRD specification: `PRD/atlas_prd_fake_news_pr_detection.md`

---

**Status:** ✅ All phases operational  
**Version:** ATLAS v3.0 Complete  
**Last Updated:** December 4, 2025
