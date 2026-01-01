# ATLAS v3 Quick Start Guide

## What is ATLAS v3?

ATLAS v3 transforms the system from a fact-checker into a **Truth Intelligence System** that can:
- ✅ Detect PR content and propaganda
- ✅ Analyze source independence (are "10 sources" really just 1 press release?)
- ✅ Score credibility on 7 axes (vs v2's 4 metrics)
- ✅ Generate reasoning-rich verdicts with transparency notes

## How to Use

### For Users

1. **Start the server:**
   ```bash
   cd backend
   python server.py
   ```

2. **Open chat interface:**
   ```
   http://localhost:8000/chat
   ```

3. **Ask any question:**
   - "What happened in recent tech news?"
   - "Is this article about climate change accurate?"
   - "Tell me about the latest political announcement"

4. **See v3 intelligence automatically:**
   - AI response appears first
   - v3 Truth Intelligence panel displays below
   - Shows verdict, PR detection, source independence, confidence

### For Developers

#### Testing v3 Modules

**Test PR Detection:**
```python
from agents.pr_detection_engine import PRDetectionEngine

engine = PRDetectionEngine()
result = engine.analyze_content(
    content="For more information, contact press@company.com",
    sources=["https://example.com"]
)
print(f"PR detected: {result['is_pr']}")
print(f"Confidence: {result['confidence']}")
```

**Test Source Independence:**
```python
from agents.source_independence_graph import SourceIndependenceGraph

sig = SourceIndependenceGraph()
sig.add_source('url1', 'This is the same content', {'domain': 'site1.com'})
sig.add_source('url2', 'This is the same content', {'domain': 'site2.com'})

analysis = sig.analyze_independence()
print(f"Independence score: {analysis['independence_score']}")
print(f"Shared origin: {analysis['shared_origin_detected']}")
```

**Test Verdict Synthesis:**
```python
from agents.verdict_synthesizer import VerdictSynthesizer

vs = VerdictSynthesizer()
verdict = vs.synthesize_verdict(
    claim="Test claim",
    evidence_data=[],
    credibility_scores={'overall_score': 0.8},
    pr_analysis={'is_pr_content': False, 'confidence': 0},
    sig_analysis={'independence_score': 0.7, 'independent_source_count': 5}
)
print(verdict['verdict']['determination'])
```

#### API Response Structure

```json
{
  "analysis": "LLM response",
  "v3_intelligence": {
    "verdict": {
      "determination": "TRUE|FALSE|MISLEADING|UNVERIFIABLE|PR_CONTENT",
      "confidence_level": "HIGH|MEDIUM|LOW",
      "confidence_score": 0.85
    },
    "pr_detection": {
      "is_pr": false,
      "indicators": []
    },
    "source_independence": {
      "independence_score": 0.75
    },
    "confidence_breakdown": { ... }
  }
}
```

## UI Components

### Verdict Section
- **Badge colors:** Green (TRUE), Red (FALSE), Orange (MISLEADING), Gray (UNVERIFIABLE), Purple (PR_CONTENT)
- **Confidence bar:** Visual progress bar showing confidence level

### PR Detection Section
- **Indicators:** Lists detected PR patterns
- **Warnings:** Flags press releases and syndicated content

### Source Independence Section
- **Score display:** Large number showing independence (0-1)
- **Warnings:** Alerts for shared origin and coordination

### Confidence Breakdown
- **4 factors:** Evidence quantity, source credibility, source independence, evidence consensus
- **Visual bars:** Color-coded progress indicators

### Transparency Notes
- **User-friendly explanations:** Why certain flags were raised
- **Helpful context:** What to watch out for

## File Locations

### Backend Modules
- `backend/agents/pr_detection_engine.py` - PR detection
- `backend/agents/source_independence_graph.py` - Source clustering
- `backend/agents/verdict_synthesizer.py` - Verdict generation
- `backend/agents/credibility_engine.py` - 7-axis scoring

### Frontend
- `backend/static/js/atlas_v3.js` - UI component
- `backend/static/css/atlas_v3.css` - Styling

### Integration
- `backend/server.py` - API endpoint integration
- `backend/services/pro_scraper.py` - PR metadata extraction

## Configuration

### Enable/Disable v3
v3 is **automatically enabled** when modules are available. No configuration needed.

To disable, set in server.py:
```python
V3_MODULES_AVAILABLE = False
```

### Adjust PR Detection Thresholds
Edit `backend/agents/pr_detection_engine.py`:
```python
self.confidence_weights = {
    'press_release_origin': 0.35,  # Adjust weight
    # ...
}
```

### Adjust SIG Clustering Threshold
Edit `backend/agents/source_independence_graph.py`:
```python
self.similarity_threshold = 0.85  # Lower = stricter clustering
```

### Adjust Verdict Confidence Weights
Edit `backend/agents/verdict_synthesizer.py`:
```python
overall_confidence = (
    evidence_score * 0.2 +      # Adjust weights
    credibility_score * 0.3 +
    independence_score * 0.25 +
    consensus_score * 0.25
)
```

## Troubleshooting

### v3 Panel Not Appearing
1. Check browser console for errors
2. Verify `atlas_v3.js` is loaded in Network tab
3. Check API response contains `v3_intelligence` object
4. Clear browser cache (Ctrl+Shift+R)

### PR Detection Always False
1. Check if content contains PR patterns (see `pr_patterns` in engine)
2. Verify scraper is extracting contact info and boilerplate
3. Check source URL patterns match wire services

### Low Independence Scores
1. Verify sources have actual content (not empty)
2. Check if sentence_transformers is working
3. Increase `similarity_threshold` for stricter clustering

### Import Errors
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Test imports
python -c "from agents.pr_detection_engine import PRDetectionEngine"
python -c "from agents.source_independence_graph import SourceIndependenceGraph"
python -c "from agents.verdict_synthesizer import VerdictSynthesizer"
```

## Performance Tips

1. **Parallel Execution:** Modules run sequentially - future optimization opportunity
2. **Caching:** Consider caching PR detection by URL hash
3. **Batch Analysis:** Process multiple queries together
4. **Lazy Loading:** Load v3 UI only when needed

## Monitoring

### Check v3 Execution
```bash
# Watch logs for v3 activity
tail -f logs/atlas.log | grep "v3"
```

### API Response Size
v3 adds ~2-5 KB to each response (negligible)

### Execution Time
~0.7-1.7 seconds overhead per query

## Next Steps

1. **Test with real queries** - Try controversial topics
2. **Monitor accuracy** - Track verdict correctness
3. **Collect feedback** - Ask users about v3 insights
4. **Plan Phase 2** - OSINT integration for deeper analysis

## Support

- **Documentation:** `PRD/ATLAS_V3_IMPLEMENTATION_COMPLETE.md`
- **PRD:** `PRD/atlas_prd_fake_news_pr_detection.md`
- **Issues:** Check server logs in `logs/` directory

---

**Quick Links:**
- [Full Implementation Doc](ATLAS_V3_IMPLEMENTATION_COMPLETE.md)
- [v3 PRD](atlas_prd_fake_news_pr_detection.md)
- [v2.1 PRD](ATLAS_COMPREHENSIVE_PRD.md)
