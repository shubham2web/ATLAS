# ATLAS v3 — Truth Intelligence System (TIS)
## **Comprehensive PRD for Fake-News Detection, PR Detection, Source Independence & Reasoning Engine**
### **(Modules A + B + C + D — Full Integration Proposal)**

**Author:** ATLAS Core Strategy  
**Purpose:** Present to internal team as the blueprint for next-generation truth analysis in ATLAS.  
**Objective:** Upgrade ATLAS from a *fact-checker* to a **truth intelligence system** capable of:
- multi-source verification  
- PR & propaganda detection  
- source-independence graphing  
- bias fingerprinting  
- reasoning-rich verdict generation  
- non-reliance on major outlets  

This document is designed to be shared with engineering + research teams.

---
# 🧩 **Overview**
Modern misinformation is not limited to fake, viral WhatsApp forwards.  
**Major outlets, PR agencies, governments, political organizations, corporations, and activists all shape narratives daily.**  
ATLAS must be able to detect:
- fake news  
- biased reporting  
- corporate PR disguised as journalism  
- government propaganda  
- coordinated narrative shaping  
- selective omission

This PRD defines how ATLAS accomplishes this.

Modules included:
- **A — PR / Propaganda Detection Engine**
- **B — Source Independence Graph (SIG Framework)**
- **C — Multi-Axis Credibility & Bias Scoring Engine**
- **D — Reasoning-Rich Verdict Synthesizer (LLM + Evidence Chain)**

---
# 🧱 **SYSTEM GOAL**
Create the world’s first **Truth Intelligence System**—a system that:
- Does *not* trust any single outlet
- Cross-verifies across ecosystems
- Detects PR origin chains
- Understands linguistic framing
- Uses OSINT-style evidence signals
- Produces short, actionable reasoning

Users receive:
- A verdict (Likely Real / Uncertain / Likely Fake)
- A 0–100 credibility score
- Key reasoning bullets
- Top independent evidence
- Bias fingerprint report

---
# 🔷 **MODULE A — PR & Propaganda Detection Engine**
(PRD Component A)

## **Purpose**
Identify when a news article is actually:
- a press release
- a political narrative
- a corporate PR message
- syndicated content copied across multiple outlets
- a narrative driven by a single actor (party/gov/brand)

## **Detection Components**

### **1. Press Release Origin Detection**
Check for:
- Vocabulary overlap with PRNewswire / BusinessWire / ANI / PTI / Govt Press Office
- Boilerplate patterns ("X company announced today…")
- Corporate tone markers ("industry-leading", "market-disrupting")

### **2. Syndication / Copy-paste Detection**
Measure similarity across outlets:
- Cosine similarity > 0.90 → identical copy
- Overlapping sentence structure
- Same timestamp or press-release tag
- Cluster articles → treat as ONE source

### **3. Narrative Alignment Detection**
Detect when language matches:
- political talking points
- corporate talking points
- military/government framing
- activist organization framing

### **4. Missing Journalism Markers**
Real journalism includes:
- direct quotes from named individuals
- timestamp & location
- multi-source attribution
- counter-positions
- background context

Articles missing these → PR-likelihood increases.

### **5. Benefit Analysis**
Ask: **“Who benefits if this story is believed?”**
If the answer matches a political or corporate actor → flag.

## **Output of Module A**
- `pr_score` (0–100)
- `pr_flags`: ["press_release_like", "identical_copy", "single_origin", "political_framing"]
- `narrative_origin_actor`: optional guess

---
# 🔶 **MODULE B — Source Independence Graph (SIG Framework)**
(PRD Component B)

## **Purpose**
Determine whether multiple sources reporting the story are truly independent.

### **Key Idea**
If 10 outlets publish the same article, but all originated from:
- **1 PR agency**, or  
- **1 governmental press office**, or  
- **1 corporate PR**, or  
- **1 news wire provider**,

Then there is actually **only one source**, not ten.

## **Graph Construction**
Each document is a node.  
Edges represent:
- text similarity
- shared metadata
- identical publishing timestamps
- shared press-release boilerplate
- shared image source

## **Cluster Types**
- **PR Cluster** — originates from PRNewswire/BusinessWire/ANI/PTI
- **Wire Cluster** — Reuters/AP/AFP syndicated reporting
- **Copy-Paste Cluster** — low-authority blogs repeating same article
- **Independent Cluster** — genuinely unique reporting

## **Output of SIG**
- `independence_index` (0–1)
- clusters list with origin labels
- count of unique sources
- evidence map for UI visualization

Higher independence → higher credibility.
Low independence → suspicious narrative.

---
# 🟣 **MODULE C — Multi-Axis Credibility & Bias Scoring Engine**
(PRD Component C)

This engine produces the **core 0–100 fake/real score**.
It uses multiple axes instead of trusting major outlets.

## **Credibility Axes**

### **1. Domain Reliability (dynamic)**
- not static; updated by historical behavior
- penalize if debunked articles accumulate

### **2. Ecosystem Cross-Verification**
Check coverage from:
- local reporters
- global outlets
- wire services
- regional languages
- citizen evidence
- OSINT data

A story is strong only when **ecosystems align**.

### **3. Temporal Consistency**
Dates/times must match:
- eyewitness social media posts
- video metadata
- sensor logs

### **4. Source Independence** (from Module B)
Low independence → big penalty.

### **5. Sentiment & Framing Analysis**
Check for:
- sensational tone
- emotional manipulation
- PR-speak
- tribal political language

### **6. Evidence Diversity**
Diverse sources → more credible.
Identical cluster → low.

### **7. Fact-checker / RAG Hit**
Strong positive OR negative boost.

## **Cumulative Scoring**
Each axis produces a probability.  
Combined via probabilistic merge:
```
p_total = 1 - Product(1 - p_i)
```
Not linear (prevents runaway).

## **Outputs**
- `fake_score` (0–100)
- `confidence_label`: High / Medium / Low
- `credibility_axes`: detailed breakdown

---
# 🔵 **MODULE D — Reasoning-Rich Verdict Synthesizer**
(PRD Component D)

## **Purpose**
Provide a **short, human-readable reasoning chain** instead of debate cycles.

## **Inputs**
LLM receives:
- claim (1–2 lines)
- top 3 independent evidence sources
- SIG independence summary
- PR flags
- bias auditor flags
- fake_score & axis breakdown
- contradictory evidence from RAG

## **Required Output Format**
### **Structured 4-part reasoning:**
1. **Verdict:** Likely Fake / Uncertain / Likely Real (with score)  
2. **Evidence reasoning (2–3 bullets)**
   - Must reference specific evidence nodes
3. **PR / bias explanations (optional)**  
4. **Confidence + suggested next step**

### Example output:
```
VERDICT: Likely Fake (82) — High confidence.
Reasoning:
• No independent outlets corroborate the claim; all 6 articles originate from the same ANI press release (SIG cluster #1).
• Reverse-image check shows the photo is from a 2019 flood event, not related to this incident.
• RAG identified a partial debunk from BoomLive dated 2023 for a similar viral claim.
Recommendation: Do not share. Flag for human review.
```

## **Why this works**
- shorter than debates  
- user-friendly  
- high information density  
- grounded in evidence, not LLM imagination

---
# 🧠 **Unified Pipeline (A+B+C+D)**

```
Input → Claim Extraction → Evidence Gathering → PR Detection → SIG Analysis →
Cross-Ecosystem Verification → Bias Audit → Probabilistic Scoring →
Verdict Synthesizer → UI Report
```

---
# 🎯 **What This Lets ATLAS Do**
- Detect fake news
- Detect PR-disguised news
- Detect propaganda
- Detect coordinated media narratives
- Detect copy-paste journalism
- Detect missing independent verification
- Expose bias patterns
- Produce concise reasoning

This is **beyond fact-checking** — this is media forensics.

---
# 📌 **Engineering Implementation Phases**

## **Phase 1 — Core Scoring & Reasoning**
- Add SIG clustering
- Add PR detection heuristics
- Integrate into `/v2/analyze`
- Add reasoning LLM step

## **Phase 2 — OSINT Integration**
- add reverse-image search
- add sensor data support
- add social media timestamp verification

## **Phase 3 — Dynamic Outlet Reliability**
- historical debunk tracking
- adjust domain authority based on performance

---
# 🏁 **Final Notes for Team**
This PRD transforms ATLAS from a misinformation detector into a **Truth Intelligence System**.  
It no longer treats BBC or TOI or any outlet as inherently trustworthy.  
It instead measures **independence, corroboration, PR origin, bias, and evidence diversity**.

This gives ATLAS:  
- higher accuracy  
- higher transparency  
- resistance to propaganda  
- ability to detect corporate/government PR

And gives users:  
- short, powerful explanations they trust.

---
# END OF DOCUMENT

