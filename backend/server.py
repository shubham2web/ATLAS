import asyncio
import json
import logging
import mimetypes
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
import sys
import random
import httpx
import feedparser
from typing import List, Dict

from quart import Quart, render_template, request, jsonify, send_from_directory, Response
from quart_cors import cors
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

# Note: These local imports need to exist in your project structure
from core.ai_agent import AiAgent
from core.config import API_KEY, DEBUG_MODE, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, OPENAI_API_KEY
from services.db_manager import AsyncDbManager, DATABASE_FILE
from services.pro_scraper import get_diversified_evidence
from core.utils import compute_advanced_analytics, format_sse
import io

# Import OCR functionality (EasyOCR - no Tesseract needed!)
# Lazy import to avoid scipy/numpy Python 3.13 compatibility issues at startup
OCR_AVAILABLE = False
_ocr_processor = None

def get_ocr_processor_lazy():
    """Lazy load OCR processor only when needed"""
    global OCR_AVAILABLE, _ocr_processor
    if _ocr_processor is None:
        try:
            from services.ocr_processor import get_ocr_processor
            _ocr_processor = get_ocr_processor()
            OCR_AVAILABLE = True
            logging.info("EasyOCR module loaded successfully (no Tesseract needed!)")
        except (ImportError, OSError, RuntimeError) as e:
            OCR_AVAILABLE = False
            logging.warning(f"OCR functionality not available: {e}. Install dependencies: pip install easyocr pillow torch torchvision")
    return _ocr_processor

# Import v2.0 routes
# Temporarily disabled due to slow transformers loading
try:
    from api.api_v2_routes import v2_bp
    V2_AVAILABLE = True
    logging.info("ATLAS v2.0 routes loaded successfully")
except ImportError as e:
    V2_AVAILABLE = False
    logging.warning(f"?? ATLAS v2.0 routes not available: {e}")

# Import Hybrid Memory System routes
try:
    from api.memory_routes import memory_bp
    from memory.memory_manager import get_memory_manager
    MEMORY_AVAILABLE = True
    logging.info("Hybrid Memory System routes loaded successfully")
except ImportError as e:
    MEMORY_AVAILABLE = False
    logging.warning(f"?? Memory System routes not available: {e}. Install: pip install -r memory_requirements.txt")

# Import ATLAS v4.0 Analysis Pipeline routes
try:
    from api.analyze_routes import analyze_bp
    ANALYZE_PIPELINE_AVAILABLE = True
    logging.info("ATLAS v4.0 Analysis Pipeline routes loaded successfully")
except ImportError as e:
    ANALYZE_PIPELINE_AVAILABLE = False
    logging.warning(f"?? Analysis Pipeline routes not available: {e}")

# Import v2 Features (Bias Auditor, Credibility Engine, Forensic Engine)
try:
    from agents.bias_auditor import BiasAuditor
    from agents.credibility_engine import CredibilityEngine, Source
    from agents.forensic_engine import get_forensic_engine
    from agents.role_reversal_engine import RoleReversalEngine
    V2_FEATURES_AVAILABLE = True
    logging.info("ATLAS v2.0 Features loaded (Bias Auditor, Credibility Engine, Forensic Engine)")
except ImportError as e:
    V2_FEATURES_AVAILABLE = False
    logging.warning(f"?? V2 Features not available: {e}")

# Import v3 Truth Intelligence Modules
try:
    from agents.pr_detection_engine import PRDetectionEngine
    from agents.source_independence_graph import SourceIndependenceGraph
    from agents.verdict_synthesizer import VerdictSynthesizer
    V3_MODULES_AVAILABLE = True
    logging.info("ATLAS v3 Truth Intelligence Modules loaded (PR Detection, SIG, Verdict Synthesizer)")
except ImportError as e:
    V3_MODULES_AVAILABLE = False
    logging.warning(f"ATLAS v3 modules not available: {e}")

# Import MongoDB Audit Logger (optional)
try:
    from memory.mongo_audit import MongoAuditLogger, get_audit_logger
    MONGO_AUDIT_AVAILABLE = True
except ImportError:
    MONGO_AUDIT_AVAILABLE = False
    logging.info("MongoDB audit logging not available (optional)")

# Import PRD Compliance Checker
try:
    from tools.prd_checker import (
        has_citation, is_factual_claim, generate_citation_prompt,
        run_full_prd_check, extract_citations
    )
    PRD_CHECKER_AVAILABLE = True
    logging.info("PRD Compliance Checker loaded")
except ImportError as e:
    PRD_CHECKER_AVAILABLE = False
    logging.warning(f"?? PRD Checker not available: {e}")
    # Fallback implementations
    def has_citation(text): return "[SRC:" in text
    def is_factual_claim(text): return any(k in text.lower() for k in ["said", "reported", "confirmed", "according"])
    def generate_citation_prompt(role, text): return f"{role}, please cite your sources using [SRC:ID] format."
    
    # Create dummy functions
    def get_audit_logger():
        return None

# --------------------------
# Setup Quart App & Executor
# --------------------------
app = Quart(__name__, 
            static_folder='static',
            static_url_path='/static')

# --- CORS setup with environment variable ---
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
app = cors(app, 
           allow_origin=ALLOWED_ORIGIN,
           allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
           allow_headers=["Content-Type", "Authorization"])

# --- Register v2.0 Blueprint ---
if V2_AVAILABLE:
    app.register_blueprint(v2_bp)
    logging.info("ATLAS v2.0 endpoints registered at /v2/*")

# --- Register Memory System Blueprint ---
if MEMORY_AVAILABLE:
    app.register_blueprint(memory_bp)
    logging.info("Hybrid Memory System endpoints registered at /memory/*")

# --- Register Chat Persistence Blueprint (MongoDB-backed) ---
try:
    from api.chat_routes import chat_bp
    CHAT_API_AVAILABLE = True
    app.register_blueprint(chat_bp)
    logging.info("Chat persistence API endpoints registered at /api/chats/*")
except Exception as e:
    CHAT_API_AVAILABLE = False
    logging.warning(f"Chat API not available: {e}")

# --- Register ATLAS v4.0 Analysis Pipeline Blueprint ---
if ANALYZE_PIPELINE_AVAILABLE:
    app.register_blueprint(analyze_bp)
    logging.info("ATLAS v4.0 Analysis Pipeline endpoints registered at /analyze/*")

# --- Register Admin Blueprint ---
try:
    from api.routes_admin import admin_bp
    app.register_blueprint(admin_bp)
    logging.info("Admin endpoints registered at /admin/*")
except Exception as e:
    logging.warning(f"Admin routes not available: {e}")

executor = ThreadPoolExecutor(max_workers=10)

# --- JSON logging (production-ready) ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
        }
        if record.exc_info:
            payload["exception"] = traceback.format_exc()
        return json.dumps(payload)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
root = logging.getLogger()
root.handlers = [handler]
root.setLevel(logging.INFO)

# --- Custom Rate Limiter for Quart using limits ---
storage = MemoryStorage()
rate_limiter = MovingWindowRateLimiter(storage)

def limit(rule: str):
    """Decorator for Quart endpoints to apply rate limiting."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if not rate_limiter.hit(parse(rule), client_ip):
                return jsonify({"error": "Rate limit exceeded"}), 429
            return await func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# -----------------------------
# Security & Middleware
# -----------------------------
@app.before_request
async def check_api_key():
    # Allow access without API key for these endpoints
    if (request.endpoint in ['home', 'chat', 'healthz', 'analyze_topic', 'ocr_upload', 'ocr_page', 'game_page', 'game_headlines'] or  # Added ocr_page, ocr_upload, and game endpoints
        request.path.startswith('/static/') or
        request.path.startswith('/v2/') or  # Allow v2.0 endpoints without API key
        request.path.startswith('/analyze') or  # Allow analyze endpoints (v4.1 verdict engine)
        request.path.startswith('/rag/') or  # Allow RAG integration endpoints
        request.path.startswith('/admin/') or  # Allow admin endpoints
        request.path.startswith('/api/chats') or  # Allow chat listing/creation without API key for local UI
        request.path.startswith('/api/game/') or  # Allow game endpoints without API key
        not API_KEY or 
        request.method == 'OPTIONS'):
        return

    # Check API key for other endpoints
    received_key = request.headers.get("X-API-Key")
    if received_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

@app.after_request
async def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Allow microphone for same-origin pages so getUserMedia can be used from the app
    # Previous value denied microphone; change to allow same-origin usage.
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self)"
    return response

# Add this to ensure correct MIME types
@app.after_request
async def add_header(response):
    if request.path.endswith('.js'):
        response.headers['Content-Type'] = 'application/javascript'
    elif request.path.endswith('.css'):
        response.headers['Content-Type'] = 'text/css'
        # Force no caching for CSS files to fix styling issues
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Explicit static file serving route with cache busting
@app.route('/static/<path:filename>')
async def serve_static(filename):
    """Serve static files with proper cache control headers"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    response = await send_from_directory('static', filename)
    
    if filename.endswith('.css'):
        response.headers['Content-Type'] = 'text/css; charset=utf-8'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    elif filename.endswith('.js'):
        response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    
    return response

# -----------------------------
# Game RSS Configuration
# -----------------------------
REAL_RSS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]
SATIRE_RSS = [
    "https://www.theonion.com/rss",
    "https://thehardtimes.net/feed/",
]

_rss_cache: Dict[str, Dict] = {}
CACHE_TTL = 300  # 5 minutes

async def _fetch_rss(url: str) -> List[Dict[str, str]]:
    now = time.time()
    if url in _rss_cache and (now - _rss_cache[url]["ts"] < CACHE_TTL):
        return _rss_cache[url]["items"]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.text)
    except Exception as e:
        logger.error(f"Failed to fetch RSS {url}: {e}")
        return []

    items = []
    for e in parsed.entries[:15]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "source": parsed.feed.get("title", url),
        })

    _rss_cache[url] = {"ts": now, "items": items}
    return items

async def _pick_news(sources: List[str], need: int) -> List[Dict[str, str]]:
    pool: List[Dict[str, str]] = []
    for url in sources:
        try:
            pool.extend(await _fetch_rss(url))
        except Exception:
            continue
    random.shuffle(pool)
    seen = set()
    uniq = []
    for x in pool:
        key = x["title"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(x)
        if len(uniq) >= need:
            break
    return uniq

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
async def home():
    """Landing/Hero page"""
    return await render_template("homepage.html")

@app.route("/chat")
async def chat():
    """Render the chat interface with optional mode parameter"""
    mode = request.args.get('mode', 'analytical')
    # Pass API_KEY into the template so the client can use it for authenticated requests (dev only)
    try:
        return await render_template('index.html', mode=mode, API_KEY=API_KEY)
    except Exception:
        return await render_template('index.html', mode=mode)

@app.route("/ocr")
async def ocr_page():
    """Render the OCR interface"""
    return await render_template('ocr.html')

@app.route("/game")
async def game_page():
    """Render the news game interface"""
    return await render_template('game.html')

@app.get("/api/game/headlines")
async def game_headlines():
    """Get 3 real news + 1 satire headline for the game"""
    # Get userId and seed for unique randomization per user
    user_id = request.args.get('userId', 'default')
    seed_value = request.args.get('seed', str(time.time()))
    
    # Use userId + seed for deterministic but unique randomization
    random.seed(f"{user_id}_{seed_value}")
    
    real = await _pick_news(REAL_RSS, 3)
    satire = await _pick_news(SATIRE_RSS, 1)
    
    if not satire:
        return {"error": "Failed to fetch satire news"}, 500
    
    items = real + satire
    satire_item = satire[0]  # Store reference before shuffle
    random.shuffle(items)
    answer_index = items.index(satire_item)  # Find position after shuffle
    
    # Reset random seed to avoid affecting other parts of the app
    random.seed()
    
    return {"items": items, "answerIndex": answer_index}

@app.route("/healthz")
async def healthz():
    """Provides a simple health check endpoint."""
    return jsonify({"status": "ok"})

# =====================================================
# TEXT ACTION ENDPOINT WITH FALLBACK STRATEGY
# Priority: 1. Grok (Groq) -> 2. HuggingFace -> 3. Gemini
# =====================================================

async def call_groq_api(prompt: str) -> str:
    """Call Groq API (Grok models)"""
    import aiohttp
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY not configured")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Groq API error: {resp.status} - {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

async def call_huggingface_api(prompt: str) -> str:
    """Call HuggingFace Inference API"""
    import aiohttp
    hf_token = os.getenv("HF_TOKEN_1") or os.getenv("HF_TOKEN_2")
    if not hf_token:
        raise Exception("HuggingFace token not configured")
    
    async with aiohttp.ClientSession() as session:
        # Using Qwen model via router endpoint
        async with session.post(
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"HuggingFace API error: {resp.status} - {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

async def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API"""
    import aiohttp
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise Exception("GEMINI_API_KEY not configured")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={gemini_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024
                }
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Gemini API error: {resp.status} - {error_text}")
            data = await resp.json()
            candidates = data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                return candidates[0]["content"]["parts"][0]["text"]
            raise Exception("Unexpected Gemini response format")

@app.route("/text_action", methods=["POST"])
@limit("20/minute")
async def text_action():
    """
    Handle text actions (summarize, explain) with fallback strategy.
    Priority: 1. Grok (Groq) -> 2. HuggingFace -> 3. Gemini
    """
    try:
        data = await request.get_json()
        action = data.get("action", "")
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if action not in ["summarize", "explain"]:
            return jsonify({"error": "Invalid action. Use 'summarize' or 'explain'"}), 400
        
        # Build the prompt based on action
        if action == "summarize":
            prompt = f"Please provide a concise summary of the following text. Be brief and capture the key points:\n\n{text}"
        else:  # explain
            prompt = f"Please explain the following text in simple, easy-to-understand terms:\n\n{text}"
        
        result = None
        provider_used = None
        errors = []
        
        # Try Groq first
        try:
            logging.info("Attempting Groq API for text action...")
            result = await call_groq_api(prompt)
            provider_used = "groq"
            logging.info("Groq API succeeded")
        except Exception as e:
            errors.append(f"Groq: {str(e)}")
            logging.warning(f"Groq API failed: {e}")
        
        # Try HuggingFace if Groq failed
        if result is None:
            try:
                logging.info("Attempting HuggingFace API for text action...")
                result = await call_huggingface_api(prompt)
                provider_used = "huggingface"
                logging.info("HuggingFace API succeeded")
            except Exception as e:
                errors.append(f"HuggingFace: {str(e)}")
                logging.warning(f"HuggingFace API failed: {e}")
        
        # Try Gemini as final fallback
        if result is None:
            try:
                logging.info("Attempting Gemini API for text action...")
                result = await call_gemini_api(prompt)
                provider_used = "gemini"
                logging.info("Gemini API succeeded")
            except Exception as e:
                errors.append(f"Gemini: {str(e)}")
                logging.warning(f"Gemini API failed: {e}")
        
        if result is None:
            return jsonify({
                "error": "All AI providers failed",
                "details": errors
            }), 503
        
        return jsonify({
            "success": True,
            "result": result,
            "provider": provider_used,
            "action": action
        })
        
    except Exception as e:
        logging.error(f"Text action error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route("/analyze_topic", methods=["POST"])
async def analyze_topic():
    """Analyze a topic and return insights (with memory support)"""
    try:
        data = await request.get_json()
        topic = data.get("topic", "")
        model = data.get("model", "llama3")
        session_id = data.get("session_id")  # Optional: maintain conversation context
        conversation_history = data.get("conversation_history", [])  # Get conversation history from frontend
        
        if not topic:
            return jsonify({"error": "No topic provided"}), 400
        
        logging.info(f"Analyzing topic: {topic} (session: {session_id or 'new'}, history: {len(conversation_history)} messages)")
        
        # ?? MEMORY INTEGRATION: Initialize or retrieve memory session
        memory = None
        if MEMORY_AVAILABLE:
            try:
                memory = get_memory_manager()
                if session_id:
                    memory.set_debate_context(session_id)
                    logging.info(f"?? Using existing chat session: {session_id}")
                else:
                    session_id = str(uuid.uuid4())
                    memory.set_debate_context(session_id)
                    logging.info(f"?? Created new chat session: {session_id}")
            except Exception as e:
                logging.warning(f"Memory initialization failed: {e}")
                memory = None
        
        loop = asyncio.get_running_loop()
        
        # --- GOD MODE: START TIMING FOR RAG ---
        rag_start_time = time.time()
        rag_status = "INTERNAL_KNOWLEDGE"  # Default
        
        try:
            # Get evidence with timeout
            evidence_bundle = await asyncio.wait_for(
                get_diversified_evidence(topic),
                timeout=60.0
            )
            logging.info(f"Found {len(evidence_bundle)} sources")
            
            # --- GOD MODE: Calculate RAG performance ---
            rag_duration = time.time() - rag_start_time
            
            # If evidence was returned in under 1.5 seconds, it's a Cache Hit
            if rag_duration < 1.5 and evidence_bundle:
                rag_status = "CACHE_HIT"
                logging.info(f"? CACHE HIT: {rag_duration:.2f}s")
            elif evidence_bundle:
                rag_status = "LIVE_FETCH"
                logging.info(f"?? LIVE FETCH: {rag_duration:.2f}s")
            else:
                rag_status = "INTERNAL_KNOWLEDGE"
            
        except asyncio.TimeoutError:
            logging.warning("Evidence gathering timed out")
            evidence_bundle = []
        
        # ATLAS v3: Enhanced Analysis Pipeline
        v3_analysis = None
        if V3_MODULES_AVAILABLE and evidence_bundle:
            try:
                logging.info("Running ATLAS v3 Truth Intelligence analysis...")
                
                # Phase 3: Dynamic Outlet Reliability (adjust evidence authority scores)
                from agents.outlet_reliability import get_outlet_reliability_tracker
                outlet_tracker = get_outlet_reliability_tracker()
                
                # Update evidence bundle with dynamic authority scores
                for evidence in evidence_bundle:
                    domain = evidence.get('domain', '')
                    if domain:
                        dynamic_authority = outlet_tracker.get_outlet_authority(domain)
                        evidence['dynamic_authority'] = dynamic_authority
                        logging.info(f"Domain {domain} - Dynamic authority: {dynamic_authority}")
                
                # Module A: PR Detection
                pr_detector = PRDetectionEngine()
                pr_analysis = pr_detector.analyze_content(
                    content=" ".join([e.get('text', '')[:1000] for e in evidence_bundle[:5]]),
                    sources=[e.get('url', '') for e in evidence_bundle]
                )
                
                # Module B: Source Independence Graph
                sig_analyzer = SourceIndependenceGraph()
                for source in evidence_bundle:
                    sig_analyzer.add_source(
                        source_id=source.get('url', ''),
                        content=source.get('text', ''),
                        metadata={
                            'title': source.get('title', ''),
                            'domain': source.get('domain', ''),
                            'published_date': source.get('published_date')
                        }
                    )
                sig_analysis = sig_analyzer.analyze_independence()
                
                # Module C: Enhanced 7-Axis Credibility Scoring
                if V2_FEATURES_AVAILABLE:
                    from datetime import datetime
                    credibility_engine = CredibilityEngine()
                    sources_for_scoring = [
                        Source(
                            url=e.get('url', ''),
                            content=e.get('text', ''),
                            domain=e.get('domain', ''),
                            timestamp=datetime.now()
                        ) for e in evidence_bundle
                    ]
                    # Extract evidence texts for credibility analysis
                    evidence_texts = [e.get('text', '') for e in evidence_bundle]
                    
                    credibility_scores = credibility_engine.calculate_credibility(
                        claim=topic,
                        sources=sources_for_scoring,
                        evidence_texts=evidence_texts,
                        sig_result=sig_analysis,
                        pr_results=pr_analysis
                    )
                else:
                    credibility_scores = {"overall_score": 0.7, "method": "fallback"}
                
                # Phase 2: Media Forensics Analysis
                from agents.media_forensics import get_media_forensics_engine
                forensics_engine = get_media_forensics_engine()
                media_forensics = forensics_engine.analyze_media(
                    claim=topic,
                    evidence_data=evidence_bundle
                )
                logging.info(f"Media forensics complete - Images analyzed: {media_forensics['images_analyzed']}")
                
                # Phase 4: Cross-Ecosystem Fact-Check Verification
                from agents.factcheck_integration import get_factcheck_integration
                from agents.social_monitoring import get_social_monitoring
                
                factcheck_engine = get_factcheck_integration()
                factcheck_results = factcheck_engine.verify_claim(topic)
                logging.info(f"Fact-check verification complete - Sources: {len(factcheck_results['factcheck_results'])}")
                
                social_monitor = get_social_monitoring()
                social_analysis = social_monitor.analyze_social_spread(topic)
                logging.info(f"Social monitoring complete - Viral velocity: {social_analysis['viral_velocity']}")
                
                # Module D: Reasoning-Rich Verdict Synthesis
                verdict_synthesizer = VerdictSynthesizer()
                v3_analysis = verdict_synthesizer.synthesize_verdict(
                    claim=topic,
                    evidence_data=evidence_bundle,
                    credibility_scores=credibility_scores,
                    pr_analysis=pr_analysis,
                    sig_analysis=sig_analysis
                )
                
                # Attach Phase 2 forensics results to v3 analysis
                v3_analysis['media_forensics'] = media_forensics
                
                # Attach Phase 4 fact-check and social monitoring results
                v3_analysis['factcheck_verification'] = factcheck_results
                v3_analysis['social_monitoring'] = social_analysis
                
                # Phase 3: Record verdict outcome for outlet reliability learning
                verdict_determination = v3_analysis['verdict']['determination']
                verdict_confidence = v3_analysis['verdict']['confidence_score']
                for evidence in evidence_bundle:
                    domain = evidence.get('domain', '')
                    if domain:
                        outlet_tracker.record_verdict_result(
                            domain=domain,
                            verdict=verdict_determination,
                            claim=topic,
                            confidence=verdict_confidence
                        )
                
                logging.info(f"v3 Analysis complete: {v3_analysis['verdict']['determination']}")
                
            except Exception as e:
                logging.error(f"v3 analysis failed: {e}", exc_info=True)
                v3_analysis = None
        
        # Create context
        if evidence_bundle:
            # Build richer evidence context: include title, url/domain and a longer excerpt
            context_items = []
            for article in evidence_bundle[:5]:
                title = article.get('title', 'N/A')
                url = article.get('url', '')
                domain = article.get('domain') or (url.split('/')[2] if url else 'N/A')
                excerpt = article.get('text', '')[:2000]
                context_items.append(
                    f"Source: {title}\nURL: {url}\nDomain: {domain}\nExcerpt:\n{excerpt}"
                )
            context = "\n\n".join(context_items)
            
            # Enhance with v3 analysis if available
            if v3_analysis:
                v3_summary = f"\n\nATLAS v3 Intelligence Assessment:\n"
                v3_summary += f"Verdict: {v3_analysis['verdict']['determination']} (Confidence: {v3_analysis['verdict']['confidence_level']})\n"
                v3_summary += f"PR Detection: {'Yes' if v3_analysis['pr_bias_analysis']['pr_detection']['is_pr'] else 'No'}\n"
                v3_summary += f"Source Independence: {v3_analysis['pr_bias_analysis']['source_independence']['independence_score']:.2f}\n"
                context += v3_summary
            
            user_message = f"Question: {topic}\n\nEvidence:\n{context}"
        else:
            user_message = f"Question: {topic}"
        
        # Add conversation history to provide context
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n".join([
                f"{msg['role'].title()}: {msg['content'][:500]}"  # Limit each message to 500 chars
                for msg in conversation_history[-6:]  # Only last 6 messages to save tokens
            ])
            user_message = f"Previous conversation:\n{history_text}\n\n{user_message}"
            logging.info(f"?? Added {len(conversation_history[-6:])} messages to context")
        
        system_prompt = """You are Atlas, an AI misinformation fighter.
        Today's date is November 12, 2025. You have knowledge up to 2025 and can discuss current events, trends, and updates from 2025.

        IMPORTANT CONTEXT AWARENESS: 
        - You are in a continuous conversation. If the user refers to previous messages (like "this link", "that article", "as I mentioned"), look for that information in the conversation history provided.
        - DO NOT ask for information that was already provided in the conversation history.
        - If the user provides a link or specific information, use it directly without asking them to provide it again.

        IMPORTANT: Use the provided Evidence block that follows the user's question. Do NOT rely solely on your internal knowledge cutoff. Instead:
        - Primarily base your answer on the Evidence provided (do not hallucinate new facts).
        - Explicitly cite the most relevant sources by title and domain and include URLs where available.
        - If sources conflict, summarize the differences and indicate uncertainty.
        - If the Evidence is insufficient to reach a conclusion, say so and point to reputable news outlets or official statements.
        - Keep your answer concise (2-3 short paragraphs), and at the end include a short 'Sources' list with titles and URLs.
        """
        
        # ?? MEMORY INTEGRATION + WEB RAG: Build complete context payload with external web content
        if memory:
            try:
                # Use build_context_payload for complete RAG (internal memories + external web)
                # This enables the Permanent Learning Loop!
                context_payload = memory.build_context_payload(
                    system_prompt=system_prompt,
                    current_task=user_message,
                    query=topic,  # Will extract URLs and fetch if present
                    enable_web_rag=True,  # Enable External RAG + Learning Loop
                    use_long_term=True,   # Search Vector DB for relevant memories
                    use_short_term=True,  # Include recent conversation
                    format_style="conversational"  # Better for chat UI
                )
                
                # Replace user_message with enriched payload
                user_message = context_payload
                logging.info(f"?? Enhanced with RAG context (web + memories)")
                
            except Exception as e:
                logging.warning(f"Memory context enhancement failed: {e}")
        
        # Generate response - FIX: Collect generator properly
        ai_agent = AiAgent()
        full_response = ""
        
        def collect_stream():
            """Helper function to collect stream in thread"""
            result = ""
            try:
                stream_gen = ai_agent.stream(
                    user_message=user_message,
                    system_prompt=system_prompt,
                    max_tokens=500
                )
                for chunk in stream_gen:  # Use for loop instead of next()
                    result += chunk
            except Exception as e:
                logging.error(f"Stream collection error: {e}")
            return result
        
        # Run in executor with timeout
        try:
            full_response = await asyncio.wait_for(
                loop.run_in_executor(executor, collect_stream),
                timeout=120.0  # Increased to 2 minutes for complex analysis
            )
        except asyncio.TimeoutError:
            full_response = "Response generation timed out. Please try a simpler question."
        
        if not full_response:
            full_response = "I couldn't generate a response. Please try again."
        
        # ?? MEMORY INTEGRATION: Store interaction in memory
        if memory:
            try:
                memory.add_interaction(
                    role="user",
                    content=topic,
                    metadata={"type": "question", "model": model},
                    store_in_rag=False  # Don't RAG-store user questions
                )
                
                # Enhanced metadata for v3 intelligence
                assistant_metadata = {
                    "type": "analysis", 
                    "model": model, 
                    "sources": len(evidence_bundle)
                }
                
                # Add v3 verdict data to metadata for RAG storage
                if v3_analysis:
                    assistant_metadata["v3_verdict"] = v3_analysis["verdict"]["determination"]
                    assistant_metadata["v3_confidence"] = v3_analysis["verdict"]["confidence_score"]
                    assistant_metadata["v3_pr_detected"] = v3_analysis["pr_bias_analysis"]["pr_detection"]["is_pr"]
                    assistant_metadata["v3_independence_score"] = v3_analysis["pr_bias_analysis"]["source_independence"]["independence_score"]
                    logging.info(f"Adding v3 verdict to memory: {v3_analysis['verdict']['determination']}")
                
                memory.add_interaction(
                    role="assistant",
                    content=full_response,
                    metadata=assistant_metadata,
                    store_in_rag=True  # Store AI responses + v3 verdicts for future retrieval
                )
                logging.debug(f"?? Stored analysis in memory session {session_id}")
            except Exception as e:
                logging.warning(f"Failed to store in memory: {e}")
        
        logging.info(f"Response generated: {len(full_response)} characters")

        # Simplify sources list for the frontend (title + url + domain)
        sources_list = []
        for art in (evidence_bundle or [])[:5]:
            sources_list.append({
                'title': art.get('title', ''),
                'url': art.get('url', ''),
                'domain': art.get('domain') or (art.get('url','').split('/')[2] if art.get('url') else '')
            })

        # Return final result
        response_data = {
            "success": True,
            "topic": topic,
            "analysis": full_response,
            "model": model,
            "sources_used": len(evidence_bundle),
            "sources": sources_list,
            "session_id": session_id if memory else None,
            
            # --- GOD MODE: Add metadata for UI visualization ---
            "meta": {
                "rag_status": rag_status,
                "latency": round(time.time() - rag_start_time, 2),
                "memory_active": True if memory else False,
                "primary_source": sources_list[0]['domain'] if sources_list else None
            }
        }
        
        # Add v3 Truth Intelligence results if available
        if v3_analysis:
            response_data["v3_intelligence"] = {
                "verdict": v3_analysis["verdict"],
                "evidence_reasoning": v3_analysis.get("evidence_reasoning", {}),
                "pr_detection": v3_analysis.get("pr_bias_analysis", {}).get("pr_detection"),
                "pr_bias_analysis": v3_analysis.get("pr_bias_analysis", {}),
                "source_independence": v3_analysis.get("pr_bias_analysis", {}).get("source_independence"),
                "sig_analysis": v3_analysis.get("pr_bias_analysis", {}).get("source_independence"),
                "confidence_breakdown": v3_analysis["confidence_breakdown"],
                "transparency_notes": v3_analysis.get("pr_bias_analysis", {}).get("transparency_notes", []),
                "media_forensics": {},  # Placeholder for Phase 2
                "metadata": v3_analysis["metadata"]
            }
            logging.info("Added v3 intelligence data to response")
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error in analyze_topic: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "analysis": "Sorry, I encountered an error processing your request."
        }), 500

@app.route("/ocr_upload", methods=["POST"])
async def ocr_upload():
    """
    Handle image upload and OCR processing (with memory support).
    Extracts text from image and optionally analyzes it with AI.
    Uses EasyOCR - no Tesseract installation required!
    """
    if not OCR_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "OCR functionality not available. Please install dependencies: pip install easyocr pillow"
        }), 503
    
    try:
        files = await request.files
        
        if 'image' not in files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = files['image']
        
        if not image_file.filename:
            return jsonify({"error": "Empty filename"}), 400
        
        # Check file format
        from services.ocr_processor import OCRProcessor
        if not OCRProcessor.is_supported_format(image_file.filename):
            return jsonify({
                "error": f"Unsupported file format. Supported formats: {', '.join(OCRProcessor.get_supported_formats())}"
            }), 400
        
        # Read image bytes - FileStorage.read() is synchronous, not async
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        # Log file info
        logging.info(f"Processing OCR for image: {image_file.filename} ({len(image_bytes)} bytes)")
        
        # Check if OCR is available (lazy load)
        ocr_processor = get_ocr_processor_lazy()
        if ocr_processor is None:
            return jsonify({
                "success": False,
                "error": "OCR functionality not available"
            }), 503
        
        # Process OCR
        loop = asyncio.get_running_loop()
        
        # Run OCR in executor to avoid blocking
        ocr_result = await loop.run_in_executor(
            executor,
            ocr_processor.extract_text_from_bytes,
            image_bytes
        )
        
        if not ocr_result["success"]:
            return jsonify({
                "success": False,
                "error": ocr_result.get("error", "OCR processing failed")
            }), 500
        
        extracted_text = ocr_result["text"]
        
        # Get analysis request from form data
        form_data = await request.form
        analyze = form_data.get('analyze', 'true').lower() == 'true'
        use_scraper = form_data.get('use_scraper', 'true').lower() == 'true'
        question = form_data.get('question', '')
        session_id = form_data.get('session_id')  # Optional: OCR conversation session
        
        # ?? FAST PATH: If not analyzing, return OCR result immediately
        if not analyze:
            logging.info(f"? Returning OCR-only result (no analysis requested)")
            return jsonify({
                "success": True,
                "ocr_result": {
                    "text": extracted_text,
                    "confidence": ocr_result["confidence"],
                    "word_count": ocr_result["word_count"]
                },
                "ai_analysis": None,
                "evidence_count": 0,
                "evidence_sources": [],
                "filename": image_file.filename,
                "session_id": None
            })
        
        # ?? MEMORY INTEGRATION: Initialize memory for OCR analysis
        memory = None
        if MEMORY_AVAILABLE and analyze:
            try:
                memory = get_memory_manager()
                if session_id:
                    memory.set_debate_context(session_id)
                    logging.info(f"?? Using existing OCR session: {session_id}")
                else:
                    session_id = str(uuid.uuid4())
                    memory.set_debate_context(session_id)
                    logging.info(f"?? Created new OCR session: {session_id}")
            except Exception as e:
                logging.warning(f"Memory initialization failed: {e}")
                memory = None
        
        ai_analysis = None
        evidence_articles = []
        
        if analyze and extracted_text:
            # If scraper is enabled, gather evidence first
            if use_scraper and len(extracted_text.split()) > 10:
                try:
                    logging.info(f"?? Using scraper to gather evidence for OCR text...")
                    
                    # Use first 200 chars or key phrases as search query
                    search_query = extracted_text[:200] if len(extracted_text) > 200 else extracted_text
                    
                     # Get evidence from scraper (async function)
                    evidence_articles = await get_diversified_evidence(

                        search_query,
                        3  # Get 3 articles for evidence
                    )
                    
                    logging.info(f"? Gathered {len(evidence_articles)} evidence articles")
                    
                except Exception as scraper_error:
                    logging.error(f"Scraper error: {scraper_error}", exc_info=True)
                    # Continue without scraper evidence
            
            # Prepare AI analysis with evidence
            if evidence_articles:
                # Build context from evidence
                evidence_context = "\n\n---EVIDENCE FROM WEB SOURCES---\n\n"
                for idx, article in enumerate(evidence_articles, 1):
                    evidence_context += f"Source {idx}: {article.get('title', 'Unknown')}\n"
                    evidence_context += f"URL: {article.get('url', 'N/A')}\n"
                    summary = article.get('summary') or article.get('text', '')[:300]
                    evidence_context += f"Content: {summary}...\n\n"
                
                if question:
                    user_message = f"""Here is text extracted from an image:

{extracted_text}

{evidence_context}

User's question: {question}

Please analyze the extracted text using the evidence provided from web sources. Verify claims, identify any misinformation, and provide a fact-checked analysis."""
                else:
                    user_message = f"""Here is text extracted from an image:

{extracted_text}

{evidence_context}

Please analyze this text using the evidence provided from web sources. Verify the accuracy of any claims, identify potential misinformation, and provide a comprehensive fact-checked analysis."""
                
                system_prompt = """You are Atlas, an advanced misinformation fighter and fact-checker.
Today's date is November 12, 2025. You have knowledge up to 2025 and can discuss current events.
You have been provided with text extracted from an image along with evidence from credible web sources.
Your task is to:
1. Identify key claims or information in the extracted text
2. Cross-reference with the provided evidence
3. Verify accuracy and flag any misinformation
4. Provide a clear, evidence-based analysis
5. Cite sources when referencing evidence

Be thorough, objective, and help users understand the truth."""
            else:
                # No evidence available, proceed with basic analysis
                if question:
                    user_message = f"Here is text extracted from an image:\n\n{extracted_text}\n\nUser's question: {question}"
                else:
                    user_message = f"Here is text extracted from an image:\n\n{extracted_text}\n\nPlease analyze this text and provide insights."
                
                system_prompt = """You are Atlas, an AI assistant helping analyze text from images.
Today's date is November 12, 2025. You have knowledge up to 2025.
Provide clear, helpful analysis of the text content.
If the text appears to contain claims or information, verify its accuracy."""
            
            # ?? MEMORY INTEGRATION: Retrieve relevant context from memory for OCR
            if memory:
                try:
                    # Retrieve relevant memories without zone formatting
                    relevant_memories = []
                    if memory.enable_rag and memory.long_term:
                        search_results = memory.long_term.search(
                            query=extracted_text[:100],
                            top_k=2,
                            filter_metadata={"debate_id": session_id}
                        )
                        # RetrievalResult is a dataclass with .text attribute
                        relevant_memories = [
                            f"Previous context: {result.text[:200]}..."
                            for result in search_results
                        ]
                    
                    # Add memory context naturally if available
                    if relevant_memories:
                        memory_context = "\n\n".join(relevant_memories)
                        user_message = f"{user_message}\n\nRelevant previous analysis:\n{memory_context}"
                        logging.info(f"?? Added {len(relevant_memories)} relevant OCR memories to context")
                    
                except Exception as e:
                    logging.warning(f"Memory context retrieval failed: {e}")
            
            # Generate AI response
            ai_agent = AiAgent()
            
            def collect_ai_stream():
                result = ""
                try:
                    stream_gen = ai_agent.stream(
                        user_message=user_message,
                        system_prompt=system_prompt,
                        max_tokens=800
                    )
                    for chunk in stream_gen:
                        result += chunk
                except Exception as e:
                    logging.error(f"AI analysis error: {e}")
                return result
            
            try:
                ai_analysis = await asyncio.wait_for(
                    loop.run_in_executor(executor, collect_ai_stream),
                    timeout=120.0  # Increased to 2 minutes for AI analysis
                )
            except asyncio.TimeoutError:
                ai_analysis = "Analysis timed out. Please try again."
            
            # ?? MEMORY INTEGRATION: Store OCR analysis in memory
            if memory and ai_analysis:
                try:
                    memory.add_interaction(
                        role="user",
                        content=f"OCR Text: {extracted_text[:200]}...",
                        metadata={"type": "ocr_input", "filename": image_file.filename},
                        store_in_rag=False
                    )
                    memory.add_interaction(
                        role="assistant",
                        content=ai_analysis,
                        metadata={"type": "ocr_analysis", "evidence_count": len(evidence_articles)},
                        store_in_rag=True  # Store analysis for future reference
                    )
                    logging.debug(f"?? Stored OCR analysis in memory session {session_id}")
                except Exception as e:
                    logging.warning(f"Failed to store OCR in memory: {e}")
        
        return jsonify({
            "success": True,
            "ocr_result": {
                "text": extracted_text,
                "confidence": ocr_result["confidence"],
                "word_count": ocr_result["word_count"]
            },
            "ai_analysis": ai_analysis,
            "evidence_count": len(evidence_articles),
            "evidence_sources": [
                {
                    "title": article.get('title', 'Unknown'),
                    "url": article.get('url', ''),
                    "domain": article.get('domain', ''),
                    "summary": article.get('summary', '')
                }
                for article in evidence_articles
            ] if evidence_articles else [],
            "filename": image_file.filename,
            "session_id": session_id if memory else None  # Return session ID for follow-up questions
        })
        
    except Exception as e:
        logging.error(f"Error in OCR upload: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/transcribe', methods=['POST'])
@limit('30/minute')
async def transcribe_audio():
    """Accepts an uploaded audio file (field name 'audio') and returns a transcript.
    Uses OpenAI Speech-to-Text when `OPENAI_API_KEY` is configured. Otherwise returns 503.
    """
    try:
        files = await request.files
        if 'audio' not in files:
            return jsonify({"success": False, "error": "No audio file provided (form field 'audio')."}), 400

        audio_file = files['audio']
        if not audio_file.filename:
            return jsonify({"success": False, "error": "Empty filename provided."}), 400

        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"success": False, "error": "Empty audio file."}), 400

        # If OPENAI_API_KEY is not configured, return explanatory error
        if not OPENAI_API_KEY:
            logging.warning('Transcription requested but OPENAI_API_KEY is not set')
            return jsonify({
                "success": False,
                "error": "Speech-to-text provider is not configured on the server. Set OPENAI_API_KEY to enable transcription."
            }), 503

        # Use openai client if available
        try:
            from openai import OpenAI
        except Exception as e:
            logging.error(f"openai package not available: {e}")
            return jsonify({"success": False, "error": "Server missing 'openai' package. Install requirements."}), 500

        loop = asyncio.get_running_loop()

        def _call_openai_transcribe():
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                bio = io.BytesIO(audio_bytes)
                # Let OpenAI detect the audio format; use 'whisper-1' model
                resp = client.audio.transcriptions.create(file=bio, model='whisper-1')
                # The client returns an object with 'text' field
                text = getattr(resp, 'text', None) or resp.get('text') if isinstance(resp, dict) else None
                return text or ''
            except Exception as err:
                logging.error(f"OpenAI transcription error: {err}", exc_info=True)
                raise

        try:
            transcript = await loop.run_in_executor(executor, _call_openai_transcribe)
        except Exception as e:
            return jsonify({"success": False, "error": f"Transcription failed: {str(e)}"}), 500

        return jsonify({"success": True, "transcript": transcript}), 200


    except Exception as e:
        logging.error(f"Error in /transcribe: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------
# Main
# -----------------------------
# Initialize database on module load (needed for uvicorn/hypercorn)
async def initialize():
    await AsyncDbManager.init_db()
    logging.info("Database has been initialized.")

# Set Windows event loop policy
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Startup event for Quart (runs when server starts)
@app.before_serving
async def startup():
    """Initialize database before server starts accepting requests"""
    await AsyncDbManager.init_db()
    logging.info("Database has been initialized.")
    # Initialize chat persistence DB if chat API is available
    try:
        if 'CHAT_API_AVAILABLE' in globals() and CHAT_API_AVAILABLE:
            # Import here to avoid import-time dependency issues with pymongo/motor
            from services.chat_store import init_chat_db
            await init_chat_db()
            logging.info("Chat persistence database initialized.")
    except Exception as e:
        logging.warning(f"Failed to initialize chat DB: {e}")

if __name__ == "__main__":
    if DEBUG_MODE and os.path.exists(DATABASE_FILE):
        try:
            logging.warning(f"Development mode: Removing existing database file '{DATABASE_FILE}'.")
            os.remove(DATABASE_FILE)
        except PermissionError:
            logging.error(
                f"Could not remove '{DATABASE_FILE}' because it is in use. "
                "Please close any other programs that might be using it and restart the server."
            )
            exit()

    if not os.path.exists("templates"):
        os.makedirs("templates")
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("<h1>AI Debate Server Chat Interface</h1>")

    logging.info("Starting Quart server on http://127.0.0.1:8000")
    
    # Use Quart with explicit Hypercorn config for Windows compatibility
    from hypercorn.config import Config
    from hypercorn.asyncio import serve
    import asyncio
    
    config = Config()
    config.bind = ["127.0.0.1:8000"]  # Bind to localhost for Windows compatibility
    config.use_reloader = False
    config.workers = 1  # Single worker for Windows
    config.accesslog = "-"  # Log to stdout
    config.errorlog = "-"   # Log errors to stdout
    
    # Set Windows-compatible event loop
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run with asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve(app, config))