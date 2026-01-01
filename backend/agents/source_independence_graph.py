"""
Source Independence Graph (SIG) Framework (Module B) for ATLAS v3

Determines if multiple sources are truly independent or all originate from:
- Single PR agency
- Single news wire
- Single government press office
- Copy-paste clusters

Builds a graph where nodes are documents and edges represent similarity/dependence.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using basic similarity. Install: pip install scikit-learn")

logger = logging.getLogger(__name__)


@dataclass
class DocumentNode:
    """Represents a single document/article in the SIG"""
    url: str
    domain: str
    title: str
    text: str
    url_hash: str
    cluster_id: Optional[str] = None
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    is_original: bool = False
    origin_type: Optional[str] = None  # 'wire', 'pr', 'copy-paste', 'independent'


@dataclass
class SourceCluster:
    """Group of similar/dependent documents"""
    cluster_id: str
    origin_type: str  # 'PR', 'Wire', 'Copy-Paste', 'Independent'
    document_hashes: Set[str]
    representative_title: str
    origin_domain: Optional[str] = None
    similarity_threshold: float = 0.0


@dataclass
class SIGResult:
    """Results from Source Independence Graph analysis"""
    independence_index: float  # 0.0 to 1.0 (higher = more independent)
    clusters: List[SourceCluster]
    unique_source_count: int
    total_document_count: int
    duplicate_content_detected: bool
    evidence_map: Dict[str, any]  # For UI visualization
    explanation: str


class SourceIndependenceGraph:
    """
    Builds a similarity graph to detect source independence.
    
    Key Capabilities:
    1. Text similarity detection (cosine similarity via TF-IDF)
    2. Syndication clustering (group identical/near-identical content)
    3. Origin type classification (PR, wire service, independent)
    4. Independence scoring (how many truly unique perspectives)
    """
    
    # Known wire service domains
    WIRE_SERVICES = {
        'reuters.com', 'apnews.com', 'afp.com', 'upi.com',
        'pti.in', 'ani.in', 'ians.in'  # Indian wire services
    }
    
    # Known PR distribution domains
    PR_DOMAINS = {
        'prnewswire.com', 'businesswire.com', 'globenewswire.com',
        'marketwired.com', 'prweb.com'
    }
    
    # Similarity thresholds
    HIGH_SIMILARITY = 0.85  # Near-identical (copy-paste)
    MEDIUM_SIMILARITY = 0.65  # Likely same source (wire/PR)
    LOW_SIMILARITY = 0.40  # Some overlap (could be covering same story)
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vectorizer = None
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
        
        # Storage for incremental analysis
        self.sources = []  # List of source dicts
    
    def add_source(self, source_id: str, content: str, metadata: Dict):
        """
        Public API: Add a source to the graph
        
        Args:
            source_id: Unique identifier for the source
            content: Text content of the source
            metadata: Additional metadata (title, domain, url, etc.)
        """
        source = {
            'url': metadata.get('url', source_id),
            'domain': metadata.get('domain', ''),
            'title': metadata.get('title', 'Untitled'),
            'text': content,
            'url_hash': hashlib.sha256(source_id.encode()).hexdigest()
        }
        self.sources.append(source)
        self.logger.info(f"Added source: {source_id} (total: {len(self.sources)})")
    
    def analyze_independence(self) -> Dict[str, any]:
        """
        Public API: Analyze independence of all added sources
        
        Returns:
            Dict with independence analysis results
        """
        if not self.sources:
            return {
                'independence_score': 1.0,
                'independence_index': 1.0,
                'clusters': [],
                'source_clusters': [],
                'unique_source_count': 0,
                'independent_source_count': 0,
                'total_document_count': 0,
                'duplicate_content_detected': False,
                'shared_origin_detected': False,
                'explanation': 'No sources provided'
            }
        
        result = self.analyze_source_independence(self.sources)
        
        return {
            'independence_score': result.independence_index,
            'independence_index': result.independence_index,
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'origin_type': c.origin_type,
                    'document_count': len(c.document_hashes),
                    'representative_title': c.representative_title,
                    'origin_domain': c.origin_domain
                }
                for c in result.clusters
            ],
            'source_clusters': [c.origin_type for c in result.clusters],
            'unique_source_count': result.unique_source_count,
            'independent_source_count': result.unique_source_count,
            'total_document_count': result.total_document_count,
            'duplicate_content_detected': result.duplicate_content_detected,
            'shared_origin_detected': result.duplicate_content_detected,
            'evidence_map': result.evidence_map,
            'explanation': result.explanation
        }

    
    def analyze_source_independence(
        self,
        articles: List[Dict]
    ) -> SIGResult:
        """
        Main analysis method - builds SIG and calculates independence
        
        Args:
            articles: List of article dicts with keys: url, domain, title, text, url_hash
            
        Returns:
            SIGResult with independence analysis
        """
        if not articles or len(articles) < 2:
            return self._create_minimal_result(articles)
        
        self.logger.info(f"Building SIG for {len(articles)} documents...")
        
        # Step 1: Create document nodes
        nodes = self._create_document_nodes(articles)
        
        # Step 2: Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(nodes)
        
        # Step 3: Cluster documents by similarity
        clusters = self._cluster_documents(nodes, similarity_matrix)
        
        # Step 4: Classify cluster origins
        clusters = self._classify_cluster_origins(clusters, nodes)
        
        # Step 5: Calculate independence index
        independence_index = self._calculate_independence_index(clusters, len(nodes))
        
        # Step 6: Detect duplicate content
        duplicate_detected = any(c.origin_type in ['Copy-Paste', 'PR'] for c in clusters)
        
        # Step 7: Build evidence map for UI
        evidence_map = self._build_evidence_map(clusters, nodes)
        
        # Step 8: Generate explanation
        explanation = self._generate_explanation(
            independence_index, clusters, len(nodes)
        )
        
        # Count unique sources (clusters with origin_type='Independent' or different domains)
        unique_sources = self._count_unique_sources(clusters, nodes)
        
        return SIGResult(
            independence_index=independence_index,
            clusters=clusters,
            unique_source_count=unique_sources,
            total_document_count=len(nodes),
            duplicate_content_detected=duplicate_detected,
            evidence_map=evidence_map,
            explanation=explanation
        )
    
    def _create_document_nodes(self, articles: List[Dict]) -> List[DocumentNode]:
        """Convert article dicts to DocumentNode objects"""
        nodes = []
        for article in articles:
            node = DocumentNode(
                url=article.get('url', ''),
                domain=article.get('domain', ''),
                title=article.get('title', 'Untitled'),
                text=article.get('text', ''),
                url_hash=article.get('url_hash') or hashlib.sha256(
                    article.get('url', '').encode()
                ).hexdigest()
            )
            nodes.append(node)
        return nodes
    
    def _calculate_similarity_matrix(self, nodes: List[DocumentNode]) -> np.ndarray:
        """
        Calculate pairwise similarity between all documents
        
        Returns:
            NxN matrix where entry (i,j) is similarity between doc i and doc j
        """
        if not SKLEARN_AVAILABLE or len(nodes) < 2:
            # Fallback: basic Jaccard similarity
            return self._calculate_jaccard_similarity_matrix(nodes)
        
        # Use TF-IDF + cosine similarity for better accuracy
        try:
            # Extract text from nodes
            texts = [f"{node.title} {node.text[:2000]}" for node in nodes]
            
            # Vectorize
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            self.logger.info(f"Calculated similarity matrix: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"TF-IDF similarity failed: {e}")
            return self._calculate_jaccard_similarity_matrix(nodes)
    
    def _calculate_jaccard_similarity_matrix(self, nodes: List[DocumentNode]) -> np.ndarray:
        """Fallback: Simple Jaccard similarity based on word sets"""
        n = len(nodes)
        matrix = [[0.0] * n for _ in range(n)]
        
        # Tokenize all documents
        word_sets = []
        for node in nodes:
            words = set(re.findall(r'\w+', f"{node.title} {node.text}".lower()))
            word_sets.append(words)
        
        # Calculate pairwise Jaccard similarity
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    intersection = len(word_sets[i].intersection(word_sets[j]))
                    union = len(word_sets[i].union(word_sets[j]))
                    similarity = intersection / union if union > 0 else 0.0
                    matrix[i][j] = similarity
                    matrix[j][i] = similarity
        
        return np.array(matrix) if SKLEARN_AVAILABLE else matrix
    
    def _cluster_documents(
        self,
        nodes: List[DocumentNode],
        similarity_matrix
    ) -> List[SourceCluster]:
        """
        Cluster documents based on similarity thresholds
        
        Uses hierarchical clustering approach:
        - Documents with similarity > HIGH_SIMILARITY are in same cluster
        - Creates separate clusters for independent documents
        """
        n = len(nodes)
        visited = set()
        clusters = []
        cluster_id_counter = 0
        
        for i in range(n):
            if i in visited:
                continue
            
            # Start new cluster
            cluster_members = {i}
            visited.add(i)
            
            # Find all similar documents
            for j in range(n):
                if j == i or j in visited:
                    continue
                
                similarity = similarity_matrix[i][j]
                
                # High similarity = same cluster
                if similarity >= self.HIGH_SIMILARITY:
                    cluster_members.add(j)
                    visited.add(j)
                # Medium similarity = likely same origin
                elif similarity >= self.MEDIUM_SIMILARITY:
                    cluster_members.add(j)
                    visited.add(j)
            
            # Create cluster
            cluster_id = f"cluster_{cluster_id_counter}"
            cluster_id_counter += 1
            
            # Get representative title (first document)
            rep_title = nodes[i].title
            
            # Determine similarity threshold for this cluster
            if len(cluster_members) > 1:
                avg_similarity = np.mean([
                    similarity_matrix[i][j]
                    for i in cluster_members
                    for j in cluster_members
                    if i != j
                ])
            else:
                avg_similarity = 0.0
            
            cluster = SourceCluster(
                cluster_id=cluster_id,
                origin_type='Unknown',  # Will be classified in next step
                document_hashes={nodes[idx].url_hash for idx in cluster_members},
                representative_title=rep_title,
                similarity_threshold=float(avg_similarity)
            )
            
            # Assign cluster ID to nodes
            for idx in cluster_members:
                nodes[idx].cluster_id = cluster_id
            
            clusters.append(cluster)
        
        self.logger.info(f"Created {len(clusters)} clusters from {n} documents")
        return clusters
    
    def _classify_cluster_origins(
        self,
        clusters: List[SourceCluster],
        nodes: List[DocumentNode]
    ) -> List[SourceCluster]:
        """
        Classify each cluster's origin type:
        - PR: Originated from PR distribution service
        - Wire: Originated from wire service (Reuters, AP, etc.)
        - Copy-Paste: High similarity, no clear origin
        - Independent: Unique reporting
        """
        for cluster in clusters:
            # Get domains in this cluster
            cluster_domains = set()
            for node in nodes:
                if node.cluster_id == cluster.cluster_id:
                    cluster_domains.add(node.domain)
            
            # Check if any domain is a known PR source
            if cluster_domains.intersection(self.PR_DOMAINS):
                cluster.origin_type = 'PR'
                cluster.origin_domain = list(cluster_domains.intersection(self.PR_DOMAINS))[0]
            
            # Check if any domain is a wire service
            elif cluster_domains.intersection(self.WIRE_SERVICES):
                cluster.origin_type = 'Wire'
                cluster.origin_domain = list(cluster_domains.intersection(self.WIRE_SERVICES))[0]
            
            # High similarity but no known origin = copy-paste
            elif cluster.similarity_threshold >= self.HIGH_SIMILARITY and len(cluster.document_hashes) > 1:
                cluster.origin_type = 'Copy-Paste'
            
            # Low similarity or single document = independent
            else:
                cluster.origin_type = 'Independent'
        
        return clusters
    
    def _calculate_independence_index(
        self,
        clusters: List[SourceCluster],
        total_docs: int
    ) -> float:
        """
        Calculate independence index (0.0 to 1.0)
        
        Formula:
        - Count clusters with origin_type='Independent'
        - Penalize PR/Wire/Copy-Paste clusters
        - Normalize by total document count
        """
        if not clusters:
            return 0.0
        
        independent_clusters = [c for c in clusters if c.origin_type == 'Independent']
        pr_wire_clusters = [c for c in clusters if c.origin_type in ['PR', 'Wire']]
        copy_paste_clusters = [c for c in clusters if c.origin_type == 'Copy-Paste']
        
        # Base score: ratio of independent clusters
        base_score = len(independent_clusters) / len(clusters)
        
        # Penalty for PR/Wire content (reduces score by 0.2 per cluster)
        pr_penalty = min(0.4, len(pr_wire_clusters) * 0.2)
        
        # Penalty for copy-paste content (reduces score by 0.15 per cluster)
        copy_penalty = min(0.3, len(copy_paste_clusters) * 0.15)
        
        # Calculate final index
        independence_index = max(0.0, base_score - pr_penalty - copy_penalty)
        
        return round(independence_index, 2)
    
    def _count_unique_sources(
        self,
        clusters: List[SourceCluster],
        nodes: List[DocumentNode]
    ) -> int:
        """
        Count truly unique sources
        
        Logic:
        - Independent clusters count as 1 source each
        - PR/Wire clusters count as 1 source total (not N documents)
        - Copy-Paste clusters count as 1 source
        """
        unique_count = 0
        
        for cluster in clusters:
            if cluster.origin_type == 'Independent':
                # Each independent cluster is a unique source
                unique_count += len(cluster.document_hashes)
            elif cluster.origin_type in ['PR', 'Wire', 'Copy-Paste']:
                # Entire cluster is 1 source
                unique_count += 1
        
        return unique_count
    
    def _build_evidence_map(
        self,
        clusters: List[SourceCluster],
        nodes: List[DocumentNode]
    ) -> Dict:
        """
        Build evidence map for UI visualization
        
        Returns:
            Dict with cluster info and node relationships
        """
        evidence_map = {
            'clusters': [],
            'nodes': [],
            'edges': []
        }
        
        # Add clusters
        for cluster in clusters:
            evidence_map['clusters'].append({
                'id': cluster.cluster_id,
                'type': cluster.origin_type,
                'size': len(cluster.document_hashes),
                'title': cluster.representative_title,
                'origin_domain': cluster.origin_domain
            })
        
        # Add nodes
        for node in nodes:
            evidence_map['nodes'].append({
                'url_hash': node.url_hash,
                'domain': node.domain,
                'title': node.title,
                'cluster_id': node.cluster_id
            })
        
        # Edges are implicit (nodes in same cluster are connected)
        
        return evidence_map
    
    def _generate_explanation(
        self,
        independence_index: float,
        clusters: List[SourceCluster],
        total_docs: int
    ) -> str:
        """Generate human-readable explanation"""
        
        parts = [f"Independence Index: {independence_index:.2f}/1.0"]
        
        if independence_index >= 0.7:
            parts.append("✅ HIGH source independence - multiple unique perspectives")
        elif independence_index >= 0.4:
            parts.append("⚠️ MODERATE independence - some shared origins detected")
        else:
            parts.append("❌ LOW independence - significant content duplication/syndication")
        
        # Count by origin type
        origin_counts = defaultdict(int)
        for cluster in clusters:
            origin_counts[cluster.origin_type] += 1
        
        if origin_counts:
            origin_summary = ", ".join([f"{count} {otype}" for otype, count in origin_counts.items()])
            parts.append(f"Clusters: {origin_summary}")
        
        parts.append(f"Total documents: {total_docs}")
        
        return " | ".join(parts)
    
    def _create_minimal_result(self, articles: List[Dict]) -> SIGResult:
        """Create result when insufficient documents for analysis"""
        return SIGResult(
            independence_index=1.0 if len(articles) == 1 else 0.0,
            clusters=[],
            unique_source_count=len(articles),
            total_document_count=len(articles),
            duplicate_content_detected=False,
            evidence_map={'clusters': [], 'nodes': [], 'edges': []},
            explanation=f"Insufficient documents for SIG analysis ({len(articles)} documents)"
        )


def analyze_source_independence(articles: List[Dict]) -> Dict:
    """
    Convenience function for source independence analysis
    
    Args:
        articles: List of article dicts
        
    Returns:
        Dict with SIG analysis results
    """
    sig = SourceIndependenceGraph()
    result = sig.analyze_source_independence(articles)
    
    return {
        'independence_index': result.independence_index,
        'unique_sources': result.unique_source_count,
        'total_documents': result.total_document_count,
        'duplicate_detected': result.duplicate_content_detected,
        'clusters': [
            {
                'id': c.cluster_id,
                'type': c.origin_type,
                'size': len(c.document_hashes),
                'title': c.representative_title,
                'origin': c.origin_domain
            }
            for c in result.clusters
        ],
        'evidence_map': result.evidence_map,
        'explanation': result.explanation
    }
