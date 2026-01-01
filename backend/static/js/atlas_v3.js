/**
 * ATLAS v3 Truth Intelligence UI Components
 * Displays PR detection, source independence, 7-axis scores, and reasoning-rich verdicts
 */

class AtlasV3UI {
    constructor() {
        this.v3Container = null;
    }

    /**
     * Render complete v3 intelligence panel
     */
    renderV3Intelligence(v3Data, containerSelector = '#v3-intelligence-panel') {
        if (!v3Data) {
            console.log('No v3 intelligence data available');
            return;
        }

        this.v3Container = document.querySelector(containerSelector);
        if (!this.v3Container) {
            // Create container if doesn't exist
            this.v3Container = document.createElement('div');
            this.v3Container.id = 'v3-intelligence-panel';
            this.v3Container.className = 'v3-intelligence-panel';
            
            // Insert after chat messages container
            const chatMessages = document.querySelector('.chat-messages');
            if (chatMessages) {
                chatMessages.parentNode.insertBefore(this.v3Container, chatMessages.nextSibling);
            } else {
                document.body.appendChild(this.v3Container);
            }
        }

        // Build complete UI
        const html = `
            <div class="v3-header">
                <h3>ATLAS v3 Truth Intelligence Analysis</h3>
                <button class="v3-toggle" onclick="atlasV3.togglePanel()">
                    <span class="toggle-icon">▼</span>
                </button>
            </div>
            
            <div class="v3-content" id="v3-content">
                ${this.renderVerdict(v3Data.verdict, v3Data.evidence_reasoning)}
                ${this.renderFactCheckVerification(v3Data.factcheck_verification)}
                ${this.renderSocialMonitoring(v3Data.social_monitoring)}
                ${this.renderPRDetection(v3Data.pr_detection || v3Data.pr_bias_analysis)}
                ${this.renderSourceIndependence(v3Data.source_independence || v3Data.sig_analysis)}
                ${this.renderEvidencePanel(v3Data.evidence_reasoning)}
                ${this.renderConfidenceBreakdown(v3Data.confidence_breakdown)}
                ${this.renderMediaForensics(v3Data.media_forensics)}
                ${this.renderTransparencyNotes(v3Data.transparency_notes)}
            </div>
        `;

        this.v3Container.innerHTML = html;
        this.v3Container.style.display = 'block';
    }

    /**
     * Render verdict section with explanation points
     */
    renderVerdict(verdict, evidenceReasoning) {
        if (!verdict) return '';

        const verdictClass = this.getVerdictClass(verdict.determination);
        const confidenceClass = this.getConfidenceClass(verdict.confidence_level);
        const truthLabel = this.getTruthLabel(verdict.determination);

        return `
            <div class="v3-section verdict-section">
                <div class="verdict-header">
                    <h4>Verdict</h4>
                    <div class="verdict-badges">
                        <span class="verdict-badge ${verdictClass}">
                            ${verdict.determination}
                        </span>
                        <span class="truth-label ${verdictClass}">
                            ${truthLabel}
                        </span>
                    </div>
                </div>
                
                <p class="verdict-summary">${verdict.summary}</p>
                
                ${evidenceReasoning && evidenceReasoning.explanation_points ? `
                    <div class="reasoning-explanation">
                        <h5>Why this verdict:</h5>
                        <ul class="explanation-points">
                            ${evidenceReasoning.explanation_points.map(point => `
                                <li>${point}</li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                <div class="confidence-display">
                    <span class="confidence-label">Confidence:</span>
                    <span class="confidence-badge ${confidenceClass}">
                        ${verdict.confidence_level}
                    </span>
                    <span class="confidence-score">${(verdict.confidence_score * 100).toFixed(0)}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" 
                             style="width: ${verdict.confidence_score * 100}%">
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render PR detection results
     */
    renderPRDetection(prData) {
        if (!prData) return '';

        const isPR = prData.is_pr;
        const confidence = prData.confidence;

        return `
            <div class="v3-section pr-detection-section">
                <h4>PR & Propaganda Detection</h4>
                
                <div class="pr-status ${isPR ? 'pr-detected' : 'pr-clear'}">
                    <span class="pr-icon">${isPR ? '⚠️' : '✓'}</span>
                    <span class="pr-text">
                        ${isPR ? 'PR Content Detected' : 'No Significant PR Detected'}
                    </span>
                    ${isPR ? `<span class="pr-confidence">(${(confidence * 100).toFixed(0)}% confidence)</span>` : ''}
                </div>
                
                ${isPR && prData.indicators && prData.indicators.length > 0 ? `
                    <div class="pr-indicators">
                        <h5>Detected Indicators:</h5>
                        <ul class="indicators-list">
                            ${prData.indicators.slice(0, 5).map(indicator => `
                                <li class="indicator-item">${this.formatIndicator(indicator)}</li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${prData.press_release_origin ? `
                    <div class="pr-warning">
                        <strong>Note:</strong> Content appears to originate from a press release.
                    </div>
                ` : ''}
                
                ${prData.syndication_pattern ? `
                    <div class="pr-warning">
                        <strong>Syndication Detected:</strong> This content may be republished across multiple outlets.
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Render source independence analysis
     */
    renderSourceIndependence(sigData) {
        if (!sigData) return '';

        const independenceScore = sigData.independence_score;
        const scoreClass = independenceScore >= 0.7 ? 'high' : independenceScore >= 0.4 ? 'medium' : 'low';

        return `
            <div class="v3-section sig-section">
                <h4>Source Independence Analysis</h4>
                
                <div class="independence-score-display">
                    <div class="score-label">Independence Score</div>
                    <div class="score-value ${scoreClass}">${independenceScore.toFixed(2)}</div>
                    <div class="score-bar">
                        <div class="score-fill ${scoreClass}" 
                             style="width: ${independenceScore * 100}%">
                        </div>
                    </div>
                </div>
                
                ${sigData.shared_origin_detected ? `
                    <div class="sig-warning">
                        <strong>⚠️ Shared Origin Detected:</strong> 
                        Multiple sources appear to derive from the same root source.
                    </div>
                ` : ''}
                
                ${sigData.narrative_coordination ? `
                    <div class="sig-warning">
                        <strong>⚠️ Coordinated Narrative:</strong> 
                        Evidence of coordinated messaging across sources.
                    </div>
                ` : ''}
                
                ${sigData.source_clusters && sigData.source_clusters.length > 0 ? `
                    <div class="source-clusters">
                        <h5>Source Clusters:</h5>
                        <p class="clusters-description">
                            ${sigData.source_clusters.length} cluster(s) of related sources identified
                        </p>
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Render confidence breakdown
     */
    renderConfidenceBreakdown(breakdown) {
        if (!breakdown) return '';

        const factors = [
            { key: 'evidence_quantity', label: 'Evidence Quantity', icon: '📊' },
            { key: 'source_credibility', label: 'Source Credibility', icon: '🔍' },
            { key: 'source_independence', label: 'Source Independence', icon: '🔗' },
            { key: 'evidence_consensus', label: 'Evidence Consensus', icon: '✓' }
        ];

        return `
            <div class="v3-section confidence-breakdown-section">
                <h4>Confidence Breakdown</h4>
                
                <div class="confidence-factors">
                    ${factors.map(factor => {
                        const value = breakdown[factor.key];
                        const percentage = (value * 100).toFixed(0);
                        const colorClass = value >= 0.7 ? 'high' : value >= 0.4 ? 'medium' : 'low';
                        
                        return `
                            <div class="confidence-factor">
                                <div class="factor-header">
                                    <span class="factor-icon">${factor.icon}</span>
                                    <span class="factor-label">${factor.label}</span>
                                    <span class="factor-value ${colorClass}">${percentage}%</span>
                                </div>
                                <div class="factor-bar">
                                    <div class="factor-fill ${colorClass}" 
                                         style="width: ${percentage}%">
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Render evidence panel showing analyzed sources
     */
    renderEvidencePanel(evidenceReasoning) {
        if (!evidenceReasoning || !evidenceReasoning.sources_analyzed) return '';

        const sources = evidenceReasoning.sources_analyzed;
        if (sources.length === 0) return '';

        return `
            <div class="v3-section evidence-panel-section">
                <h4>Evidence Sources Analyzed</h4>
                
                <div class="evidence-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Sources:</span>
                        <span class="stat-value">${sources.length}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Independent Articles:</span>
                        <span class="stat-value">${evidenceReasoning.independent_source_count || sources.length}</span>
                    </div>
                </div>
                
                <div class="sources-list">
                    ${sources.map((source, index) => `
                        <div class="source-item">
                            <div class="source-number">${index + 1}</div>
                            <div class="source-details">
                                <div class="source-domain">${source.domain || new URL(source.url).hostname}</div>
                                <a href="${source.url}" target="_blank" class="source-url" title="${source.url}">
                                    ${source.title || source.url}
                                </a>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Render media forensics section (Phase 2 - now fully functional)
     */
    renderMediaForensics(mediaData) {
        if (!mediaData || mediaData.images_analyzed === 0 || mediaData.images_analyzed === undefined) {
            // No images to analyze
            return `
                <div class="v3-section media-forensics-section">
                    <h4>Media Integrity Status</h4>
                    <p class="forensics-note">No images found in evidence to analyze.</p>
                </div>
            `;
        }
        
        const authenticity = mediaData.overall_authenticity_score || 100;
        const reverseResults = mediaData.reverse_image_results || [];
        const redFlags = mediaData.red_flags || [];
        
        // Determine authenticity class
        let authenticityClass = 'authenticity-high';
        if (authenticity < 50) authenticityClass = 'authenticity-low';
        else if (authenticity < 75) authenticityClass = 'authenticity-medium';
        
        return `
            <div class="v3-section media-forensics-section">
                <h4>Media Integrity Status</h4>
                
                <div class="forensics-summary">
                    <div class="authenticity-score ${authenticityClass}">
                        <span class="authenticity-label">Authenticity Score:</span>
                        <span class="authenticity-value">${authenticity}/100</span>
                    </div>
                    <div class="images-analyzed">
                        ${mediaData.images_analyzed || 0} image(s) analyzed
                    </div>
                </div>
                
                <div class="forensics-checks">
                    <div class="forensics-item ${reverseResults.length > 0 ? 'checked' : 'not-checked'}">
                        <span class="forensics-icon">${reverseResults.length > 0 ? '✓' : '—'}</span>
                        <span class="forensics-label">Reverse Image Search:</span>
                        <span class="forensics-status">
                            ${reverseResults.length > 0 ? `${reverseResults.length} match(es) found` : 'No matches found'}
                        </span>
                    </div>
                    
                    ${reverseResults.length > 0 ? `
                        <div class="reverse-image-details">
                            ${reverseResults.map((result, idx) => `
                                <div class="reverse-result">
                                    <strong>Image ${idx + 1}:</strong>
                                    ${result.earliest_date ? `First seen: <span class="date-badge">${result.earliest_date}</span>` : 'No date information'}
                                    ${result.matches_found ? ` - ${result.matches_found} total matches` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    <div class="forensics-item ${mediaData.metadata_analysis && Object.keys(mediaData.metadata_analysis).length > 0 ? 'checked' : 'not-checked'}">
                        <span class="forensics-icon">${mediaData.metadata_analysis && Object.keys(mediaData.metadata_analysis).length > 0 ? '✓' : '—'}</span>
                        <span class="forensics-label">Metadata Analysis:</span>
                        <span class="forensics-status">
                            ${mediaData.metadata_analysis && Object.keys(mediaData.metadata_analysis).length > 0 ? 'Completed' : 'Not Available'}
                        </span>
                    </div>
                    
                    <div class="forensics-item ${mediaData.tampering_detection && Object.keys(mediaData.tampering_detection).length > 0 ? 'checked' : 'not-checked'}">
                        <span class="forensics-icon">${mediaData.tampering_detection && Object.keys(mediaData.tampering_detection).length > 0 ? '⚠️' : '—'}</span>
                        <span class="forensics-label">Tampering Detection:</span>
                        <span class="forensics-status">
                            ${mediaData.tampering_detection && Object.keys(mediaData.tampering_detection).length > 0 ? 'Analysis complete' : 'No Data'}
                        </span>
                    </div>
                </div>
                
                ${redFlags.length > 0 ? `
                    <div class="forensics-red-flags">
                        <h5>⚠️ Red Flags Detected:</h5>
                        <ul class="red-flags-list">
                            ${redFlags.map(flag => `
                                <li class="red-flag-item severity-${flag.severity}">
                                    <span class="flag-type">${flag.type.replace(/_/g, ' ').toUpperCase()}</span>
                                    <span class="flag-description">${flag.description}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                <p class="forensics-note">
                    ${mediaData.images_analyzed > 0 ? 'Phase 2 media forensics analysis complete.' : 'Media forensics analysis will be available in Phase 2 implementation.'}
                </p>
            </div>
        `;
    }

    /**
     * Render fact-check verification section (Phase 4)
     */
    renderFactCheckVerification(factcheckData) {
        if (!factcheckData || (!factcheckData.factcheck_results?.length && !factcheckData.historical_matches?.length)) {
            return '';
        }

        const results = factcheckData.factcheck_results || [];
        const historical = factcheckData.historical_matches || [];
        const consensus = factcheckData.verdict_consensus || 'UNVERIFIED';
        const score = factcheckData.cross_verification_score || 0;

        let consensusClass = 'consensus-unverified';
        if (consensus === 'VERIFIED') consensusClass = 'consensus-verified';
        else if (consensus === 'DEBUNKED') consensusClass = 'consensus-debunked';
        else if (consensus === 'MIXED') consensusClass = 'consensus-mixed';

        return `
            <div class="v3-section factcheck-section">
                <h4>🔍 Fact-Check Verification (Phase 4)</h4>
                
                <div class="factcheck-summary">
                    <div class="consensus-badge ${consensusClass}">
                        <span class="consensus-label">Consensus:</span>
                        <span class="consensus-value">${consensus}</span>
                    </div>
                    <div class="cross-verification-score">
                        <span class="score-label">Cross-Verification:</span>
                        <span class="score-value">${score}/100</span>
                    </div>
                </div>

                ${historical.length > 0 ? `
                    <div class="historical-matches">
                        <h5>📋 Historical Matches:</h5>
                        ${historical.map(match => `
                            <div class="historical-match">
                                <span class="match-type">${match.type === 'exact_match' ? '✓ Exact Match' : '≈ Similar Claim'}</span>
                                ${match.previous_verdict ? `<span class="match-verdict verdict-${match.previous_verdict.toLowerCase()}">${match.previous_verdict}</span>` : ''}
                                ${match.verification_count ? `<span class="match-count">Verified ${match.verification_count}x</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                ` : ''}

                ${results.length > 0 ? `
                    <div class="factcheck-results">
                        <h5>🌐 Professional Fact-Checkers:</h5>
                        <ul class="factcheck-list">
                            ${results.map(result => `
                                <li class="factcheck-item">
                                    <div class="factcheck-source">${result.organization}</div>
                                    <div class="factcheck-verdict verdict-${(result.verdict || '').toLowerCase().replace(/\s/g, '-')}">${result.verdict}</div>
                                    ${result.url ? `<a href="${result.url}" target="_blank" class="factcheck-link">View Report →</a>` : ''}
                                    ${result.date ? `<span class="factcheck-date">${result.date}</span>` : ''}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}

                <p class="phase-note">✓ Phase 4: Cross-ecosystem verification complete</p>
            </div>
        `;
    }

    /**
     * Render social monitoring section (Phase 4)
     */
    renderSocialMonitoring(socialData) {
        if (!socialData || !socialData.platform_presence) {
            return '';
        }

        const velocity = socialData.viral_velocity || 0;
        const status = socialData.trending_status || 'dormant';
        const twitter = socialData.platform_presence.twitter || {};
        const reddit = socialData.platform_presence.reddit || {};
        const botIndicators = socialData.bot_indicators || [];

        let velocityClass = 'velocity-low';
        if (velocity >= 70) velocityClass = 'velocity-high';
        else if (velocity >= 40) velocityClass = 'velocity-medium';

        let statusClass = 'status-dormant';
        if (status === 'viral') statusClass = 'status-viral';
        else if (status === 'emerging') statusClass = 'status-emerging';
        else if (status === 'spreading') statusClass = 'status-spreading';

        return `
            <div class="v3-section social-monitoring-section">
                <h4>📱 Social Media Analysis (Phase 4)</h4>
                
                <div class="social-summary">
                    <div class="viral-velocity ${velocityClass}">
                        <span class="velocity-label">Viral Velocity:</span>
                        <span class="velocity-value">${velocity}/100</span>
                    </div>
                    <div class="trending-status ${statusClass}">
                        <span class="status-icon">${status === 'viral' ? '🔥' : status === 'emerging' ? '📈' : status === 'spreading' ? '➡️' : '💤'}</span>
                        <span class="status-text">${status.toUpperCase()}</span>
                    </div>
                </div>

                <div class="platform-presence">
                    ${twitter.present ? `
                        <div class="platform-card twitter-card">
                            <h5>𝕏 Twitter</h5>
                            <div class="platform-stats">
                                <span class="stat">Tweets: ${twitter.tweet_count || 0}</span>
                                <span class="stat">❤️ ${twitter.engagement?.likes || 0}</span>
                                <span class="stat">🔄 ${twitter.engagement?.retweets || 0}</span>
                            </div>
                            ${twitter.verified_accounts > 0 ? `<span class="verified-badge">✓ ${twitter.verified_accounts} verified accounts</span>` : ''}
                        </div>
                    ` : ''}

                    ${reddit.present ? `
                        <div class="platform-card reddit-card">
                            <h5>Reddit</h5>
                            <div class="platform-stats">
                                <span class="stat">Posts: ${reddit.post_count || 0}</span>
                                <span class="stat">⬆️ ${reddit.engagement?.upvotes || 0}</span>
                                <span class="stat">💬 ${reddit.engagement?.comments || 0}</span>
                            </div>
                        </div>
                    ` : ''}
                </div>

                ${botIndicators.length > 0 ? `
                    <div class="bot-indicators">
                        <h5>⚠️ Bot Activity Detected:</h5>
                        <ul class="bot-list">
                            ${botIndicators.map(indicator => `
                                <li class="bot-indicator severity-${indicator.severity}">
                                    <span class="indicator-type">${indicator.indicator.replace(/_/g, ' ').toUpperCase()}</span>
                                    <span class="indicator-desc">${indicator.description}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}

                <p class="phase-note">✓ Phase 4: Social media monitoring complete</p>
            </div>
        `;
    }

    /**
     * Render transparency notes
     */
    renderTransparencyNotes(notes) {
        if (!notes || notes.length === 0) return '';

        return `
            <div class="v3-section transparency-section">
                <h4>Transparency Notes</h4>
                <div class="transparency-notes">
                    ${notes.map(note => `
                        <div class="transparency-note">
                            <span class="note-icon">ℹ️</span>
                            <span class="note-text">${note}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Helper: Get CSS class for verdict type
     */
    getVerdictClass(determination) {
        const classMap = {
            'TRUE': 'verdict-true',
            'FALSE': 'verdict-false',
            'MISLEADING': 'verdict-misleading',
            'UNVERIFIABLE': 'verdict-unverifiable',
            'PR_CONTENT': 'verdict-pr'
        };
        return classMap[determination] || 'verdict-unknown';
    }

    /**
     * Helper: Get truth label for verdict
     */
    getTruthLabel(determination) {
        const labelMap = {
            'TRUE': 'LIKELY TRUE',
            'FALSE': 'LIKELY FALSE',
            'MISLEADING': 'MISLEADING',
            'UNVERIFIABLE': 'UNVERIFIABLE',
            'PR_CONTENT': 'PR CONTENT'
        };
        return labelMap[determination] || 'UNKNOWN';
    }

    /**
     * Helper: Get CSS class for confidence level
     */
    getConfidenceClass(level) {
        const classMap = {
            'HIGH': 'confidence-high',
            'MEDIUM': 'confidence-medium',
            'LOW': 'confidence-low'
        };
        return classMap[level] || 'confidence-unknown';
    }

    /**
     * Helper: Format PR indicator for display
     */
    formatIndicator(indicator) {
        const formatted = indicator
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
        return formatted;
    }

    /**
     * Toggle v3 panel visibility
     */
    togglePanel() {
        const content = document.getElementById('v3-content');
        const toggleIcon = document.querySelector('.toggle-icon');
        
        if (content && toggleIcon) {
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggleIcon.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggleIcon.textContent = '▶';
            }
        }
    }

    /**
     * Clear/hide v3 panel
     */
    clearPanel() {
        if (this.v3Container) {
            this.v3Container.style.display = 'none';
            this.v3Container.innerHTML = '';
        }
    }
}

// Global instance
const atlasV3 = new AtlasV3UI();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AtlasV3UI, atlasV3 };
}
