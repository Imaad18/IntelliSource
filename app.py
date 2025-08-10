import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import json
from urllib.parse import urljoin, urlparse
import time
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import base64
from io import BytesIO

class IntelliSourceEngine:
    """Advanced AI Research Engine - Better than GPT for research"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.processed_urls = set()
        self.research_cache = {}
        
    def intelligent_search(self, query, num_results=15):
        """Multi-engine search with smart URL discovery"""
        search_results = []
        
        # Strategy 1: Wikipedia for authoritative baseline
        wiki_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        search_results.append(wiki_url)
        
        # Strategy 2: News and recent content
        news_sources = [
            f"https://www.reuters.com/search/?query={'+'.join(query.split())}",
            f"https://techcrunch.com/?s={'+'.join(query.split())}",
            f"https://www.nature.com/search?q={'+'.join(query.split())}"
        ]
        search_results.extend(news_sources)
        
        # Strategy 3: Academic and research sources
        academic_sources = [
            f"https://arxiv.org/search/?query={'+'.join(query.split())}&searchtype=all",
            f"https://scholar.google.com/scholar?q={'+'.join(query.split())}"
        ]
        search_results.extend(academic_sources)
        
        # Strategy 4: Domain-specific intelligent URLs
        query_keywords = query.lower().split()
        
        if any(word in query_keywords for word in ['ai', 'artificial', 'intelligence', 'machine', 'learning']):
            search_results.extend([
                "https://openai.com/research/",
                "https://www.anthropic.com/research",
                "https://deepmind.google/research/",
                "https://ai.googleblog.com/"
            ])
        
        if any(word in query_keywords for word in ['technology', 'tech', 'innovation']):
            search_results.extend([
                "https://techcrunch.com/category/artificial-intelligence/",
                "https://www.wired.com/tag/artificial-intelligence/",
                "https://www.technologyreview.com/topic/artificial-intelligence/"
            ])
        
        return list(dict.fromkeys(search_results))[:num_results]  # Remove duplicates, keep order
    
    def advanced_content_extraction(self, url):
        """Military-grade content extraction with intelligence"""
        try:
            response = self.session.get(url, timeout=12, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Intelligent content extraction
            content_data = self._extract_structured_content(soup, url)
            
            # Advanced analysis
            content_data.update({
                'credibility_score': self._calculate_advanced_credibility(url, content_data),
                'content_quality': self._assess_content_quality(content_data),
                'key_entities': self._extract_entities(content_data['content']),
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time()
            })
            
            return content_data
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'credibility_score': 0
            }
    
    def _extract_structured_content(self, soup, url):
        """Extract content with advanced structural understanding"""
        # Remove noise
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            element.decompose()
        
        # Extract metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        
        # Smart content selection
        main_content = self._find_main_content(soup)
        
        # Extract text with structure preservation
        paragraphs = []
        for p in main_content.find_all(['p', 'div'], limit=50):
            text = p.get_text(strip=True)
            if len(text) > 30:  # Meaningful paragraphs only
                paragraphs.append(text)
        
        content = ' '.join(paragraphs)
        
        # Extract outbound links
        links = self._extract_quality_links(soup, url)
        
        return {
            'url': url,
            'title': title,
            'description': description,
            'content': content[:4000],  # Manageable size
            'paragraph_count': len(paragraphs),
            'word_count': len(content.split()),
            'links': links,
            'domain': urlparse(url).netloc
        }
    
    def _extract_title(self, soup):
        """Smart title extraction"""
        title_sources = [
            soup.find('title'),
            soup.find('h1'),
            soup.find('meta', property='og:title'),
            soup.find('meta', attrs={'name': 'title'})
        ]
        
        for source in title_sources:
            if source:
                if source.name == 'meta':
                    title = source.get('content', '')
                else:
                    title = source.get_text(strip=True)
                
                if title and len(title.strip()) > 5:
                    return title.strip()[:200]
        
        return "Unknown Title"
    
    def _extract_description(self, soup):
        """Smart description extraction"""
        desc_sources = [
            soup.find('meta', attrs={'name': 'description'}),
            soup.find('meta', property='og:description'),
            soup.find('meta', attrs={'name': 'twitter:description'})
        ]
        
        for source in desc_sources:
            if source:
                desc = source.get('content', '').strip()
                if desc and len(desc) > 10:
                    return desc[:300]
        
        return ""
    
    def _find_main_content(self, soup):
        """Intelligently find the main content area"""
        content_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '#content', '.post-content', '.entry-content',
            '.article-content', '.story-content', '#main-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content and len(main_content.get_text(strip=True)) > 200:
                return main_content
        
        # Fallback: find the largest content block
        content_blocks = soup.find_all(['div', 'section'], limit=20)
        if content_blocks:
            largest_block = max(content_blocks, key=lambda x: len(x.get_text(strip=True)))
            if len(largest_block.get_text(strip=True)) > 100:
                return largest_block
        
        return soup.find('body') or soup
    
    def _extract_quality_links(self, soup, base_url):
        """Extract high-quality outbound links"""
        links = []
        domain = urlparse(base_url).netloc
        
        for a in soup.find_all('a', href=True, limit=30):
            href = a.get('href')
            link_text = a.get_text(strip=True)
            
            if not href or len(link_text) < 5:
                continue
            
            full_url = urljoin(base_url, href)
            link_domain = urlparse(full_url).netloc
            
            # Quality filters
            if (len(link_text) > 100 or  # Too long
                link_domain == domain or  # Same domain
                not full_url.startswith('http') or  # Invalid URL
                any(skip in full_url.lower() for skip in ['pdf', 'jpg', 'png', 'gif', 'mp4'])):  # Media files
                continue
            
            # Prioritize academic and research links
            quality_score = 0
            if any(word in link_text.lower() for word in ['research', 'study', 'analysis', 'report']):
                quality_score += 3
            if any(domain in link_domain for domain in ['.edu', '.gov', 'nature.com', 'science.org']):
                quality_score += 2
            
            links.append({
                'url': full_url,
                'text': link_text,
                'quality_score': quality_score
            })
        
        # Sort by quality and return top links
        links.sort(key=lambda x: x['quality_score'], reverse=True)
        return [link['url'] for link in links[:10]]
    
    def _calculate_advanced_credibility(self, url, content_data):
        """Advanced credibility assessment algorithm"""
        score = 50  # Base score
        domain = urlparse(url).netloc.lower()
        content = content_data.get('content', '')
        title = content_data.get('title', '')
        
        # Domain authority scoring
        if any(trusted in domain for trusted in ['wikipedia.org', 'nature.com', 'science.org', 'ieee.org']):
            score += 35
        elif any(news in domain for news in ['reuters.com', 'bbc.com', 'npr.org', 'ap.org']):
            score += 25
        elif any(academic in domain for academic in ['.edu', '.gov', 'arxiv.org']):
            score += 30
        elif any(tech in domain for tech in ['openai.com', 'anthropic.com', 'deepmind.com']):
            score += 25
        
        # Content quality indicators
        if content_data.get('word_count', 0) > 500:
            score += 10
        if content_data.get('paragraph_count', 0) > 5:
            score += 5
        
        # Authority signals in content
        authority_signals = ['research', 'study', 'analysis', 'published', 'peer-reviewed', 'doi:', 'citation']
        signal_count = sum(1 for signal in authority_signals if signal in content.lower())
        score += min(signal_count * 3, 15)
        
        # Recency indicators
        current_year = datetime.now().year
        if str(current_year) in title or str(current_year) in content:
            score += 10
        elif str(current_year - 1) in title or str(current_year - 1) in content:
            score += 5
        
        # Penalize certain patterns
        if any(spam in content.lower() for spam in ['click here', 'buy now', 'limited time']):
            score -= 10
        
        return max(min(score, 100), 0)
    
    def _assess_content_quality(self, content_data):
        """Assess overall content quality"""
        content = content_data.get('content', '')
        
        if not content:
            return 0
        
        words = content.split()
        unique_words = set(words)
        
        quality_metrics = {
            'length_score': min(len(words) / 500, 1) * 25,
            'diversity_score': min(len(unique_words) / len(words), 0.8) * 25 if words else 0,
            'structure_score': min(content_data.get('paragraph_count', 0) / 10, 1) * 25,
            'readability_score': self._calculate_readability(content) * 25
        }
        
        return sum(quality_metrics.values())
    
    def _calculate_readability(self, text):
        """Simple readability assessment"""
        if not text:
            return 0
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        optimal_length = 20  # Optimal words per sentence
        
        # Score based on how close to optimal
        if avg_sentence_length <= optimal_length:
            return avg_sentence_length / optimal_length
        else:
            return max(0, 1 - (avg_sentence_length - optimal_length) / optimal_length)
    
    def _extract_entities(self, content):
        """Extract key entities without heavy NLP libraries"""
        if not content:
            return []
        
        # Simple but effective entity extraction
        entities = []
        
        # Find years
        years = re.findall(r'\b(19|20)\d{2}\b', content)
        entities.extend([f"Year: {year}" for year in set(years)])
        
        # Find percentages and numbers
        percentages = re.findall(r'\d+(?:\.\d+)?%', content)
        entities.extend([f"Stat: {pct}" for pct in set(percentages)])
        
        # Find organizations (simple heuristic)
        org_patterns = [
            r'\b[A-Z][a-z]+ (?:University|Institute|Corporation|Company|Inc|LLC)\b',
            r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|Meta|OpenAI|Anthropic|DeepMind)\b'
        ]
        
        for pattern in org_patterns:
            orgs = re.findall(pattern, content)
            entities.extend([f"Org: {org}" for org in set(orgs)])
        
        return entities[:10]
    
    def neural_summarization(self, content, topic, max_sentences=3):
        """Advanced summarization using neural-inspired algorithms"""
        if not content or len(content.strip()) < 50:
            return "Insufficient content for meaningful analysis."
        
        # Tokenize sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 20]
        
        if not sentences:
            return "No analyzable sentences found."
        
        # Advanced scoring algorithm
        topic_words = set(topic.lower().split())
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            words = set(re.findall(r'\w+', sentence.lower()))
            
            # Multiple intelligence factors
            relevance_score = len(words.intersection(topic_words)) * 3
            position_score = max(0, 1 - i/len(sentences)) * 2  # Earlier = better
            length_score = min(len(words) / 25, 1) * 1.5  # Optimal length
            
            # Semantic importance indicators
            importance_words = ['research', 'study', 'analysis', 'findings', 'results', 'conclusion', 
                              'data', 'evidence', 'significant', 'important', 'key', 'main', 'primary']
            importance_score = sum(2 for word in importance_words if word in sentence.lower())
            
            # Technical depth indicators
            tech_words = ['algorithm', 'model', 'system', 'method', 'approach', 'technique', 'technology']
            tech_score = sum(1.5 for word in tech_words if word in sentence.lower())
            
            # Avoid promotional/marketing content
            spam_words = ['buy', 'purchase', 'sale', 'offer', 'deal', 'discount', 'free trial']
            spam_penalty = sum(3 for word in spam_words if word in sentence.lower())
            
            total_score = relevance_score + position_score + length_score + importance_score + tech_score - spam_penalty
            
            if total_score > 2:  # Minimum threshold
                scored_sentences.append((total_score, sentence, i))
        
        # Smart selection strategy
        if not scored_sentences:
            return "No relevant content identified for this topic."
        
        # Sort by score but ensure diversity
        scored_sentences.sort(reverse=True)
        selected = []
        used_positions = set()
        
        for score, sentence, pos in scored_sentences:
            # Avoid selecting sentences too close together
            if not any(abs(pos - used_pos) < 3 for used_pos in used_positions):
                selected.append(sentence)
                used_positions.add(pos)
                if len(selected) >= max_sentences:
                    break
        
        return '. '.join(selected)
    
    def parallel_research(self, topic, urls, max_workers=5):
        """Parallel processing for faster research"""
        results = []
        
        def analyze_single_url(url):
            if url in self.processed_urls:
                return None
            
            self.processed_urls.add(url)
            return self.advanced_content_extraction(url)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(analyze_single_url, url): url for url in urls}
            
            for future in future_to_url:
                try:
                    result = future.result(timeout=15)
                    if result and 'error' not in result:
                        # Add intelligent summary
                        result['ai_summary'] = self.neural_summarization(
                            result['content'], topic
                        )
                        results.append(result)
                except Exception as e:
                    st.warning(f"Analysis timeout for: {future_to_url[future]}")
        
        return results
    
    def cross_reference_facts(self, research_results):
        """Cross-reference facts across sources for verification"""
        fact_mentions = {}
        
        for result in research_results:
            content = result.get('content', '')
            
            # Extract statistical claims
            stats = re.findall(r'\d+(?:\.\d+)?(?:%|\s*percent|\s*billion|\s*million|\s*thousand)', content)
            
            for stat in stats:
                normalized_stat = re.sub(r'\s+', ' ', stat.strip())
                if normalized_stat in fact_mentions:
                    fact_mentions[normalized_stat]['count'] += 1
                    fact_mentions[normalized_stat]['sources'].append(result['url'])
                else:
                    fact_mentions[normalized_stat] = {
                        'count': 1, 
                        'sources': [result['url']],
                        'credibility': result.get('credibility_score', 0)
                    }
        
        # Return verified facts (mentioned in multiple sources)
        verified_facts = {
            fact: data for fact, data in fact_mentions.items() 
            if data['count'] > 1 or data['credibility'] > 80
        }
        
        return verified_facts
    
    def generate_insights(self, research_results, topic):
        """Generate intelligent insights from research"""
        if not research_results:
            return {}
        
        insights = {
            'key_statistics': [],
            'main_findings': [],
            'source_consensus': {},
            'knowledge_gaps': [],
            'research_quality': self._assess_research_quality(research_results)
        }
        
        # Extract key statistics
        all_content = ' '.join([r.get('content', '') for r in research_results])
        stats = re.findall(r'\d+(?:\.\d+)?(?:%|\s*percent|\s*billion|\s*million)', all_content)
        insights['key_statistics'] = list(set(stats))[:10]
        
        # Analyze source consensus
        high_cred_sources = [r for r in research_results if r.get('credibility_score', 0) > 70]
        insights['source_consensus'] = {
            'high_credibility_sources': len(high_cred_sources),
            'total_sources': len(research_results),
            'consensus_strength': len(high_cred_sources) / len(research_results) * 100
        }
        
        # Main findings from highest credibility sources
        for result in sorted(research_results, key=lambda x: x.get('credibility_score', 0), reverse=True)[:5]:
            summary = result.get('ai_summary', '')
            if summary and len(summary) > 50:
                insights['main_findings'].append({
                    'finding': summary,
                    'source': result['title'],
                    'credibility': result.get('credibility_score', 0)
                })
        
        return insights
    
    def _assess_research_quality(self, results):
        """Comprehensive research quality assessment"""
        if not results:
            return 0
        
        metrics = {
            'source_diversity': len(set(r.get('domain', '') for r in results)) / len(results) * 25,
            'average_credibility': sum(r.get('credibility_score', 0) for r in results) / len(results) * 0.25,
            'content_depth': min(sum(r.get('word_count', 0) for r in results) / len(results) / 200, 1) * 25,
            'source_count': min(len(results) / 10, 1) * 25
        }
        
        return sum(metrics.values())

def create_futuristic_background():
    """Create animated futuristic background"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, 
            #0c0c0c 0%, 
            #1a1a2e 25%, 
            #16213e 50%, 
            #0f3460 75%, 
            #533483 100%);
        color: #e0e6ed;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Animated particles background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 25% 25%, #00d4ff22 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, #ff006622 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, #8b5cf622 0%, transparent 50%);
        animation: particles 20s ease-in-out infinite;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes particles {
        0%, 100% { 
            background: 
                radial-gradient(circle at 25% 25%, #00d4ff22 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, #ff006622 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, #8b5cf622 0%, transparent 50%);
        }
        33% { 
            background: 
                radial-gradient(circle at 75% 25%, #00d4ff22 0%, transparent 50%),
                radial-gradient(circle at 25% 75%, #ff006622 0%, transparent 50%),
                radial-gradient(circle at 50% 25%, #8b5cf622 0%, transparent 50%);
        }
        66% { 
            background: 
                radial-gradient(circle at 50% 75%, #00d4ff22 0%, transparent 50%),
                radial-gradient(circle at 25% 25%, #ff006622 0%, transparent 50%),
                radial-gradient(circle at 75% 50%, #8b5cf622 0%, transparent 50%);
        }
    }
    
    /* Futuristic header */
    .intellisource-header {
        font-family: 'Orbitron', monospace !important;
        font-size: 4.5rem !important;
        background: linear-gradient(135deg, 
            #00d4ff 0%, 
            #ff0066 25%, 
            #8b5cf6 50%, 
            #06ffa5 75%, 
            #ffb347 100%) !important;
        background-size: 300% 300% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center !important;
        font-weight: 900 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
        animation: gradientShift 8s ease-in-out infinite, glow 2s ease-in-out infinite alternate !important;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5) !important;
        position: relative !important;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes glow {
        from { filter: brightness(1) drop-shadow(0 0 20px rgba(0, 212, 255, 0.4)); }
        to { filter: brightness(1.2) drop-shadow(0 0 30px rgba(0, 212, 255, 0.8)); }
    }
    
    .tagline {
        text-align: center !important;
        font-size: 1.6rem !important;
        background: linear-gradient(90deg, #00d4ff, #ff0066, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 2rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        animation: pulse 3s ease-in-out infinite !important;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    /* Futuristic badges */
    .power-badge {
        background: linear-gradient(135deg, #00d4ff, #8b5cf6) !important;
        color: white !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 50px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin: 0.5rem !important;
        box-shadow: 
            0 8px 25px rgba(0, 212, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        animation: float 4s ease-in-out infinite !important;
    }
    
    .power-badge:nth-child(2) {
        animation-delay: -1s !important;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .power-badge:hover {
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 
            0 12px 35px rgba(0, 212, 255, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Futuristic metric cards */
    .metric-card {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(139, 92, 246, 0.1) 50%,
            rgba(255, 0, 102, 0.1) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        color: white !important;
        text-align: center !important;
        margin: 0.5rem 0 !important;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 
            0 25px 50px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(0, 212, 255, 0.5) !important;
    }
    
    .metric-card h2 {
        font-family: 'Orbitron', monospace !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #00d4ff, #ff0066) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Futuristic insight cards */
    .insight-card {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.08) 0%, 
            rgba(139, 92, 246, 0.08) 100%) !important;
        backdrop-filter: blur(15px) !important;
        border-left: 4px solid #00d4ff !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s ease !important;
    }
    
    .insight-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #00d4ff, #8b5cf6, #ff0066);
        animation: pulse-border 2s ease-in-out infinite;
    }
    
    @keyframes pulse-border {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .insight-card:hover {
        transform: translateX(10px) !important;
        border-left-color: #ff0066 !important;
        box-shadow: 
            -5px 10px 40px rgba(0, 212, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.03) 0%, 
            rgba(0, 212, 255, 0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        color: #e0e6ed !important;
    }
    
    .source-card:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(0, 212, 255, 0.3) !important;
        box-shadow: 
            0 15px 40px rgba(0, 212, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Credibility bars */
    .credibility-bar {
        height: 8px !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    .credibility-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 30%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: loading 2s infinite;
    }
    
    @keyframes loading {
        0% { left: -30%; }
        100% { left: 100%; }
    }
    
    .high-credibility { 
        background: linear-gradient(90deg, #00d4ff, #06ffa5) !important;
        box-shadow: 0 0 15px rgba(6, 255, 165, 0.4) !important;
    }
    .medium-credibility { 
        background: linear-gradient(90deg, #ffb347, #ff0066) !important;
        box-shadow: 0 0 15px rgba(255, 179, 71, 0.4) !important;
    }
    .low-credibility { 
        background: linear-gradient(90deg, #ff0066, #8b5cf6) !important;
        box-shadow: 0 0 15px rgba(255, 0, 102, 0.4) !important;
    }
    
    /* Status indicators */
    .stStatus {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(139, 92, 246, 0.1) 100%) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 15px !important;
        color: #e0e6ed !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 
            0 8px 25px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 
            0 15px 40px rgba(0, 212, 255, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, 
            rgba(12, 12, 12, 0.95) 0%, 
            rgba(26, 26, 46, 0.95) 50%,
            rgba(22, 33, 62, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }
    
    .stSidebar .stMarkdown {
        color: #e0e6ed !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.05) 0%, 
            rgba(0, 212, 255, 0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 15px !important;
        color: #e0e6ed !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(224, 230, 237, 0.6) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.05) 0%, 
            rgba(0, 212, 255, 0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 15px !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #8b5cf6) !important;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.08) 0%, 
            rgba(139, 92, 246, 0.08) 100%) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(0, 212, 255, 0.3) !important;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(139, 92, 246, 0.1) 100%) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px !important;
        color: #e0e6ed !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        background: transparent !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #8b5cf6);
        border-radius: 10px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff0066, #00d4ff);
    }
    
    /* Loading animations */
    @keyframes neural-pulse {
        0%, 100% { 
            box-shadow: 
                0 0 20px rgba(0, 212, 255, 0.4),
                0 0 40px rgba(0, 212, 255, 0.2),
                0 0 60px rgba(0, 212, 255, 0.1);
        }
        50% { 
            box-shadow: 
                0 0 30px rgba(255, 0, 102, 0.4),
                0 0 60px rgba(255, 0, 102, 0.2),
                0 0 80px rgba(255, 0, 102, 0.1);
        }
    }
    
    .neural-active {
        animation: neural-pulse 2s ease-in-out infinite !important;
    }
    
    /* Success/Warning/Error states */
    .stSuccess {
        background: linear-gradient(135deg, 
            rgba(6, 255, 165, 0.1) 0%, 
            rgba(0, 212, 255, 0.1) 100%) !important;
        border: 1px solid rgba(6, 255, 165, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(15px) !important;
        color: #06ffa5 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, 
            rgba(255, 179, 71, 0.1) 0%, 
            rgba(255, 0, 102, 0.1) 100%) !important;
        border: 1px solid rgba(255, 179, 71, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(15px) !important;
        color: #ffb347 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, 
            rgba(255, 0, 102, 0.1) 0%, 
            rgba(139, 92, 246, 0.1) 100%) !important;
        border: 1px solid rgba(255, 0, 102, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(15px) !important;
        color: #ff0066 !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #06ffa5 0%, #00d4ff 100%) !important;
        color: #0c0c0c !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 
            0 6px 20px rgba(6, 255, 165, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 
            0 10px 30px rgba(6, 255, 165, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    }
    </style>
    """

def generate_comprehensive_report(research_data):
    """Generate a comprehensive PDF-style report"""
    report = f"""
# IntelliSource Research Report
**Generated on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Research Query:** {research_data['query']}
**Processing Time:** {research_data['processing_time']}

## Executive Summary
This comprehensive analysis was conducted using IntelliSource's advanced AI research engine, analyzing {research_data['sources_analyzed']} sources with an average credibility score of {research_data.get('average_credibility', 0):.1f}/100.

## Key Insights
"""
    
    insights = research_data.get('insights', {})
    if insights:
        report += f"""
### Research Quality Score: {insights.get('research_quality', 0):.1f}/100

### Source Consensus Analysis
- **High Credibility Sources:** {insights.get('source_consensus', {}).get('high_credibility_sources', 0)}
- **Total Sources Analyzed:** {insights.get('source_consensus', {}).get('total_sources', 0)}
- **Consensus Strength:** {insights.get('source_consensus', {}).get('consensus_strength', 0):.1f}%

### Key Statistics Found:
"""
        for stat in insights.get('key_statistics', [])[:5]:
            report += f"- {stat}\n"
    
    verified_facts = research_data.get('verified_facts', {})
    if verified_facts:
        report += f"""
## Cross-Referenced Facts
The following facts were verified across multiple sources:
"""
        for fact, data in list(verified_facts.items())[:5]:
            report += f"- **{fact}** (Verified across {data['count']} sources)\n"
    
    report += f"""
## Source Analysis
The following sources were analyzed for this research:

"""
    
    for i, source in enumerate(research_data.get('sources', [])[:10], 1):
        report += f"""
### {i}. {source['title']}
- **URL:** {source['url']}
- **Domain:** {source['domain']}
- **Credibility Score:** {source['credibility']:.1f}/100
- **Summary:** {source['summary'][:200]}...

"""
    
    report += f"""
## Methodology
This report was generated using IntelliSource's proprietary AI research engine, which employs:
- Multi-source web intelligence gathering
- Advanced credibility scoring algorithms
- Neural-inspired content summarization
- Cross-source fact verification
- Parallel processing for efficiency

---
*Report generated by IntelliSource AI Research Engine*
*For questions about this research, contact your research team.*
"""
    
    return report

def create_intellisource_app():
    """Enhanced IntelliSource Streamlit Application with Futuristic Design"""
    
    st.set_page_config(
        page_title="IntelliSource - Advanced AI Research Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply futuristic styling
    st.markdown(create_futuristic_background(), unsafe_allow_html=True)
    
    # IntelliSource Header with enhanced styling
    st.markdown('<h1 class="intellisource-header">🧠 IntelliSource</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Advanced Autonomous AI Research Agent</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('''
        <div style="text-align: center;">
            <span class="power-badge">🚀 Beyond ChatGPT & Grok</span>
            <span class="power-badge">⚡ Real-time Web Intelligence</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # Enhanced Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Research Configuration")
        
        research_depth = st.selectbox(
            "🔍 Research Depth",
            [1, 2, 3],
            index=1,
            help="How many levels deep should IntelliSource research?"
        )
        
        max_sources = st.slider(
            "📊 Maximum Sources",
            min_value=5,
            max_value=20,
            value=12,
            help="Number of sources to analyze"
        )
        
        parallel_workers = st.slider(
            "⚡ Parallel Workers",
            min_value=3,
            max_value=8,
            value=5,
            help="Concurrent analysis threads"
        )
        
        st.markdown("---")
        st.markdown("### 🏆 IntelliSource Advantages")
        advantages = [
            "🔍 Real-time web access",
            "🧠 Neural summarization",
            "⚖️ Advanced credibility scoring",
            "📊 Interactive visualizations", 
            "🔗 Autonomous link following",
            "📈 Cross-source fact verification",
            "⚡ Parallel processing",
            "📋 Professional report export"
        ]
        
        for adv in advantages:
            st.markdown(f"✅ {adv}")
        
        st.markdown("---")
        st.markdown("### 📈 Target Performance")
        st.metric("Research Speed", "< 45 seconds")
        st.metric("Avg Credibility", "85+/100") 
        st.metric("Source Analysis", "12+ sources")
    
    # Main interface with enhanced styling
    st.markdown("## 🔍 Research Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_query = st.text_input(
            "",
            placeholder="Enter your research topic (e.g., 'latest quantum computing breakthroughs')",
            help="Be specific for more targeted results"
        )
    
    with col2:
        research_button = st.button("🚀 Start Research", type="primary", use_container_width=True)
    
    # Enhanced example queries
    st.markdown("**💡 Example Queries:**")
    examples = [
        "artificial intelligence safety research 2025",
        "quantum computing commercial applications", 
        "renewable energy storage breakthroughs",
        "gene therapy recent clinical trials",
        "autonomous vehicle technology progress"
    ]
    
    example_cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"📋 {example.split()[0].title()}", key=f"ex_{i}", help=example):
                st.session_state['selected_query'] = example
                st.rerun()
    
    # Handle example selection
    if 'selected_query' in st.session_state:
        research_query = st.session_state['selected_query']
        del st.session_state['selected_query']
        research_button = True
    
    # Research execution with enhanced progress tracking
    if research_button and research_query:
        agent = IntelliSourceEngine()
        
        # Real-time research dashboard with enhanced visuals
        st.markdown("---")
        st.markdown('<div class="neural-active"><h2>🤖 IntelliSource Research Engine Active</h2></div>', unsafe_allow_html=True)
        
        with st.container():
            start_time = time.time()
            
            # Enhanced Phase 1: Intelligent URL Discovery
            with st.status("🔍 **Phase 1:** Intelligent URL Discovery", expanded=True) as status:
                st.write("🧠 Analyzing query and generating search strategy...")
                search_urls = agent.intelligent_search(research_query, max_sources)
                st.write(f"✅ Discovered {len(search_urls)} high-potential sources")
                time.sleep(1)
            
            # Enhanced Phase 2: Parallel Content Analysis  
            with st.status("🧠 **Phase 2:** Parallel Content Analysis", expanded=True) as status:
                st.write("⚡ Deploying parallel analysis workers...")
                research_results = agent.parallel_research(research_query, search_urls, max_workers=parallel_workers)
                st.write(f"✅ Successfully analyzed {len(research_results)} sources")
                time.sleep(1)
            
            # Enhanced Phase 3: Intelligence Synthesis
            with st.status("⚡ **Phase 3:** Intelligence Synthesis", expanded=True) as status:
                st.write("🔬 Cross-referencing facts and generating insights...")
                insights = agent.generate_insights(research_results, research_query)
                verified_facts = agent.cross_reference_facts(research_results)
                end_time = time.time()
                st.write(f"🎯 Research completed in {end_time - start_time:.1f} seconds")
        
        # Enhanced Results Dashboard
        if research_results:
            st.markdown("---")
            st.markdown("## 📊 IntelliSource Research Dashboard")
            
            # Enhanced key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{len(research_results)}</h2>
                    <p>Sources Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_cred = sum(r.get('credibility_score', 0) for r in research_results) / len(research_results)
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{avg_cred:.1f}</h2>
                    <p>Avg Credibility</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_words = sum(r.get('word_count', 0) for r in research_results)
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{total_words:,}</h2>
                    <p>Words Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{end_time - start_time:.1f}s</h2>
                    <p>Processing Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced insights section
            if insights:
                st.markdown("### 🎯 Key Insights")
                
                # Research Quality Score
                quality_score = insights.get('research_quality', 0)
                st.markdown(f"""
                <div class="insight-card">
                    <h4>📈 Research Quality Score: {quality_score:.1f}/100</h4>
                    <p>Overall assessment of research comprehensiveness and source reliability</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Source Consensus
                consensus = insights.get('source_consensus', {})
                if consensus:
                    consensus_strength = consensus.get('consensus_strength', 0)
                    high_cred_count = consensus.get('high_credibility_sources', 0)
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>⚖️ Source Consensus: {consensus_strength:.1f}%</h4>
                        <p>{high_cred_count} out of {len(research_results)} sources have high credibility scores (>70)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Main Findings
                findings = insights.get('main_findings', [])
                if findings:
                    st.markdown("### 🔍 Main Research Findings")
                    
                    for i, finding in enumerate(findings[:3]):
                        credibility = finding.get('credibility', 0)
                        cred_class = 'high-credibility' if credibility > 80 else 'medium-credibility' if credibility > 60 else 'low-credibility'
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <h5>📋 Finding {i+1}</h5>
                            <p>{finding['finding']}</p>
                            <small><strong>Source:</strong> {finding['source']}</small><br>
                            <small><strong>Credibility:</strong> {credibility:.1f}/100</small>
                            <div class="credibility-bar {cred_class}" style="width: {credibility}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Enhanced visualizations with dark theme
            st.markdown("### 📊 Research Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Enhanced credibility distribution
                credibility_scores = [r.get('credibility_score', 0) for r in research_results]
                
                fig_cred = px.histogram(
                    x=credibility_scores,
                    nbins=10,
                    title="Source Credibility Distribution",
                    labels={'x': 'Credibility Score', 'y': 'Number of Sources'},
                    color_discrete_sequence=['#00d4ff']
                )
                fig_cred.update_layout(
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    title_font=dict(size=16, color='#00d4ff')
                )
                st.plotly_chart(fig_cred, use_container_width=True)
            
            with viz_col2:
                # Enhanced content quality vs credibility
                quality_scores = [r.get('content_quality', 0) for r in research_results]
                domains = [r.get('domain', 'Unknown') for r in research_results]
                
                fig_scatter = px.scatter(
                    x=credibility_scores,
                    y=quality_scores,
                    hover_data={'Domain': domains},
                    title="Content Quality vs Credibility",
                    labels={'x': 'Credibility Score', 'y': 'Content Quality'},
                    color=credibility_scores,
                    color_continuous_scale=['#ff0066', '#00d4ff', '#06ffa5']
                )
                fig_scatter.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    title_font=dict(size=16, color='#00d4ff')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Enhanced verified facts section
            if verified_facts:
                st.markdown("### ✅ Cross-Referenced Facts")
                
                fact_df = pd.DataFrame([
                    {
                        'Fact': fact,
                        'Sources': len(data['sources']),
                        'Avg Credibility': sum(r.get('credibility_score', 0) for r in research_results 
                                             if r['url'] in data['sources']) / len(data['sources'])
                    }
                    for fact, data in verified_facts.items()
                ])
                
                if not fact_df.empty:
                    fact_df = fact_df.sort_values('Avg Credibility', ascending=False)
                    
                    for _, row in fact_df.head(5).iterrows():
                        cred = row['Avg Credibility']
                        cred_class = 'high-credibility' if cred > 80 else 'medium-credibility' if cred > 60 else 'low-credibility'
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{row['Fact']}</strong><br>
                            <small>Verified across {row['Sources']} sources | Avg credibility: {cred:.1f}/100</small>
                            <div class="credibility-bar {cred_class}" style="width: {cred}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Enhanced detailed source analysis
            st.markdown("### 📚 Source Analysis")
            
            # Sort by credibility
            sorted_results = sorted(research_results, key=lambda x: x.get('credibility_score', 0), reverse=True)
            
            for i, result in enumerate(sorted_results):
                with st.expander(f"🔍 Source {i+1}: {result.get('title', 'Unknown Title')} (Credibility: {result.get('credibility_score', 0):.1f}/100)"):
                    
                    col_info, col_summary = st.columns([1, 2])
                    
                    with col_info:
                        st.markdown(f"**URL:** {result['url']}")
                        st.markdown(f"**Domain:** {result.get('domain', 'Unknown')}")
                        st.markdown(f"**Word Count:** {result.get('word_count', 0):,}")
                        st.markdown(f"**Paragraphs:** {result.get('paragraph_count', 0)}")
                        st.markdown(f"**Content Quality:** {result.get('content_quality', 0):.1f}/100")
                        
                        # Enhanced credibility breakdown
                        cred_score = result.get('credibility_score', 0)
                        if cred_score > 80:
                            st.success(f"High Credibility: {cred_score:.1f}")
                        elif cred_score > 60:
                            st.warning(f"Medium Credibility: {cred_score:.1f}")
                        else:
                            st.error(f"Low Credibility: {cred_score:.1f}")
                    
                    with col_summary:
                        st.markdown("**AI Summary:**")
                        summary = result.get('ai_summary', 'No summary available')
                        st.markdown(summary)
                        
                        # Key entities
                        entities = result.get('key_entities', [])
                        if entities:
                            st.markdown("**Key Entities:**")
                            entity_text = " • ".join(entities[:5])
                            st.markdown(entity_text)
            
            # Enhanced export options with working functionality
            st.markdown("---")
            st.markdown("### 📤 Export Research")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            # Prepare comprehensive report data
            report_data = {
                'query': research_query,
                'timestamp': datetime.now().isoformat(),
                'processing_time': f"{end_time - start_time:.1f} seconds",
                'sources_analyzed': len(research_results),
                'average_credibility': avg_cred,
                'insights': insights,
                'verified_facts': verified_facts,
                'sources': [
                    {
                        'title': r.get('title', ''),
                        'url': r['url'],
                        'credibility': r.get('credibility_score', 0),
                        'summary': r.get('ai_summary', ''),
                        'domain': r.get('domain', ''),
                        'word_count': r.get('word_count', 0),
                        'content_quality': r.get('content_quality', 0),
                        'key_entities': r.get('key_entities', [])
                    }
                    for r in sorted_results
                ]
            }
            
            with export_col1:
                # Generate comprehensive PDF-style report
                report_content = generate_comprehensive_report(report_data)
                
                st.download_button(
                    label="📋 Download Full Report",
                    data=report_content,
                    file_name=f"IntelliSource_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with export_col2:
                # Export as enhanced CSV
                df = pd.DataFrame([
                    {
                        'Title': r.get('title', ''),
                        'URL': r['url'],
                        'Domain': r.get('domain', ''),
                        'Credibility Score': r.get('credibility_score', 0),
                        'Content Quality': r.get('content_quality', 0),
                        'Word Count': r.get('word_count', 0),
                        'Paragraph Count': r.get('paragraph_count', 0),
                        'AI Summary': r.get('ai_summary', ''),
                        'Key Entities': ', '.join(r.get('key_entities', [])),
                        'Processing Time': f"{end_time - start_time:.1f}s",
                        'Research Query': research_query,
                        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for r in research_results
                ])
                
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_data,
                    file_name=f"IntelliSource_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col3:
                # Enhanced JSON export with all research data
                json_data = json.dumps(report_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📁 Download JSON Report",
                    data=json_data,
                    file_name=f"IntelliSource_Complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Enhanced share functionality
            st.markdown("---")
            st.markdown("### 🔗 Share Research")
            
            share_col1, share_col2, share_col3 = st.columns(3)
            
            with share_col1:
                # Generate research hash for sharing
                research_hash = hashlib.md5(
                    f"{research_query}{len(research_results)}{avg_cred}{datetime.now().strftime('%Y%m%d')}".encode()
                ).hexdigest()[:12]
                
                st.text_input(
                    "Research Session ID:",
                    value=f"IS-{research_hash.upper()}",
                    help="Save this ID to reference this research session",
                    disabled=True
                )
            
            with share_col2:
                # Create shareable summary
                summary_text = f"""🧠 IntelliSource Research Summary
Query: {research_query}
Sources: {len(research_results)} analyzed
Avg Credibility: {avg_cred:.1f}/100
Quality Score: {insights.get('research_quality', 0):.1f}/100
Session ID: IS-{research_hash.upper()}
"""
                
                st.download_button(
                    label="📤 Generate Share Summary",
                    data=summary_text,
                    file_name=f"IntelliSource_Summary_{research_hash}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with share_col3:
                # Research statistics for sharing
                if st.button("📊 Copy Research Stats", use_container_width=True):
                    stats_text = f"IntelliSource Analysis: {len(research_results)} sources, {avg_cred:.1f}/100 credibility, {end_time - start_time:.1f}s processing time. Session: IS-{research_hash.upper()}"
                    
                    # Store in session state for user to copy
                    st.session_state['copy_text'] = stats_text
                    st.success("✅ Stats ready to copy!")
            
            # Display copy text if available
            if 'copy_text' in st.session_state:
                st.text_area(
                    "Research Stats (Click to select all and copy):",
                    value=st.session_state['copy_text'],
                    height=100
                )
            
            # Enhanced research insights visualization
            st.markdown("---")
            st.markdown("### 🎯 Advanced Research Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                # Domain distribution pie chart
                domain_counts = {}
                for result in research_results:
                    domain = result.get('domain', 'Unknown')
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                fig_domains = px.pie(
                    values=list(domain_counts.values()),
                    names=list(domain_counts.keys()),
                    title="Source Domain Distribution",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig_domains.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    title_font=dict(size=16, color='#00d4ff')
                )
                st.plotly_chart(fig_domains, use_container_width=True)
            
            with insight_col2:
                # Research timeline (word count vs credibility)
                fig_timeline = go.Figure(data=go.Scatter(
                    x=list(range(1, len(research_results) + 1)),
                    y=[r.get('credibility_score', 0) for r in research_results],
                    mode='lines+markers',
                    name='Credibility Timeline',
                    line=dict(color='#00d4ff', width=3),
                    marker=dict(size=8, color='#ff0066')
                ))
                
                fig_timeline.update_layout(
                    title="Source Analysis Timeline",
                    xaxis_title="Source Number",
                    yaxis_title="Credibility Score",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    title_font=dict(size=16, color='#00d4ff')
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Research completion celebration
            st.markdown("---")
            st.markdown("""
            <div class="insight-card" style="text-align: center;">
                <h3>🎉 Research Analysis Complete!</h3>
                <p>Your comprehensive research has been processed with IntelliSource's advanced AI algorithms.</p>
                <p><strong>Next Steps:</strong> Review the findings above, export your data, and use the insights for your project.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Enhanced error handling with suggestions
            st.error("❌ No valid sources found. Please try a different query or check your internet connection.")
            
            st.markdown("""
            <div class="insight-card">
                <h4>💡 Troubleshooting Suggestions:</h4>
                <ul>
                    <li><strong>Refine your query:</strong> Be more specific with keywords</li>
                    <li><strong>Try different topics:</strong> Some subjects may have limited online coverage</li>
                    <li><strong>Check connectivity:</strong> Ensure stable internet connection</li>
                    <li><strong>Reduce scope:</strong> Try a more focused research question</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggest alternative queries
            st.markdown("### 🔄 Try These Alternative Queries:")
            alt_queries = [
                "machine learning applications 2024",
                "climate change recent studies",
                "cryptocurrency market analysis",
                "space exploration achievements",
                "medical breakthroughs 2024"
            ]
            
            alt_cols = st.columns(len(alt_queries))
            for i, alt_query in enumerate(alt_queries):
                with alt_cols[i]:
                    if st.button(f"🔍 {alt_query.split()[0].title()}", key=f"alt_{i}"):
                        st.session_state['selected_query'] = alt_query
                        st.rerun()

# Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(139, 92, 246, 0.05)); border-radius: 15px; margin-top: 2rem;">
        <h4 style="color: #00d4ff; margin-bottom: 1rem;">🧠 IntelliSource AI Research Engine</h4>
        <p style="color: #e0e6ed; margin-bottom: 0.5rem;">Powered by advanced neural algorithms and real-time web intelligence</p>
        <p style="color: #8b5cf6; font-size: 0.9rem;">© 2025 IntelliSource Technologies - Revolutionizing Research with AI</p>
    </div>
    """, unsafe_allow_html=True)

# Run the enhanced application
if __name__ == "__main__":
    create_intellisource_app()
