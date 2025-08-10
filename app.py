# Advanced Autonomous Research Agent
# This will be BETTER than ChatGPT/Grok for research

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from datetime import datetime
import re
import json
from urllib.parse import urljoin, urlparse
import time
from collections import Counter
import networkx as nx

# Simple text processing without heavy ML dependencies
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
    STOPWORDS = set(stopwords.words('english'))
except:
    # Fallback if NLTK fails
    STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    sent_tokenize = lambda x: re.split(r'[.!?]+', x)

class AdvancedResearchAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize AI models (simplified for Streamlit Cloud)
        self.research_graph = nx.DiGraph()
        self.fact_database = {}
        self.source_credibility = {}
    
    def smart_summarize(self, content, focus_topic, max_sentences=3):
        """Advanced summarization without heavy ML dependencies"""
        sentences = sent_tokenize(content)
        if not sentences:
            return "No content to summarize."
        
        # Clean and score sentences
        topic_words = set(focus_topic.lower().split()) - STOPWORDS
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
                
            words = set(re.findall(r'\w+', sentence.lower())) - STOPWORDS
            
            # Multiple scoring factors
            topic_score = len(words.intersection(topic_words))
            length_score = min(len(words) / 20, 1)  # Prefer medium-length sentences
            position_score = 1 / (sentences.index(sentence) + 1)  # Earlier sentences score higher
            
            # Look for key indicators
            key_indicators = ['research', 'study', 'analysis', 'report', 'found', 'shows', 'indicates']
            indicator_score = sum(1 for indicator in key_indicators if indicator in sentence.lower())
            
            total_score = topic_score * 3 + length_score + position_score * 0.5 + indicator_score
            
            if total_score > 0:
                scored_sentences.append((total_score, sentence.strip()))
        
        # Sort and select best sentences
        scored_sentences.sort(reverse=True)
        selected_sentences = [sent[1] for sent in scored_sentences[:max_sentences]]
        
        return '. '.join(selected_sentences) if selected_sentences else "No relevant content found."
    
    def search_web(self, query, num_results=10):
        """Smart web search using multiple search engines"""
        search_urls = []
        
        # DuckDuckGo search (no API key needed)
        try:
            search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True)[:num_results]:
                href = link['href']
                if href.startswith('//duckduckgo.com/l/?uddg='):
                    # Extract actual URL from DuckDuckGo redirect
                    actual_url = requests.utils.unquote(href.split('uddg=')[1].split('&')[0])
                    if actual_url.startswith('http'):
                        search_urls.append(actual_url)
        except Exception as e:
            st.warning(f"Search error: {e}")
        
        # Fallback: Use predefined quality sources
        if not search_urls:
            search_urls = [
                f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
                f"https://www.nature.com/search?q={query.replace(' ', '+')}"
            ]
        
        return search_urls[:num_results]
    
    def extract_advanced_content(self, url):
        """Advanced content extraction with metadata"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract metadata
            title = soup.find('title')
            title = title.text.strip() if title else "Unknown Title"
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            # Extract main content intelligently
            content_selectors = ['main', 'article', '.content', '#content', '.post-content']
            main_content = None
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            # Extract text and links
            text = main_content.get_text(separator=' ', strip=True) if main_content else ''
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            
            # Content quality metrics
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            return {
                'url': url,
                'title': title,
                'description': description,
                'content': text[:5000],  # Limit for processing
                'word_count': word_count,
                'sentence_count': sentence_count,
                'links': links[:20],  # Top 20 links
                'credibility_score': self.assess_credibility(url, title, text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def assess_credibility(self, url, title, content):
        """Assess source credibility"""
        score = 50  # Base score
        
        # Domain-based scoring
        domain = urlparse(url).netloc.lower()
        
        trusted_domains = ['wikipedia.org', 'nature.com', 'science.org', 'ieee.org', 'acm.org']
        news_domains = ['reuters.com', 'bbc.com', 'npr.org', 'ap.org']
        academic_domains = ['.edu', '.gov']
        
        if any(trusted in domain for trusted in trusted_domains):
            score += 30
        elif any(news in domain for news in news_domains):
            score += 20
        elif any(academic in domain for academic in academic_domains):
            score += 25
        
        # Content-based scoring
        if len(content.split()) > 500:
            score += 10
        if re.search(r'\d{4}', title):  # Has year in title
            score += 5
        if 'research' in content.lower() or 'study' in content.lower():
            score += 10
        
        return min(score, 100)
    
    def ai_summarize(self, content, focus_topic):
        """Advanced AI summarization"""
        if not self.summarizer:
            # Fallback: Extractive summarization
            sentences = re.split(r'[.!?]+', content)
            topic_words = focus_topic.lower().split()
            
            scored_sentences = []
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    score = sum(1 for word in topic_words if word in sentence.lower())
                    scored_sentences.append((score, sentence.strip()))
            
            scored_sentences.sort(reverse=True)
            return '. '.join([sent[1] for sent in scored_sentences[:3]])
        
        try:
            # Use transformer model for better summarization
            summary = self.summarizer(content[:1000], max_length=150, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except:
            return content[:300] + "..."
    
    def find_related_links(self, content, original_links):
        """Find most relevant links for deeper research"""
        if not original_links:
            return []
        
        # Score links based on context
        scored_links = []
        content_words = set(content.lower().split())
        
        for link in original_links:
            link_text = urlparse(link).path.lower()
            link_score = sum(1 for word in content_words if word in link_text)
            
            # Prefer certain types of links
            if any(domain in link for domain in ['research', 'study', 'report', 'analysis']):
                link_score += 5
            
            if link_score > 0:
                scored_links.append((link_score, link))
        
        scored_links.sort(reverse=True)
        return [link[1] for link in scored_links[:5]]
    
    def autonomous_research(self, topic, max_depth=2, max_sources=15):
        """Fully autonomous research with depth"""
        research_results = {
            'topic': topic,
            'sources': [],
            'insights': [],
            'fact_network': {},
            'credibility_analysis': {},
            'research_path': []
        }
        
        # Level 1: Initial search and analysis
        st.write(f"🚀 **Starting autonomous research on: {topic}**")
        
        initial_urls = self.search_web(topic, 5)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analyzed_urls = set()
        to_analyze = initial_urls.copy()
        
        for depth in range(max_depth):
            st.write(f"🔍 **Research Depth Level {depth + 1}**")
            
            current_batch = to_analyze[:max_sources]
            to_analyze = []
            
            for i, url in enumerate(current_batch):
                if url in analyzed_urls or len(research_results['sources']) >= max_sources:
                    continue
                
                status_text.text(f"Analyzing: {url[:50]}...")
                progress = (i + 1) / len(current_batch)
                progress_bar.progress(progress)
                
                # Extract content
                extracted = self.extract_advanced_content(url)
                
                if 'error' not in extracted:
                    # AI-powered analysis
                    summary = self.ai_summarize(extracted['content'], topic)
                    
                    # Find insights and facts
                    insights = self.extract_insights(extracted['content'], topic)
                    
                    # Store results
                    source_data = {
                        **extracted,
                        'summary': summary,
                        'insights': insights,
                        'depth_level': depth + 1
                    }
                    
                    research_results['sources'].append(source_data)
                    research_results['credibility_analysis'][url] = extracted['credibility_score']
                    
                    # Find related links for next depth level
                    if depth < max_depth - 1:
                        related_links = self.find_related_links(extracted['content'], extracted.get('links', []))
                        to_analyze.extend(related_links)
                
                analyzed_urls.add(url)
                time.sleep(0.5)  # Rate limiting
        
        # Generate final insights
        research_results['final_analysis'] = self.generate_insights(research_results)
        research_results['research_quality_score'] = self.calculate_research_quality(research_results)
        
        return research_results
    
    def extract_insights(self, content, topic):
        """Extract key insights using NLP"""
        insights = []
        
        # Find statistics and numbers
        stats = re.findall(r'\d+(?:\.\d+)?%?(?:\s*(?:billion|million|thousand|percent|%))', content)
        if stats:
            insights.extend([f"Key statistic: {stat}" for stat in stats[:3]])
        
        # Find trends and predictions
        trend_patterns = [
            r'(?:by|in|until)\s+20\d{2}',
            r'(?:increasing|decreasing|growing|declining)\s+(?:by|at)\s+\d+',
            r'(?:expected|projected|anticipated)\s+to\s+\w+'
        ]
        
        for pattern in trend_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            insights.extend([f"Trend: {match}" for match in matches[:2]])
        
        return insights[:5]
    
    def generate_insights(self, research_data):
        """Generate high-level insights from all research"""
        all_insights = []
        high_credibility_sources = []
        
        for source in research_data['sources']:
            all_insights.extend(source.get('insights', []))
            if source.get('credibility_score', 0) > 70:
                high_credibility_sources.append(source)
        
        return {
            'total_sources_analyzed': len(research_data['sources']),
            'high_credibility_sources': len(high_credibility_sources),
            'key_insights': all_insights[:10],
            'average_credibility': sum(research_data['credibility_analysis'].values()) / len(research_data['credibility_analysis']) if research_data['credibility_analysis'] else 0
        }
    
    def calculate_research_quality(self, research_data):
        """Calculate overall research quality score"""
        if not research_data['sources']:
            return 0
        
        credibility_score = sum(research_data['credibility_analysis'].values()) / len(research_data['credibility_analysis'])
        source_diversity = len(set(urlparse(s['url']).netloc for s in research_data['sources']))
        content_depth = sum(s['word_count'] for s in research_data['sources']) / len(research_data['sources'])
        
        quality_score = (credibility_score * 0.4 + 
                        min(source_diversity * 10, 50) * 0.3 + 
                        min(content_depth / 100, 50) * 0.3)
        
        return min(quality_score, 100)

def create_streamlit_app():
    """Advanced Streamlit interface"""
    st.set_page_config(
        page_title="IntelliSource - AI Research Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional IntelliSource branding
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tagline {
        text-align: center; 
        font-size: 1.3rem; 
        color: #555;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        border-left: 4px solid #667eea;
        padding: 1rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .intellisource-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with IntelliSource branding
    st.markdown('<h1 class="main-header">🧠 IntelliSource</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Advanced Autonomous AI Research Agent • Beyond ChatGPT & Grok</p>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><span class="intellisource-badge">🚀 Built by Imaad Mahmood</span></div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Research Configuration")
        
        research_depth = st.selectbox(
            "Research Depth",
            [1, 2, 3],
            index=1,
            help="How many levels deep should the agent research?"
        )
        
        max_sources = st.slider(
            "Maximum Sources",
            min_value=5,
            max_value=25,
            value=15,
            help="Maximum number of sources to analyze"
        )
        
        analysis_focus = st.multiselect(
            "Analysis Focus",
            ["Statistics", "Trends", "Expert Opinions", "Recent Developments", "Controversies"],
            default=["Statistics", "Trends"]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Research Quality Targets")
        st.markdown("- **Speed:** < 60 seconds")
        st.markdown("- **Sources:** 15+ analyzed")
        st.markdown("- **Credibility:** 80+ average")
        st.markdown("- **Depth:** Multi-level analysis")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_query = st.text_input(
            "🔍 Enter your research topic:",
            placeholder="e.g., 'latest developments in quantum computing'",
            help="Be specific for better results"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            research_button = st.button("🚀 Start Research", type="primary", use_container_width=True)
        with col_b:
            if st.button("💡 Suggest Topics", use_container_width=True):
                suggestions = [
                    "artificial intelligence safety research 2025",
                    "quantum computing breakthrough recent",
                    "renewable energy technology advances",
                    "biotech innovation gene therapy",
                    "space exploration private companies"
                ]
                st.write("**Trending Research Topics:**")
                for suggestion in suggestions:
                    if st.button(f"📋 {suggestion}", key=suggestion):
                        st.session_state.suggested_query = suggestion
        with col_c:
            export_button = st.button("📊 Export Results", use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Why IntelliSource Beats ChatGPT/Grok")
        st.markdown("""
        ✅ **Real-time web access**  
        ✅ **Multi-source fact checking**  
        ✅ **Credibility scoring**  
        ✅ **Interactive visualizations**  
        ✅ **Professional report export**  
        ✅ **Autonomous link following**  
        ✅ **Source diversity analysis**
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Research Quality")
        st.markdown("**Target Metrics:**")
        st.markdown("- Sources: 15+ analyzed")
        st.markdown("- Speed: < 60 seconds") 
        st.markdown("- Credibility: 80+ average")
        st.markdown("- Depth: Multi-level")
        
        st.markdown("---")
        st.markdown('<div style="text-align: center; font-size: 0.9rem; color: #888;">Powered by IntelliSource Engine v1.0</div>', unsafe_allow_html=True)
    
    # Handle suggested query
    if 'suggested_query' in st.session_state:
        research_query = st.session_state.suggested_query
        del st.session_state.suggested_query
    
    # Research execution
    if research_button and research_query:
        agent = AdvancedResearchAgent()
        
        with st.spinner("🤖 AI Agent is researching autonomously..."):
            start_time = time.time()
            
            # Show real-time progress
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 🔄 Research Progress")
                research_results = agent.autonomous_research(
                    research_query, 
                    max_depth=research_depth,
                    max_sources=max_sources
                )
            
            end_time = time.time()
            research_time = end_time - start_time
        
        # Display results with advanced visualizations
        if research_results['sources']:
            # Metrics dashboard
            st.markdown("### 📊 Research Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(research_results['sources'])}</h3>
                    <p>Sources Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_credibility = research_results['final_analysis']['average_credibility']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_credibility:.1f}/100</h3>
                    <p>Avg Credibility</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{research_time:.1f}s</h3>
                    <p>Research Time</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                quality_score = research_results['research_quality_score']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{quality_score:.1f}/100</h3>
                    <p>Quality Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Credibility visualization
            st.markdown("### 📈 Source Credibility Analysis")
            
            credibility_data = []
            for source in research_results['sources']:
                domain = urlparse(source['url']).netloc
                credibility_data.append({
                    'Domain': domain,
                    'Credibility Score': source['credibility_score'],
                    'Word Count': source['word_count']
                })
            
            df = pd.DataFrame(credibility_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_credibility = px.bar(
                    df, 
                    x='Domain', 
                    y='Credibility Score',
                    title="Source Credibility Scores",
                    color='Credibility Score',
                    color_continuous_scale='viridis'
                )
                fig_credibility.update_xaxes(tickangle=45)
                st.plotly_chart(fig_credibility, use_container_width=True)
            
            with col2:
                fig_content = px.scatter(
                    df,
                    x='Word Count',
                    y='Credibility Score',
                    size='Word Count',
                    hover_data=['Domain'],
                    title="Content Depth vs Credibility"
                )
                st.plotly_chart(fig_content, use_container_width=True)
            
            # Research summary
            st.markdown("### 🧠 AI-Generated Research Summary")
            
            # Combine all summaries intelligently
            all_summaries = [s['summary'] for s in research_results['sources'] if s.get('summary')]
            combined_content = ' '.join(all_summaries)
            
            if combined_content:
                final_summary = agent.ai_summarize(combined_content, research_query)
                st.markdown(f"""
                <div class="insight-box">
                <h4>🎯 Key Findings</h4>
                <p>{final_summary}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed source analysis
            st.markdown("### 📚 Detailed Source Analysis")
            
            for i, source in enumerate(research_results['sources'], 1):
                with st.expander(f"📄 Source {i}: {source['title'][:60]}... (Credibility: {source['credibility_score']}/100)"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**URL:** {source['url']}")
                        st.markdown(f"**Summary:** {source['summary']}")
                        
                        if source.get('insights'):
                            st.markdown("**Key Insights:**")
                            for insight in source['insights']:
                                st.markdown(f"• {insight}")
                    
                    with col2:
                        st.metric("Words", source['word_count'])
                        st.metric("Sentences", source['sentence_count'])
                        st.metric("Credibility", f"{source['credibility_score']}/100")
            
            # Export functionality
            if export_button:
                # Generate comprehensive report
                report_data = {
                    'research_query': research_query,
                    'timestamp': datetime.now().isoformat(),
                    'results': research_results,
                    'configuration': {
                        'depth': research_depth,
                        'max_sources': max_sources,
                        'analysis_focus': analysis_focus
                    }
                }
                
                st.download_button(
                    label="📥 Download Research Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"research_report_{research_query.replace(' ', '_')}.json",
                    mime="application/json"
                )
                
                # Also generate markdown report
                markdown_report = generate_markdown_report(research_results, research_query)
                st.download_button(
                    label="📝 Download Research Report (Markdown)",
                    data=markdown_report,
                    file_name=f"research_report_{research_query.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
        
        else:
            st.info("🔍 Enter a research topic and click 'Start Research' to begin autonomous analysis")
    
    # Footer with IntelliSource branding
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    <h3 style="color: #667eea; margin-bottom: 1rem;">🧠 IntelliSource</h3>
    <p><strong>Advanced Autonomous AI Research Agent</strong></p>
    <p>Built by <strong>Imaad Mahmood</strong> | Pakistan 🇵🇰</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
    <a href="https://github.com/Imaad18/intellisource" target="_blank">📚 GitHub</a> • 
    <a href="https://linkedin.com/in/imaad-mahmood" target="_blank">💼 LinkedIn</a> • 
    <a href="https://kaggle.com/imaadmahmood" target="_blank">🏆 Kaggle</a>
    </p>
    <p style="font-size: 0.8rem; color: #999;">Autonomous • Intelligent • Comprehensive</p>
    </div>
    """, unsafe_allow_html=True)

def generate_markdown_report(research_data, query):
    """Generate a comprehensive markdown report"""
    report = f"""# Autonomous Research Report: {query}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This research was conducted autonomously by an AI agent, analyzing {len(research_data['sources'])} sources with an average credibility score of {research_data['final_analysis']['average_credibility']:.1f}/100.

## Key Findings
"""
    
    for insight in research_data['final_analysis']['key_insights'][:5]:
        report += f"- {insight}\n"
    
    report += f"""
## Source Analysis
"""
    
    for i, source in enumerate(research_data['sources'], 1):
        report += f"""
### Source {i}: {source['title']}
- **URL:** {source['url']}
- **Credibility Score:** {source['credibility_score']}/100
- **Summary:** {source['summary']}

"""
    
    report += f"""
## Research Quality Metrics
- **Total Sources:** {len(research_data['sources'])}
- **Average Credibility:** {research_data['final_analysis']['average_credibility']:.1f}/100
- **Research Quality Score:** {research_data['research_quality_score']:.1f}/100
- **High-Quality Sources:** {research_data['final_analysis']['high_credibility_sources']}

---
*Generated by IntelliSource - Advanced Autonomous AI Research Agent*
*Built by Imaad Mahmood | https://github.com/Imaad18/intellisource*
"""
    
    return report

if __name__ == "__main__":
    create_streamlit_app()
