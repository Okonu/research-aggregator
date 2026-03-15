import streamlit as st
import requests
import xmltodict
from typing import List, Dict
import json

class ResearchAggregator:
    def __init__(self):
        self.base_urls = {
            'arxiv': 'http://export.arxiv.org/api/query',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'openalex': 'https://api.openalex.org/works',
            'crossref': 'https://api.crossref.org/works',
            'core': 'https://api.core.ac.uk/v3/search/works'
        }

    def classify_document_type(self, paper: Dict) -> str:
        """Classify document type based on content and source"""
        title_lower = paper.get('title', '').lower()
        venue_lower = paper.get('venue', '').lower()
        authors_lower = paper.get('authors', '').lower()
        source_lower = paper.get('source', '').lower()

        # Regulatory/Institutional patterns
        regulatory_keywords = [
            'federal reserve', 'sr letter', 'guidance', 'supervisory', 'regulatory',
            'nist', 'framework', 'compliance', 'basel', 'regulation', 'policy',
            'central bank', 'fed', 'treasury', 'sec', 'occ', 'fdic', 'bis',
            'risk management', 'prudential', 'capital requirements', 'stress test'
        ]

        # Conference patterns
        conference_keywords = [
            'icml', 'neurips', 'nips', 'iclr', 'aaai', 'ijcai', 'kdd', 'proceedings',
            'conference', 'workshop', 'symposium', 'annual meeting', 'acm',
            'ieee conference', 'international conference'
        ]

        # Check title, venue, and authors
        text_to_check = f"{title_lower} {venue_lower} {authors_lower}"

        # Priority: Regulatory first (more specific)
        if any(keyword in text_to_check for keyword in regulatory_keywords):
            return 'regulatory'

        # Check for conference content
        if any(keyword in text_to_check for keyword in conference_keywords):
            return 'conference'

        # Check venue for journal patterns
        journal_patterns = ['journal', 'review', 'letters', 'transactions', 'quarterly']
        if any(pattern in venue_lower for pattern in journal_patterns):
            return 'journal'

        # Default to research paper
        return 'research'

    def display_paper_card(self, paper: Dict):
        """Display individual paper card with appropriate styling and icons"""
        doc_type = paper.get('doc_type', 'research')

        # Choose icon and styling based on document type
        if doc_type == 'regulatory':
            icon = "🏛️"
            badge_color = "red"
            badge_text = "REGULATORY"
        elif doc_type == 'conference':
            icon = "🎯"
            badge_color = "blue"
            badge_text = "CONFERENCE"
        elif doc_type == 'journal':
            icon = "📋"
            badge_color = "green"
            badge_text = "JOURNAL"
        else:
            icon = "📄"
            badge_color = "gray"
            badge_text = "RESEARCH"

        with st.expander(f"{icon} {paper['title']}", expanded=False):
            # Document type badge
            st.markdown(f'<span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold;">{badge_text}</span>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Authors:** {paper['authors']}")
                st.write(f"**Source:** {paper['source']}")
                if paper.get('published'):
                    st.write(f"**Published:** {paper['published']}")
                if paper.get('venue'):
                    st.write(f"**Venue:** {paper['venue']}")
                if paper.get('type'):
                    st.write(f"**Type:** {paper['type']}")

                if paper['abstract'] and paper['abstract'] != 'No abstract available':
                    with st.expander("Abstract", expanded=False):
                        st.write(paper['abstract'])

            with col2:
                if paper['url']:
                    st.link_button("📖 View Full Paper", paper['url'], use_container_width=True)
                else:
                    st.info("No direct link available")

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv for papers"""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results
        }

        response = requests.get(self.base_urls['arxiv'], params=params)
        if response.status_code == 200:
            data = xmltodict.parse(response.content)
            entries = data.get('feed', {}).get('entry', [])

            if isinstance(entries, dict):
                entries = [entries]

            papers = []
            for entry in entries:
                authors = entry.get('author', [])
                if isinstance(authors, dict):
                    authors = [authors]
                author_names = [auth.get('name', '') for auth in authors]

                paper = {
                    'title': entry.get('title', '').replace('\n', ' ').strip(),
                    'authors': ', '.join(author_names),
                    'abstract': entry.get('summary', '').replace('\n', ' ').strip(),
                    'url': entry.get('id', ''),
                    'published': entry.get('published', ''),
                    'source': 'arXiv'
                }
                papers.append(paper)
            return papers
        return []

    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,abstract,url,year,venue'
        }

        try:
            response = requests.get(self.base_urls['semantic_scholar'], params=params)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for paper_data in data.get('data', []):
                    authors = paper_data.get('authors', [])
                    author_names = [auth.get('name', '') for auth in authors]

                    paper = {
                        'title': paper_data.get('title', ''),
                        'authors': ', '.join(author_names),
                        'abstract': paper_data.get('abstract', '') or 'No abstract available',
                        'url': paper_data.get('url', ''),
                        'published': str(paper_data.get('year', '')),
                        'venue': paper_data.get('venue', ''),
                        'source': 'Semantic Scholar'
                    }
                    papers.append(paper)
                return papers
        except Exception as e:
            st.error(f"Error searching Semantic Scholar: {e}")
        return []

    def search_openalex(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search OpenAlex for papers"""
        params = {
            'search': query,
            'per_page': max_results,
            'select': 'title,authorships,abstract_inverted_index,open_access,publication_year,host_venue'
        }

        try:
            response = requests.get(self.base_urls['openalex'], params=params)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for work in data.get('results', []):
                    authorships = work.get('authorships', [])
                    author_names = [auth.get('author', {}).get('display_name', '') for auth in authorships]

                    abstract = "No abstract available"
                    if work.get('abstract_inverted_index'):
                        # Reconstruct abstract from inverted index
                        try:
                            inverted = work['abstract_inverted_index']
                            words = [''] * 500  # Max length
                            for word, positions in inverted.items():
                                for pos in positions:
                                    if pos < len(words):
                                        words[pos] = word
                            abstract = ' '.join([w for w in words if w]).strip()
                        except:
                            abstract = "Abstract available but could not be parsed"

                    url = work.get('open_access', {}).get('oa_url', '') or work.get('id', '')

                    paper = {
                        'title': work.get('title', ''),
                        'authors': ', '.join(author_names),
                        'abstract': abstract,
                        'url': url,
                        'published': str(work.get('publication_year', '')),
                        'venue': work.get('host_venue', {}).get('display_name', '') if work.get('host_venue') else '',
                        'source': 'OpenAlex'
                    }
                    papers.append(paper)
                return papers
        except Exception as e:
            st.error(f"Error searching OpenAlex: {e}")
        return []

    def search_crossref(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search CrossRef for conference papers and journal articles"""
        params = {
            'query': query,
            'rows': max_results,
            'select': 'title,author,abstract,URL,published-print,container-title,type'
        }

        try:
            response = requests.get(self.base_urls['crossref'], params=params)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for work in data.get('message', {}).get('items', []):
                    authors = work.get('author', [])
                    author_names = []
                    for author in authors:
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if given and family:
                            author_names.append(f"{given} {family}")
                        elif family:
                            author_names.append(family)

                    # Get publication date
                    pub_date = ''
                    if work.get('published-print', {}).get('date-parts'):
                        year = work['published-print']['date-parts'][0][0]
                        pub_date = str(year)

                    # Get venue (journal/conference name)
                    venue = ''
                    if work.get('container-title'):
                        venue = work['container-title'][0] if work['container-title'] else ''

                    paper = {
                        'title': work.get('title', [''])[0] if work.get('title') else '',
                        'authors': ', '.join(author_names),
                        'abstract': work.get('abstract', 'No abstract available'),
                        'url': work.get('URL', ''),
                        'published': pub_date,
                        'venue': venue,
                        'type': work.get('type', ''),
                        'source': 'CrossRef'
                    }
                    papers.append(paper)
                return papers
        except Exception as e:
            st.error(f"Error searching CrossRef: {e}")
        return []

    def search_core(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search CORE for academic papers"""
        headers = {
            'Authorization': 'Bearer YOUR_API_KEY_HERE'  # You'd need to register for CORE API
        }
        params = {
            'q': query,
            'limit': max_results
        }

        try:
            response = requests.get(self.base_urls['core'], params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for work in data.get('results', []):
                    authors_list = work.get('authors', [])
                    author_names = [author.get('name', '') for author in authors_list if author.get('name')]

                    paper = {
                        'title': work.get('title', ''),
                        'authors': ', '.join(author_names),
                        'abstract': work.get('abstract', 'No abstract available'),
                        'url': work.get('downloadUrl', '') or work.get('urls', [''])[0] if work.get('urls') else '',
                        'published': str(work.get('yearPublished', '')),
                        'venue': work.get('journals', [{}])[0].get('title', '') if work.get('journals') else '',
                        'source': 'CORE'
                    }
                    papers.append(paper)
                return papers
        except Exception as e:
            # CORE API requires authentication, so we'll skip it for now
            pass
        return []

    def search_by_author(self, author_name: str, max_results: int = 10) -> List[Dict]:
        """Search for papers by author name"""
        # Request more from each source to ensure we get enough results
        target_per_source = max_results // 3 + 2  # Split between 3 sources with buffer

        arxiv_query = f'au:{author_name}'
        arxiv_results = self.search_arxiv(arxiv_query, target_per_source)

        # For Semantic Scholar, we'll search by author in the query
        ss_query = f'author:{author_name}'
        ss_results = self.search_semantic_scholar(ss_query, target_per_source)

        # CrossRef author search
        crossref_query = f'author:{author_name}'
        crossref_results = self.search_crossref(crossref_query, target_per_source)

        # Combine and limit to requested amount
        all_results = arxiv_results + ss_results + crossref_results
        return all_results[:max_results]

def main():
    st.set_page_config(page_title="Research Aggregator", page_icon="📚", layout="wide")

    st.title("📚 Research Paper Aggregator")
    st.markdown("Search across multiple research databases: arXiv, Semantic Scholar, OpenAlex, and CrossRef")

    aggregator = ResearchAggregator()

    # Search options
    search_type = st.radio("Search by:", ["Topic/Title", "Author"], horizontal=True)

    if search_type == "Topic/Title":
        query = st.text_input("Enter research topic, keywords, or paper title:",
                             placeholder="e.g., machine learning, neural networks, quantum computing")
    else:
        query = st.text_input("Enter author name:",
                             placeholder="e.g., Geoffrey Hinton, Yann LeCun")

    col1, col2 = st.columns([1, 4])
    with col1:
        max_results = st.selectbox("Max results:", [5, 10, 20, 30], index=1)

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching research databases..."):
                all_papers = []

                if search_type == "Topic/Title":
                    # Search all databases for topic/title - request more from each to ensure we get enough results
                    target_per_source = max_results // 4 + 2  # Now splitting among 4 sources
                    arxiv_papers = aggregator.search_arxiv(query, target_per_source)
                    ss_papers = aggregator.search_semantic_scholar(query, target_per_source)
                    openalex_papers = aggregator.search_openalex(query, target_per_source)
                    crossref_papers = aggregator.search_crossref(query, target_per_source)

                    # Combine and limit to requested amount
                    all_papers = arxiv_papers + ss_papers + openalex_papers + crossref_papers
                    all_papers = all_papers[:max_results]  # Trim to exact requested amount
                else:
                    # Search by author
                    all_papers = aggregator.search_by_author(query, max_results)

                if all_papers:
                    # Classify and categorize papers
                    research_papers = []
                    regulatory_docs = []
                    conference_papers = []
                    journal_papers = []

                    for paper in all_papers:
                        doc_type = aggregator.classify_document_type(paper)
                        paper['doc_type'] = doc_type

                        if doc_type == 'regulatory':
                            regulatory_docs.append(paper)
                        elif doc_type == 'conference':
                            conference_papers.append(paper)
                        elif doc_type == 'journal':
                            journal_papers.append(paper)
                        else:
                            research_papers.append(paper)

                    st.success(f"Found {len(all_papers)} papers")

                    # Create tabs for different document types
                    tab1, tab2, tab3, tab4 = st.tabs([
                        f"📄 Research Papers ({len(research_papers)})",
                        f"🏛️ Regulatory Guidance ({len(regulatory_docs)})",
                        f"🎯 Conference Papers ({len(conference_papers)})",
                        f"📋 Journal Articles ({len(journal_papers)})"
                    ])

                    with tab1:
                        if research_papers:
                            for paper in research_papers:
                                aggregator.display_paper_card(paper)
                        else:
                            st.info("No research papers found in this category.")

                    with tab2:
                        if regulatory_docs:
                            for paper in regulatory_docs:
                                aggregator.display_paper_card(paper)
                        else:
                            st.info("No regulatory documents found.")

                    with tab3:
                        if conference_papers:
                            for paper in conference_papers:
                                aggregator.display_paper_card(paper)
                        else:
                            st.info("No conference papers found.")

                    with tab4:
                        if journal_papers:
                            for paper in journal_papers:
                                aggregator.display_paper_card(paper)
                        else:
                            st.info("No journal articles found.")
                else:
                    st.warning("No papers found. Try different keywords or check your search terms.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()