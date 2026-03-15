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

    def search_institutional_documents(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for direct institutional/regulatory documents"""
        papers = []

        # Federal Reserve SR Letters and Publications
        if any(term in query.lower() for term in ['federal reserve', 'fed', 'sr letter', 'model risk', 'supervisory']):
            fed_docs = self.search_federal_reserve_docs(query, max_results)
            papers.extend(fed_docs)

        # NIST Publications
        if any(term in query.lower() for term in ['nist', 'framework', 'ai risk', 'cybersecurity']):
            nist_docs = self.search_nist_docs(query, max_results)
            papers.extend(nist_docs)

        # EU Regulatory Documents
        if any(term in query.lower() for term in ['eu', 'european', 'gdpr', 'ai act', 'regulation', 'directive']):
            eu_docs = self.search_eu_docs(query, max_results)
            papers.extend(eu_docs)

        return papers[:max_results]

    def search_federal_reserve_docs(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Federal Reserve institutional documents"""
        # Known Federal Reserve SR letters and key documents related to model risk
        fed_documents = [
            {
                'title': 'SR 11-7: Guidance on Model Risk Management',
                'authors': 'Board of Governors of the Federal Reserve System',
                'abstract': 'This guidance outlines sound practices for managing model risk, which is the potential for adverse consequences from decisions based on incorrect or misused models.',
                'url': 'https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm',
                'published': '2011',
                'venue': 'Federal Reserve Supervisory Letters',
                'type': 'Supervisory Guidance',
                'source': 'Federal Reserve',
                'doc_type': 'regulatory'
            },
            {
                'title': 'SR 09-4: Applying Supervisory Guidance and Regulations on the Management of Model Risk',
                'authors': 'Board of Governors of the Federal Reserve System',
                'abstract': 'Guidance on supervisory expectations for model risk management practices at banking organizations.',
                'url': 'https://www.federalreserve.gov/supervisionreg/srletters/sr0904.htm',
                'published': '2009',
                'venue': 'Federal Reserve Supervisory Letters',
                'type': 'Supervisory Guidance',
                'source': 'Federal Reserve',
                'doc_type': 'regulatory'
            },
            {
                'title': 'Federal Reserve System Model Risk Management Guidance',
                'authors': 'Federal Reserve System',
                'abstract': 'Comprehensive framework for identifying, measuring, monitoring, and controlling model risk in financial institutions.',
                'url': 'https://www.federalreserve.gov/supervisionreg/topics/model-risk-management.htm',
                'published': '2023',
                'venue': 'Federal Reserve Supervision and Regulation',
                'type': 'Regulatory Framework',
                'source': 'Federal Reserve',
                'doc_type': 'regulatory'
            }
        ]

        # Filter documents based on query relevance
        relevant_docs = []
        query_terms = query.lower().split()

        for doc in fed_documents:
            doc_text = f"{doc['title']} {doc['abstract']}".lower()
            if any(term in doc_text for term in query_terms):
                relevant_docs.append(doc)

        return relevant_docs[:max_results]

    def search_nist_docs(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search NIST institutional documents"""
        nist_documents = [
            {
                'title': 'AI Risk Management Framework (AI RMF 1.0)',
                'authors': 'National Institute of Standards and Technology',
                'abstract': 'Framework to better manage risks to individuals, organizations, and society associated with artificial intelligence.',
                'url': 'https://www.nist.gov/itl/ai-risk-management-framework',
                'published': '2023',
                'venue': 'NIST Special Publication',
                'type': 'Technical Framework',
                'source': 'NIST',
                'doc_type': 'regulatory'
            },
            {
                'title': 'NIST Cybersecurity Framework 2.0',
                'authors': 'National Institute of Standards and Technology',
                'abstract': 'Framework for improving critical infrastructure cybersecurity and organizational cyber risk management.',
                'url': 'https://www.nist.gov/cyberframework',
                'published': '2024',
                'venue': 'NIST Framework',
                'type': 'Technical Framework',
                'source': 'NIST',
                'doc_type': 'regulatory'
            }
        ]

        # Filter based on query relevance
        relevant_docs = []
        query_terms = query.lower().split()

        for doc in nist_documents:
            doc_text = f"{doc['title']} {doc['abstract']}".lower()
            if any(term in doc_text for term in query_terms):
                relevant_docs.append(doc)

        return relevant_docs[:max_results]

    def search_eu_docs(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search EU institutional documents"""
        eu_documents = [
            {
                'title': 'EU AI Act (Artificial Intelligence Act)',
                'authors': 'European Parliament and Council of the European Union',
                'abstract': 'Regulation laying down harmonised rules on artificial intelligence and amending certain Union legislative acts.',
                'url': 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj',
                'published': '2024',
                'venue': 'Official Journal of the European Union',
                'type': 'EU Regulation',
                'source': 'European Union',
                'doc_type': 'regulatory'
            },
            {
                'title': 'General Data Protection Regulation (GDPR)',
                'authors': 'European Parliament and Council of the European Union',
                'abstract': 'Regulation on the protection of natural persons with regard to the processing of personal data and on the free movement of such data.',
                'url': 'https://eur-lex.europa.eu/eli/reg/2016/679/oj',
                'published': '2016',
                'venue': 'Official Journal of the European Union',
                'type': 'EU Regulation',
                'source': 'European Union',
                'doc_type': 'regulatory'
            },
            {
                'title': 'EU Digital Services Act (DSA)',
                'authors': 'European Parliament and Council of the European Union',
                'abstract': 'Regulation on a Single Market for Digital Services and amending Directive 2000/31/EC.',
                'url': 'https://eur-lex.europa.eu/eli/reg/2022/2065/oj',
                'published': '2022',
                'venue': 'Official Journal of the European Union',
                'type': 'EU Regulation',
                'source': 'European Union',
                'doc_type': 'regulatory'
            },
            {
                'title': 'MiCA (Markets in Crypto-Assets Regulation)',
                'authors': 'European Parliament and Council of the European Union',
                'abstract': 'Regulation on markets in crypto-assets, and amending Regulations (EU) No 1093/2010 and (EU) No 1095/2010 and Directives 2013/36/EU and (EU) 2019/1937.',
                'url': 'https://eur-lex.europa.eu/eli/reg/2023/1114/oj',
                'published': '2023',
                'venue': 'Official Journal of the European Union',
                'type': 'EU Regulation',
                'source': 'European Union',
                'doc_type': 'regulatory'
            }
        ]

        # Filter based on query relevance
        relevant_docs = []
        query_terms = query.lower().split()

        for doc in eu_documents:
            doc_text = f"{doc['title']} {doc['abstract']}".lower()
            if any(term in doc_text for term in query_terms):
                relevant_docs.append(doc)

        return relevant_docs[:max_results]

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
                    # Search all sources with higher targets to ensure enough results for each category
                    # Each tab should be able to show up to max_results papers
                    target_per_source = max_results + 5  # Buffer to ensure enough results

                    institutional_docs = aggregator.search_institutional_documents(query, max_results * 2)
                    arxiv_papers = aggregator.search_arxiv(query, target_per_source)
                    ss_papers = aggregator.search_semantic_scholar(query, target_per_source)
                    openalex_papers = aggregator.search_openalex(query, target_per_source)
                    crossref_papers = aggregator.search_crossref(query, target_per_source)

                    # Combine all sources (no total limit yet - we'll limit per category)
                    all_papers = institutional_docs + arxiv_papers + ss_papers + openalex_papers + crossref_papers
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

                        # Limit each category to max_results
                        if doc_type == 'regulatory' and len(regulatory_docs) < max_results:
                            regulatory_docs.append(paper)
                        elif doc_type == 'conference' and len(conference_papers) < max_results:
                            conference_papers.append(paper)
                        elif doc_type == 'journal' and len(journal_papers) < max_results:
                            journal_papers.append(paper)
                        elif doc_type == 'research' and len(research_papers) < max_results:
                            research_papers.append(paper)

                    total_found = len(regulatory_docs) + len(conference_papers) + len(journal_papers) + len(research_papers)
                    st.success(f"Found {total_found} papers across all categories")

                    # Try to use tabs, fallback to expanders if tabs not supported
                    try:
                        # Create tabs for different document types (each limited to max_results)
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

                    except AttributeError:
                        # Fallback: Use expanders if tabs not available
                        st.info("Using fallback display mode (tabs not supported in this Streamlit version)")

                        if regulatory_docs:
                            with st.expander(f"🏛️ Regulatory Guidance ({len(regulatory_docs)})", expanded=True):
                                for paper in regulatory_docs:
                                    aggregator.display_paper_card(paper)

                        if conference_papers:
                            with st.expander(f"🎯 Conference Papers ({len(conference_papers)})", expanded=len(regulatory_docs) == 0):
                                for paper in conference_papers:
                                    aggregator.display_paper_card(paper)

                        if journal_papers:
                            with st.expander(f"📋 Journal Articles ({len(journal_papers)})", expanded=len(regulatory_docs) == 0 and len(conference_papers) == 0):
                                for paper in journal_papers:
                                    aggregator.display_paper_card(paper)

                        if research_papers:
                            with st.expander(f"📄 Research Papers ({len(research_papers)})", expanded=len(regulatory_docs) == 0 and len(conference_papers) == 0 and len(journal_papers) == 0):
                                for paper in research_papers:
                                    aggregator.display_paper_card(paper)
                else:
                    st.warning("No papers found. Try different keywords or check your search terms.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()