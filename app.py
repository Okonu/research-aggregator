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
            'openalex': 'https://api.openalex.org/works'
        }

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

    def search_by_author(self, author_name: str, max_results: int = 10) -> List[Dict]:
        """Search for papers by author name"""
        # Request more from each source to ensure we get enough results
        target_per_source = max_results // 2 + 3  # Split between 2 sources with buffer

        arxiv_query = f'au:{author_name}'
        arxiv_results = self.search_arxiv(arxiv_query, target_per_source)

        # For Semantic Scholar, we'll search by author in the query
        ss_query = f'author:{author_name}'
        ss_results = self.search_semantic_scholar(ss_query, target_per_source)

        # Combine and limit to requested amount
        all_results = arxiv_results + ss_results
        return all_results[:max_results]

def main():
    st.set_page_config(page_title="Research Aggregator", page_icon="📚", layout="wide")

    st.title("📚 Research Paper Aggregator")
    st.markdown("Search across multiple research databases: arXiv, Semantic Scholar, and OpenAlex")

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
                    target_per_source = max_results // 3 + 2  # Add buffer to account for failures
                    arxiv_papers = aggregator.search_arxiv(query, target_per_source)
                    ss_papers = aggregator.search_semantic_scholar(query, target_per_source)
                    openalex_papers = aggregator.search_openalex(query, target_per_source)

                    # Combine and limit to requested amount
                    all_papers = arxiv_papers + ss_papers + openalex_papers
                    all_papers = all_papers[:max_results]  # Trim to exact requested amount
                else:
                    # Search by author
                    all_papers = aggregator.search_by_author(query, max_results)

                if all_papers:
                    st.success(f"Found {len(all_papers)} papers")

                    for i, paper in enumerate(all_papers):
                        with st.expander(f"📄 {paper['title']}", expanded=False):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.write(f"**Authors:** {paper['authors']}")
                                st.write(f"**Source:** {paper['source']}")
                                if paper.get('published'):
                                    st.write(f"**Published:** {paper['published']}")
                                if paper.get('venue'):
                                    st.write(f"**Venue:** {paper['venue']}")

                                if paper['abstract']:
                                    with st.expander("Abstract", expanded=False):
                                        st.write(paper['abstract'])

                            with col2:
                                if paper['url']:
                                    st.link_button("📖 View Full Paper", paper['url'], use_container_width=True)
                                else:
                                    st.info("No direct link available")
                else:
                    st.warning("No papers found. Try different keywords or check your search terms.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()