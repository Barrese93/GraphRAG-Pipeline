"""Graph nodes for the LangGraph workflow - Refactored as Tools."""

import logging
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_query_rewriter_chain(llm):
    """Create a query rewriter chain for Adaptive RAG."""
    system_prompt = """Sei un riscrittore di domande che converte una domanda di input in una versione migliore, ottimizzata per la ricerca vettoriale o web. Analizza l'input e ragiona sull'intento semantico sottostante."""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Ecco la domanda iniziale: \n\n {question} \n\n Formula una domanda migliorata."),
    ])
    return rewrite_prompt | llm | StrOutputParser()


@tool
def rewrite_query_tool(question: str, rewriter_chain: Any = None) -> str:
    """
    Riscrive una domanda per ottimizzarla per la ricerca. Utile se la domanda iniziale è ambigua.
    Input: la domanda originale.
    Output: la domanda riscritta.
    """
    if rewriter_chain is None:
        return question
    try:
        return rewriter_chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"Error in query rewriting: {str(e)}")
        return question


@tool
def grade_documents_tool(documents: List[Dict], question: str, relevance_grader: Any = None) -> List[Dict]:
    """
    Filtra i documenti per pertinenza, scartando quelli non rilevanti.
    Input: una lista di documenti e la domanda dell'utente.
    Output: una lista di documenti filtrata contenente solo quelli pertinenti.
    """
    if not relevance_grader or not documents:
        return documents
    
    try:
        filtered_docs = []
        for d in documents:
            try:
                content = d.get("content", "") or d.get("page_content", "")
                if content:
                    grade = relevance_grader.invoke({"question": question, "document": content})
                    if grade.binary_score == "yes":
                        filtered_docs.append(d)
            except Exception as doc_error:
                logger.warning(f"Error grading document: {str(doc_error)}")
                # Keep document if grading fails
                filtered_docs.append(d)
        return filtered_docs
    except Exception as e:
        logger.error(f"Error in document grading: {str(e)}")
        return documents


@tool
def grade_answer_tool(generation: str, documents: List[Dict], question: str, hallucination_grader: Any = None, usefulness_grader: Any = None) -> str:
    """
    Valuta la qualità di una risposta finale. Controlla se è basata sui fatti (non ha allucinazioni)
    e se risponde effettivamente alla domanda.
    Output: 'utile' se entrambi i controlli passano, altrimenti 'non utile' o 'non supportato'.
    """
    if not hallucination_grader or not usefulness_grader:
        return "utile"  # Default to useful if graders not available
    
    try:
        # Check hallucinations
        if hallucination_grader:
            # Extract document contents for hallucination check
            doc_contents = [d.get("content", "") or d.get("page_content", "") for d in documents]
            doc_context = "\n\n".join([content for content in doc_contents if content])
            
            hallucination_grade = hallucination_grader.invoke({
                "documents": doc_context, 
                "generation": generation
            })
            if hallucination_grade.binary_score == 'no':
                return "non supportato"

        # Check usefulness
        if usefulness_grader:
            usefulness_grade = usefulness_grader.invoke({
                "question": question, 
                "generation": generation
            })
            if usefulness_grade.binary_score == 'no':
                return "non utile"

        return "utile"
    except Exception as e:
        logger.error(f"Error in answer grading: {str(e)}")
        return "utile"  # Default to useful if grading fails


def create_hybrid_search_tool(vector_store: Any = None, graph: Any = None):
    """Create a hybrid search tool with injected dependencies."""
    
    @tool
    def hybrid_search_tool(query: str) -> List[Dict]:
        """
        Esegue una ricerca ibrida (vettoriale e per parole chiave) nel knowledge graph per trovare documenti rilevanti.
        Usa questa funzione per domande generiche o che richiedono di trovare informazioni all'interno dei testi.
        Restituisce una lista di documenti con contenuto e metadati, arricchiti con contesto dal grafo.
        
        Args:
            query: La domanda dell'utente da cercare
            
        Returns:
            Lista di documenti trovati con contenuto e metadati
        """
        if not vector_store:
            return []
        
        try:
            all_results = []
            # Perform hybrid search with metadata filtering if available
            search_kwargs = {"k": 3}
            
            # Perform hybrid search - this will use both vector and keyword search
            results = vector_store.similarity_search(query, **search_kwargs)
            
            # Enhance results with custom retrieval if graph is available
            enhanced_results = []
            for doc in results:
                enhanced_doc = {
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                }
                
                # If graph is available, fetch additional context
                if graph:
                    try:
                        # Get document ID from metadata
                        doc_id = doc.metadata.get('id') or doc.metadata.get('source', '')
                        if doc_id:
                            # Fetch related information from the graph
                            related_query = """
                                MATCH (d:Document)-[r]-(related)
                                WHERE d.id = $doc_id OR d.source = $doc_id
                                RETURN d.name as document_name,
                                       collect(DISTINCT related.name) as related_entities,
                                       collect(DISTINCT labels(related)[0]) as entity_types
                                LIMIT 5
                            """
                            related_results = graph.query(related_query, params={"doc_id": doc_id})
                            if related_results:
                                enhanced_doc["related_context"] = related_results[0]
                    except Exception as graph_error:
                        logger.warning(f"Error fetching related context: {str(graph_error)}")
                
                enhanced_results.append(enhanced_doc)
            
            all_results.extend(enhanced_results)
            logger.info(f"Found {len(all_results)} hybrid search results with enhanced context")
            return all_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    return hybrid_search_tool

def create_structured_query_tool(graph_qa_chain: Any = None, graph: Any = None):
    """Create a structured query tool with injected dependencies."""
    
    @tool
    def structured_query_tool(query: str) -> List[Dict]:
        """
        Esegue una query strutturata (Cypher) sul knowledge graph.
        Usa questa funzione per domande specifiche su conteggi, relazioni dirette o proprietà di entità.
        (es. "Chi ha firmato X?", "Quanti provvedimenti nel 2024?").
        
        Args:
            query: La domanda strutturata da eseguire sul grafo
            
        Returns:
            Lista di risultati strutturati dal grafo
        """
        if not graph_qa_chain and not graph:
            return []
        
        try:
            all_results = []
            try:
                # Use GraphCypherQAChain to generate and execute Cypher query
                result = graph_qa_chain.invoke({"query": query})
                all_results.append({
                    "question": query,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error in GraphCypherQAChain for query '{query}': {str(e)}")
                # Fallback to simple search
                if graph:
                    cypher_query = """
                        MATCH (e)
                        WHERE toLower(e.name) CONTAINS toLower($query) 
                           OR toLower(e.description) CONTAINS toLower($query)
                        RETURN e.type as type, e.name as name, e.description as description
                        LIMIT 5
                    """
                    results = graph.query(cypher_query, params={"query": query})
                    all_results.extend([dict(record) for record in results])
            
            logger.info(f"Found {len(all_results)} graph query results")
            return all_results
        except Exception as e:
            logger.error(f"Error in structured query: {str(e)}")
            return []
    
    return structured_query_tool

def create_web_search_tool(tavily_client: Any = None):
    """Create a web search tool with injected dependencies."""
    
    @tool
    def web_search_tool(query: str) -> List[Dict]:
        """
        Esegue una ricerca sul web quando le informazioni non sono disponibili nel knowledge graph.
        Usa questa funzione come fallback se gli altri tool non trovano risultati pertinenti.
        
        Args:
            query: La domanda da cercare sul web
            
        Returns:
            Lista di risultati della ricerca web
        """
        if not tavily_client:
            logger.warning("Tavily client not initialized - skipping web search")
            return []
        
        try:
            # Perform web search
            search_results = tavily_client.search(
                query=query,
                max_results=3,
                search_depth="advanced"
            )
            
            # Extract and format results
            web_results = []
            if "results" in search_results:
                for result in search_results["results"]:
                    web_results.append({
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "title": result.get("title", "")
                    })
            
            logger.info(f"Found {len(web_results)} web search results")
            return web_results
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    return web_search_tool

def create_metadata_filter_tool(vector_store: Any = None):
    """Create a metadata filter tool with injected dependencies."""
    
    @tool
    def metadata_filter_tool(query: str, filter_dict: Dict[str, Any]) -> List[Dict]:
        """
        Esegue una ricerca vettoriale con filtri sui metadati.
        Usa questa funzione per domande che richiedono di filtrare per proprietà specifiche (es. "provvedimenti del 2024").
        
        Args:
            query: La domanda dell'utente da cercare
            filter_dict: Un dizionario con i filtri da applicare (es. {"document_type": "Provvedimento"})
            
        Returns:
            Lista di documenti trovati con contenuto e metadati
        """
        if not vector_store:
            return []
        
        try:
            results = vector_store.similarity_search(query, k=3, filter=filter_dict)
            
            enhanced_results = [{
                "content": doc.page_content, 
                "metadata": doc.metadata
            } for doc in results]
            
            logger.info(f"Found {len(enhanced_results)} results with metadata filter")
            return enhanced_results
        except Exception as e:
            logger.error(f"Error in metadata filter search: {str(e)}")
            return []
    
    return metadata_filter_tool
