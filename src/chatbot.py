"""Advanced GraphRAG chatbot using LangGraph for intelligent query handling - Agent-based approach."""

import logging
from typing import Dict, List, Any, Optional
from functools import partial
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from tavily import TavilyClient

from config.settings import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, 
    OPENAI_API_KEY, OUTPUT_JSON_PATH, TAVILY_API_KEY
)

# Import from new modular files
from src.graph_state import AgentState
from src.graph_nodes import (
    create_hybrid_search_tool, create_structured_query_tool, create_web_search_tool,
    rewrite_query_tool, grade_documents_tool, grade_answer_tool, create_query_rewriter_chain
)
from src.graders import (
    create_relevance_grader, create_hallucination_grader, create_answer_usefulness_grader
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGChatbot:
    """Advanced GraphRAG chatbot using LangGraph with agent-based approach."""

    # Define a custom prompt for Cypher query generation
    CYPHER_GENERATION_TEMPLATE = """
Sei un esperto di Neo4j e Cypher. Il tuo compito è generare query Cypher per rispondere a domande degli utenti basate su uno schema di grafo.

Istruzioni:
1.  **Ispezione dello Schema**: Prima di tutto, ispeziona lo schema del grafo fornito per capire i tipi di nodi, le loro proprietà e le relazioni tra di essi.
2.  **Query Semplici**: Per domande semplici, genera una query Cypher diretta.
3.  **Query Complesse**: Per domande complesse, scomponile in sotto-query logiche.
4.  **Flessibilità**: Usa `toLower()` e `CONTAINS` per ricerche di testo flessibili, poiché i dati potrebbero non corrispondere esattamente. Non usare l'operatore `= ` per le stringhe a meno che non sia esplicitamente richiesto.
5.  **Date**: Quando cerchi per data, considera che la data potrebbe essere parte di una stringa più lunga (es. nel nome di un file). Usa `CONTAINS`.
6.  **Efficienza**: Assicurati che le query siano efficienti. Usa `MATCH` e `WHERE` in modo appropriato.
7.  **Sicurezza**: Non generare query che modificano i dati (es. `CREATE`, `SET`, `DELETE`).
8.  **Output**: Restituisci solo la query Cypher, senza spiegazioni aggiuntive.

**Schema del Grafo:**
{schema}

**Esempi di Query:**

Domanda: "Quali provvedimenti sono stati emessi dal Garante Privacy?"
Query:
```cypher
MATCH (p:Provvedimento)-[:EMESSO_DA]->(e:Ente)
WHERE toLower(e.name) CONTAINS toLower('Garante Privacy')
RETURN p.name, p.description
```

Domanda: "Quali documenti sono stati pubblicati il 10 febbraio 1998?"
Query:
```cypher
MATCH (p)
WHERE p.publication_date = '1998-02-10'
RETURN p.name as name, p.description as description, labels(p) as type
```

Domanda: "Trova i documenti che menzionano il GDPR"
Query:
```cypher
MATCH (d)-[:MENZIONA]->(e:FonteNormativa)
WHERE toLower(e.name) CONTAINS toLower('GDPR')
RETURN d.name, d.description
```

Domanda: "Cosa è successo il 10 febbraio 1998?"
Query:
```cypher
MATCH (p)
WHERE p.publication_date = '1998-02-10'
RETURN p.name as name, p.description as description, labels(p) as type
```

**Domanda dell'utente:**
{question}

**Query Cypher:**
"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the GraphRAG chatbot."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Initialize Neo4j graph for structured queries
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        # Initialize GraphCypherQAChain for structured queries with custom prompt
        self.graph_qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            cypher_prompt=self.CYPHER_GENERATION_PROMPT,
            allow_dangerous_requests=True
        )
        
        # Initialize vector store for hybrid search
        try:
            # First try to connect to existing hybrid index
            self.vector_store = Neo4jVector.from_existing_index(
                OpenAIEmbeddings(openai_api_key=openai_api_key),
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="vector_index",
                keyword_index_name="keyword_index",  # Add keyword index for hybrid search
                text_node_property="description",
                embedding_node_property="embedding",
                search_type="hybrid"  # Enable hybrid search
            )
        except Exception as e:
            logger.warning(f"Could not initialize hybrid vector store from existing index: {str(e)}")
            # If that fails, try to create from existing graph data with hybrid support
            try:
                # Get all unique labels from the graph
                labels_query = """
                    MATCH (n) 
                    WITH DISTINCT labels(n) as label_list
                    UNWIND label_list as label
                    RETURN DISTINCT label
                """
                result = self.graph.query(labels_query)
                all_labels = [record['label'] for record in result]
                
                # Create vector store from existing graph data for all labels with hybrid search
                self.vector_store = Neo4jVector.from_existing_graph(
                    OpenAIEmbeddings(openai_api_key=openai_api_key),
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    index_name="vector_index",
                    node_label="FonteNormativa",  # Start with one common label
                    text_node_properties=["name", "description"],
                    embedding_node_property="embedding",
                    search_type="hybrid",  # Enable hybrid search
                    keyword_index_name="keyword_index"  # Add keyword index
                )
            except Exception as e2:
                logger.warning(f"Could not initialize hybrid vector store from existing graph: {str(e2)}")
                # Fallback to basic vector store
                try:
                    self.vector_store = Neo4jVector.from_existing_index(
                        OpenAIEmbeddings(openai_api_key=openai_api_key),
                        url=NEO4J_URI,
                        username=NEO4J_USERNAME,
                        password=NEO4J_PASSWORD,
                        index_name="vector_index",
                        text_node_property="description",
                        embedding_node_property="embedding"
                    )
                except Exception as e3:
                    logger.warning(f"Could not initialize basic vector store: {str(e3)}")
                    self.vector_store = None
        
        # Initialize Tavily client for web search
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        
        # Initialize answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un assistente esperto in normativa sulla privacy. Usa il contesto fornito dai tool per rispondere in modo accurato e dettagliato alla domanda dell'utente.

Istruzioni:
- Rispondi solo con informazioni presenti nel contesto fornito dai tool
- Se il contesto non contiene informazioni sufficienti, rispondi "Non ho informazioni sufficienti per rispondere a questa domanda."
- Sii preciso e cita fonti specifiche quando possibile
- Usa un linguaggio chiaro e professionale"""),
            ("human", "{question}")
        ])
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()
    
    def build_workflow(self) -> StateGraph:
        """Costruisce il workflow agentico con un ciclo ReAct."""
        workflow = StateGraph(AgentState)

        # 1. Create tools with injected dependencies
        hybrid_search_tool = create_hybrid_search_tool(vector_store=self.vector_store, graph=self.graph)
        structured_query_tool = create_structured_query_tool(graph_qa_chain=self.graph_qa_chain, graph=self.graph)
        web_search_tool = create_web_search_tool(tavily_client=self.tavily_client)
        
        # Create advanced RAG tools
        query_rewriter_chain = create_query_rewriter_chain(self.llm)
        
        # Create tool instances with proper dependency injection
        def rewrite_tool_func(question: str) -> str:
            return rewrite_query_tool(question, query_rewriter_chain)
        
        relevance_grader = create_relevance_grader(self.llm)
        hallucination_grader = create_hallucination_grader(self.llm)
        usefulness_grader = create_answer_usefulness_grader(self.llm)
        
        def grade_docs_tool_func(documents: List[Dict], question: str) -> List[Dict]:
            return grade_documents_tool(documents, question, relevance_grader)
        
        def grade_answer_tool_func(generation: str, documents: List[Dict], question: str) -> str:
            return grade_answer_tool(generation, documents, question, hallucination_grader, usefulness_grader)
        
        # Update tool names and descriptions for clarity
        from langchain_core.tools import Tool
        
        rewrite_tool = Tool(
            name="rewrite_query_tool",
            description="Riscrive una domanda per ottimizzarla per la ricerca. Utile se la domanda iniziale è ambigua.",
            func=rewrite_tool_func
        )
        
        grade_docs_tool = Tool(
            name="grade_documents_tool",
            description="Filtra i documenti per pertinenza, scartando quelli non rilevanti.",
            func=grade_docs_tool_func
        )
        
        grade_answer_tool_instance = Tool(
            name="grade_answer_tool",
            description="Valuta la qualità di una risposta finale. Controlla se è basata sui fatti e se risponde effettivamente alla domanda.",
            func=grade_answer_tool_func
        )
        
        tools = [
            hybrid_search_tool, 
            structured_query_tool, 
            web_search_tool,
            rewrite_tool,
            grade_docs_tool,
            grade_answer_tool_instance
        ]

        # 2. Define the agent node
        # This node invokes the LLM, which will decide whether to respond or call a tool.
        llm_with_tools = self.llm.bind_tools(tools)
        def agent_node(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        # 3. Define the ToolNode
        # This node executes the tools called by the agent.
        tool_node = ToolNode(tools)

        # 4. Add nodes to the workflow
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        # 5. Define edges
        workflow.set_entry_point("agent")

        # Add conditional edge: after the 'agent' node,
        # check if tools were called.
        # If yes, go to 'tool_node'. Otherwise, end (END).
        workflow.add_conditional_edges(
            "agent",
            tools_condition, # Predefined function from LangGraph
            {
                "tools": "tools",
                END: END
            }
        )

        # After tool execution, always return to the 'agent' node for evaluation.
        workflow.add_edge("tools", "agent")

        return workflow.compile()
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        try:
            # Build and compile workflow
            app = self.build_workflow()
            
            # Create initial state with human message
            initial_state = {"messages": [HumanMessage(content=question)]}
            
            # Run the workflow
            final_state = app.invoke(initial_state)
            
            # Extract the final answer from the last message
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return last_message['content']
            
            return "Mi dispiace, non sono riuscito a generare una risposta."
            
        except Exception as e:
            logger.error(f"Error in chatbot: {str(e)}")
            return "Mi dispiace, si è verificato un errore nel processare la tua domanda."
    
    def close(self):
        """Close connections."""
        # Neo4jGraph handles connection cleanup automatically
        pass

def main():
    """Main function to run the chatbot."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import OPENAI_API_KEY
    
    # Initialize chatbot
    chatbot = GraphRAGChatbot(openai_api_key=OPENAI_API_KEY)
    
    try:
        print("GraphRAG Chatbot - Digita 'esci' per uscire")
        print("-" * 50)
        
        while True:
            question = input("\nLa tua domanda: ").strip()
            if question.lower() in ['esci', 'exit', 'quit']:
                break
            
            if question:
                answer = chatbot.ask(question)
                print(f"\nRisposta: {answer}")
            else:
                print("Per favore, inserisci una domanda.")
                
    except KeyboardInterrupt:
        print("\nArrivederci!")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print("Si è verificato un errore. Controlla i log per i dettagli.")
    finally:
        chatbot.close()

if __name__ == "__main__":
    main()
