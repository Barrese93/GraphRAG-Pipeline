"""Schema definitions for the GraphRAG knowledge graph using LangChain."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Entity types that the LLM should look for
ENTITY_TYPES = [
    'Provvedimento',           # Main legal document/provision
    'FonteNormativa',          # Legal sources (laws, regulations, GDPR, etc.)
    'Persona',                 # People mentioned in documents
    'Ruolo',                   # Roles/positions of people
    'ConcettoGiuridico',       # Legal concepts and principles
    'Azienda',                 # Companies and organizations
    'Luogo',                   # Geographic locations
    'AutoritaGarante'          # Regulatory authorities (especially privacy authority)
]

# Relationship types that the LLM should look for
RELATIONSHIP_TYPES = [
    'CITA',                    # Document cites a legal source
    'RIFERIMENTA',             # Document refers to another document/concept
    'FIRMA',                   # Person signs/approves the document
    'HA_RUOLO',                # Person has a specific role
    'MENZIONA',                # Document mentions a concept/company/location
    'EMESSO_DA'                # Document is issued by an authority
]

# Document types for classification
DOCUMENT_TYPES = [
    'Deliberazione',           # Deliberation
    'Parere',                  # Opinion
    'Provvedimento',           # Provision/Decision
    'Opinion'                  # English opinion
]

class Entity(BaseModel):
    """Entity extracted from the document."""
    id: str = Field(description="Unique identifier for the entity")
    type: str = Field(description=f"Type of entity. Must be one of: {', '.join(ENTITY_TYPES)}")
    name: str = Field(description="Name or title of the entity")
    description: Optional[str] = Field(default="", description="Additional details about the entity")
    publication_date: Optional[str] = Field(default=None, description="Publication date of the document in YYYY-MM-DD format")
    document_type: Optional[str] = Field(default=None, description=f"Specific type of the document, if applicable. Must be one of: {', '.join(DOCUMENT_TYPES)}")
    reference_number: Optional[str] = Field(default=None, description="Official reference number of the document, e.g., 'n. 370 del 20 giugno 2024'")

class Relationship(BaseModel):
    """Relationship between two entities."""
    id: str = Field(description="Unique identifier for the relationship")
    source_entity_id: str = Field(description="ID of the source entity")
    target_entity_id: str = Field(description="ID of the target entity")
    type: str = Field(description=f"Type of relationship. Must be one of: {', '.join(RELATIONSHIP_TYPES)}")
    description: Optional[str] = Field(default="", description="Additional details about the relationship")

class KnowledgeGraph(BaseModel):
    """Knowledge graph extracted from the document."""
    entities: List[Entity] = Field(description="List of entities extracted from the document")
    relationships: List[Relationship] = Field(description="List of relationships between entities")

def get_extraction_prompt() -> ChatPromptTemplate:
    """Return the schema guidance prompt for the LLM using LangChain's ChatPromptTemplate."""
    return ChatPromptTemplate.from_messages([
        ("system", """Sei un esperto analista legale specializzato in privacy. Analizza il testo fornito e estrai un grafo di conoscenza strutturato.

SCHEMA DESIDERATO:
- Nodi (Entità): {entity_types} con le proprietà: id, type, name, description, publication_date (formato YYYY-MM-DD), document_type, reference_number.
- Relazioni: {relationship_types}

ISTRUZIONI DETTAGLIATE:
1.  **Entità Principale**: Identifica il documento principale descritto nel testo. Crea un'entità per esso (es. type='Provvedimento', 'Parere').
    *   **name**: Usa il titolo completo del documento.
    *   **document_type**: Classifica il documento usando uno dei seguenti: {document_types}.
    *   **publication_date**: Estrai la data di pubblicazione e normalizzala nel formato YYYY-MM-DD.
    *   **reference_number**: Estrai qualsiasi numero di registro o protocollo (es. 'n. 370 del 20 giugno 2024').
2.  **Altre Entità**: Estrai tutte le altre entità menzionate:
    *   'FonteNormativa': Leggi, regolamenti, direttive (es. 'GDPR', 'Codice dei contratti pubblici').
    *   'Persona': Persone fisiche (es. 'Pasquale Stanzione').
    *   'Ruolo': Ruoli o cariche (es. 'Presidente', 'Relatore').
    *   'AutoritaGarante': Enti e autorità (es. 'Garante per la protezione dei dati personali').
3.  **Relazioni**: Collega le entità con le seguenti relazioni:
    *   `EMESSO_DA`: Collega il documento principale alla sua 'AutoritaGarante'.
    *   `CITA`: Collega il documento principale alle 'FonteNormativa' che cita.
    *   `HA_RUOLO`: Collega una 'Persona' al suo 'Ruolo'.
    *   `FIRMA` o `MENZIONA`: Collega il documento principale alle 'Persone' rilevanti.

Restituisci il risultato in formato JSON seguendo lo schema Pydantic fornito."""),
        ("human", "{text}")
    ]).partial(
        entity_types=", ".join(ENTITY_TYPES),
        relationship_types=", ".join(RELATIONSHIP_TYPES),
        document_types=", ".join(DOCUMENT_TYPES)
    )

def get_system_prompt(document_type: str = "documento giuridico") -> str:
    """Return the base system prompt for the LLM (legacy compatibility)."""
    return f"""Sei un esperto analista legale specializzato in privacy. Analizza il testo seguente e estrai un grafo di conoscenza.

SCHEMA DESIDERATO:
- Nodi (Entità): {', '.join(ENTITY_TYPES)}
- Relazioni: {', '.join(RELATIONSHIP_TYPES)}

ISTRUZIONI:
1. Identifica l'entità principale 'Provvedimento' e i suoi attributi chiave (data, numero, oggetto).
2. Estrai tutte le 'FontiNormative' menzionate (es. 'GDPR', 'Codice dei contratti pubblici') e collegale al 'Provvedimento' con la relazione 'CITA'.
3. Estrai tutte le 'Persone' e i loro 'Ruoli', collegandoli tra loro e al 'Provvedimento'.
4. Estrai qualsiasi 'Azienda', 'Luogo' o 'ConcettoGiuridico' rilevante e collegalo con 'MENZIONA'.
5. Identifica l''AutoritaGarante' che ha emesso l'atto e collegala con 'EMESSO_DA'.
"""
