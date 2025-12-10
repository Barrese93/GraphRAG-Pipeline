"""Graders for document and answer evaluation in RAG pipeline."""

import logging
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Punteggio binario per la pertinenza dei documenti recuperati."""
    binary_score: str = Field(description="I documenti sono pertinenti alla domanda, 'yes' o 'no'")


class GradeHallucinations(BaseModel):
    """Punteggio binario sulla presenza di allucinazioni nella risposta."""
    binary_score: str = Field(description="La risposta è basata sui fatti, 'yes' o 'no'")


class GradeAnswerUsefulness(BaseModel):
    """Punteggio binario sull'utilità della risposta."""
    binary_score: str = Field(description="La risposta risolve la domanda, 'yes' o 'no'")


def create_relevance_grader(llm):
    """Create a document relevance grader chain."""
    system_prompt = """Sei un valutatore che giudica la pertinenza di un documento recuperato rispetto a una domanda dell'utente. L'obiettivo è scartare recuperi erronei. Se il documento contiene parole chiave o significato semantico relativo alla domanda, consideralo pertinente. Fornisci un punteggio binario 'yes' o 'no'."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Documento recuperato: \n\n {document} \n\n Domanda utente: {question}"),
    ])
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    return grade_prompt | structured_llm_grader


def create_hallucination_grader(llm):
    """Create a hallucination grader chain."""
    system_prompt = """Sei un valutatore che controlla se una risposta è basata sui fatti forniti nei documenti di contesto. Fornisci un punteggio binario 'yes' o 'no'. 'yes' significa che la risposta è basata sui fatti, 'no' significa che contiene allucinazioni."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Documenti di contesto: \n\n {documents} \n\n Risposta generata: {generation}"),
    ])
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    return hallucination_prompt | structured_llm_grader


def create_answer_usefulness_grader(llm):
    """Create an answer usefulness grader chain."""
    system_prompt = """Sei un valutatore che controlla se una risposta è utile per risolvere una domanda dell'utente. Fornisci un punteggio binario 'yes' o 'no'. 'yes' significa che la risposta risolve la domanda, 'no' significa che non è utile."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Domanda utente: {question} \n\n Risposta generata: {generation}"),
    ])
    structured_llm_grader = llm.with_structured_output(GradeAnswerUsefulness)
    return answer_prompt | structured_llm_grader
