from typing import Dict, List, Optional, Any
import streamlit as st
import networkx as nx
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import faiss
from openai import AsyncOpenAI
from transformers import AutoTokenizer

import pickle
import random
import time
import os
import numpy as np
import re
import json

class Documentation(BaseModel):
    """Documentation of a code object."""

    name: str = Field(..., description="Name")
    identifier: str = Field(..., description="Identifier")
    documentation: str = Field(..., description="Detailed documentation")
    summary: str = Field(..., description="Short and concise summary of the documentation")

def seperate_docs(input_text):
    """
    Splits a concatenated documentation string (created by concatenate_docs) back into a list of individual documentations.
    
    Args:
        input_text (str): Concatenated documentation string with START/END markers.
        
    Returns:
        list: List of individual documentation strings.
    """
    docs = []
    current_doc = []
    in_doc = False
    for line in input_text.splitlines():
        if line.strip() == "START":
            in_doc = True
            current_doc = []
        elif line.strip() == "END":
            in_doc = False
            docs.append('\n'.join(current_doc).strip())
        elif in_doc:
            current_doc.append(line)
    return docs

def build_function_doc_lookup(function_documentations):
    """
    Build a lookup dictionary from function name to its identifier and documentation.
    Assumes each item in function_documentations has at least 'name', 'identifier', and 'documentation' fields.
    """
    lookup = {}
    for doc in function_documentations:
        # Try to get the name, identifier, and documentation fields
        # Adjust field names if needed based on actual structure
        identifier = doc.identifier
        if identifier:
            lookup[identifier] = doc
    return lookup

def load_documentation_from_file(load_from_file: str):
    """
    Load documentation from a file using seperate_docs and parse into Documentation objects.

    Args:
        load_from_file (str): Path to a file containing concatenated docs.

    Returns:
        list: List of Documentation objects.
    """
    with open(os.path.join(os.getcwd(), load_from_file), 'r', encoding='utf-8') as f:
        file_content = f.read()
    docs = seperate_docs(file_content)
    documentation = []
    for doc_text in docs:
        name = ""
        identifier = ""
        documentation_text = ""
        summary = ""
        doc_start = None
        summary_start = None
        lines = doc_text.split('\n')
        for idx, line in enumerate(lines):
            if line.startswith("Name:"):
                name = line[len("Name:"):].strip()
            elif line.startswith("Identifier:"):
                identifier = line[len("Identifier:"):].strip()
            elif line.startswith("Documentation:"):
                doc_start = idx + 1
            elif line.startswith("Summary:"):
                summary_start = idx + 1
                break
        if doc_start is not None and summary_start is not None:
            documentation_text = "\n".join(lines[doc_start:summary_start-1]).strip()
            summary = "\n".join(lines[summary_start:]).strip()
        documentation.append(Documentation(name=name, identifier=identifier, documentation=documentation_text, summary=summary))
    return documentation

def load_components_from_json(filename: str) -> Dict[str, Any]:
    """
    Load component documentation and analysis from a JSON file.
    
    Args:
        filename: Input JSON filename
        
    Returns:
        Dictionary containing the loaded component data
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[load_components_from_json] Successfully loaded component data from {filename}")
        print(f"[load_components_from_json] Metadata: {data.get('metadata', {})}")
        print(f"[load_components_from_json] Number of components: {len(data.get('components', []))}")
        
        return data
    except FileNotFoundError:
        print(f"[load_components_from_json] Error: File {filename} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"[load_components_from_json] Error: Invalid JSON in {filename}: {e}")
        return {}
    except Exception as e:
        print(f"[load_components_from_json] Error loading from {filename}: {e}")
        return {}

def get_hf_tokens(prompt: str) -> int:
    """
    Calculates the number of tokens in a prompt for Hugging Face models,
    including Microsoft Phi models.

    Args:
        prompt (str): The text prompt to tokenize.
        model_name_or_path (str): The exact name or path of the Hugging Face model.
                                  Examples: "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
                                  or the path to a local model directory.

    Returns:
        int: The number of tokens in the prompt.
    """

    # Tokenize the text. `encode` returns a list of token IDs.
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def format_context(context: List[str], max_tokens: int):
    summaries = []
    tokens = get_hf_tokens("---------------------\n".join(context))
    while tokens > max_tokens:
        print(f"Total tokens ({tokens}) exceed max ({max_tokens}), summarizing...")
        doc_to_summarize = context.pop()
        print("Summarizing function documentation:")
        match = re.search(r'identifier:\s*([^\n]+)', doc_to_summarize)
        identifier = match.group(1).strip() if match else None
        match = re.search(r'name:\s*([^\n]+)', doc_to_summarize)
        name = match.group(1).strip() if match else None
        if not identifier:
            context.insert(0, doc_to_summarize)
            continue
        summary = code_item_dict[identifier].summary
        summarized = f"Summary of function {name} \nID: {identifier} \n{summary}"
        summaries.append(summarized)
        tokens = get_hf_tokens("---------------------\n".join(context + summaries))
    return "\n---------------------\n".join(context + summaries)

def Documentation_to_string(doc: Documentation, includeSummary: bool) -> str:
    """
    Convert a Documentation object to a formatted string.
    
    Args:
        doc (Documentation): The Documentation object to convert.
        
    Returns:
        str: Formatted string representation of the documentation.
    """
    if includeSummary:
        return (
            f"Name: {doc.name}\n"
            f"Identifier: {doc.identifier}\n"
            f"Documentation:\n{doc.documentation}\n"
            f"Summary:\n{doc.summary}\n"
        )
    return (
        f"Name: {doc.name}\n"
        f"Identifier: {doc.identifier}\n"
        f"Documentation:\n{doc.documentation}\n"
    )

class LocalEmbeddings(Embeddings):
    """Custom embeddings wrapper for local sentence-transformers model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

class SemanticSearch:
    def __init__(self, strings):
        """Initialize with a list of strings"""
        self.strings = strings
        
        # Initialize local embeddings model
        self.embedding_model = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Get embedding dimension
        embedding_dim = len(self.embedding_model.embed_query("hello world"))
        self.embedding_dim = embedding_dim
        
        # Create FAISS index with inner product (cosine similarity)
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Create vector store
        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Create and add documents with normalized embeddings
        documents = [Document(page_content=text, metadata={"index": i}) 
                    for i, text in enumerate(strings)]
        self.all_embeddings = np.empty((0, embedding_dim), dtype=np.float32)

        self._add_documents_with_normalization(documents)

    def _add_documents_with_normalization(self, documents):
        """Add documents with normalized embeddings"""
        if not documents:
            return
            
        # Get embeddings from the model
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings to unit length (for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Get current index size for proper ID mapping
        current_size = self.vectorstore.index.ntotal
        
        # Add normalized embeddings to the index
        self.vectorstore.index.add(embeddings)

        self.all_embeddings = np.vstack([self.all_embeddings, embeddings])
        
        # Add documents to docstore and create mapping
        ids = [str(current_size + i) for i in range(len(documents))]
        self.vectorstore.index_to_docstore_id.update(
            {current_size + i: doc_id for i, doc_id in enumerate(ids)}
        )
        
        for doc_id, doc in zip(ids, documents):
            self.vectorstore.docstore.add({doc_id: doc})
    
    def search(self, query, top_k=10):
        """Search for similar documents"""
        if self.vectorstore.index.ntotal == 0:
            return []
            
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Ensure top_k doesn't exceed available documents
        top_k = min(top_k, self.vectorstore.index.ntotal)
        
        # Search the index
        scores, indices = self.vectorstore.index.search(query_embedding, top_k)
        
        # Convert to LangChain format
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i < 0:  # FAISS returns -1 for invalid indices
                continue
            doc_id = self.vectorstore.index_to_docstore_id[i]
            doc = self.vectorstore.docstore.search(doc_id)
            results.append((doc, float(score)))
            
        return results
   
    def save_vectorstore(self, filepath):
        """Save the vectorstore (FAISS index, docstore, and mapping) to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.vectorstore.index, filepath + ".faiss")
        
        # Save docstore and mapping
        with open(filepath + ".meta.pkl", "wb") as f:
            pickle.dump({
                "docstore": self.vectorstore.docstore,
                "index_to_docstore_id": self.vectorstore.index_to_docstore_id,
                "strings": self.strings,
                "all_embeddings": self.all_embeddings,
                "embedding_dim": self.embedding_dim,
                "model_name": self.embedding_model.model.get_sentence_embedding_dimension()
            }, f)

    @classmethod
    def load_vectorstore(cls, filepath, embedding_model=None):
        """Load the vectorstore from disk."""
        # Check if files exist
        if not os.path.exists(filepath + ".faiss") or not os.path.exists(filepath + ".meta.pkl"):
            raise FileNotFoundError(f"Vectorstore files not found at {filepath}")
            
        # Load FAISS index
        index = faiss.read_index(filepath + ".faiss")
        
        # Load docstore and mapping
        with open(filepath + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
            
        # Use provided embedding_model or default
        if embedding_model is None:
            embedding_model = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
            
        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.strings = meta["strings"]
        obj.embedding_model = embedding_model
        obj.embedding_dim = meta.get("embedding_dim", len(embedding_model.embed_query("test")))
        obj.vectorstore = FAISS(
            embedding_function=obj.embedding_model,
            index=index,
            docstore=meta["docstore"],
            index_to_docstore_id=meta["index_to_docstore_id"],
        )
        obj.all_embeddings = meta["all_embeddings"]
        return obj
    
    def add_documents(self, new_strings):
        """Add new documents to existing vectorstore"""
        documents = [Document(page_content=text, metadata={"index": len(self.strings) + i}) 
                    for i, text in enumerate(new_strings)]
        self.strings.extend(new_strings)
        self._add_documents_with_normalization(documents)
    
    def get_document_count(self):
        """Get the number of documents in the vectorstore"""
        return self.vectorstore.index.ntotal
    
    def similarity_matrix(self):
        """Compute similarity matrix between all documents"""
        if len(self.all_embeddings) == 0:
            return np.array([])
        return np.dot(self.all_embeddings, self.all_embeddings.T)

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str, temperature: float=0.7, top_p: float=1.0, max_tokens: int=512):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def ask_LLM(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

class subQuestions(BaseModel):
    questions: List[Tuple[str, int]] = Field(description="List of (QUESTION, COMPONENT ID) pairs")

class SubQuestionsClient:
    def __init__(self, base_url: str, api_key: str, model_name: str, temperature: float, 
                 top_p: float, max_tokens: int):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._messages_template = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": ""}
        ]

    def _get_system_prompt(self) -> str:
        return """Provide sub-questions in this exact format:
        {
            "questions": [
                ["question1", component_id1],
                ["question2", component_id2]
            ]
        }
        """

    async def get_sub_questions(self, question: str, components: str) -> Optional[subQuestions]:
        messages = self._messages_template.copy()
        prompt = (
            f"Given the main request: '{question}'\n"
            f"And the following components:\n"
            f"{components}\n"
            f"Generate a list of sub-questions related to the main request, where each sub-question is directed towards a specific component.\n"
            "Don't ask questions to components that aren't relevant to the main question.\n"
            "Only ask important questions.\n"
            f"Provide the sub-questions and their corresponding component IDs in the specified format."
            f"Only include the final JSON response, nothing else."
        )
        messages[1]["content"] = prompt
        print(messages)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )

        print(response)

        if response.choices and response.choices[0].message.content:
            return subQuestions.model_validate_json(
                response.choices[0].message.content
            )

        # response = input("enter response: ")
        # return subQuestions.model_validate_json(response)

def load(file_path: str = "configurations.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    global tokenizer, code_item_dict, base_url, api_key, model_name, temperature, top_p, max_output_tokens, compItems_SE, comp_docs, max_input_tokens, codebase_overview

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in '{file_path}': {e}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"An error occurred while reading '{file_path}': {e}")

    output_dir = data["output_dir"]
    code_repo_root = data["code_repo_root"]
    existing_docs_file = data["existing_docs_file"]
    base_url = data["base_url"]
    model_name = data["model_name"]
    api_key = data["api_key"]
    temperature = data["temperature"]
    top_p = data["top_p"]
    max_output_tokens = data["max_output_tokens"]
    max_tokens = data["max_tokens"]
    graph_file = os.path.join(output_dir, "code_graph.graphml")
    overview_file = os.path.join(output_dir, "codebase_overview.txt")
    doc_file = os.path.join(output_dir, "codeItem_documentations.txt")
    component_doc_file = os.path.join(output_dir, "component_documentation.json")
    codeItem_se_file = os.path.join(output_dir, "codeItem_searchEngine")
    component_se_file = os.path.join(output_dir, "component_searchEngine")
    codeItem_by_comp_dir = os.path.join(output_dir, "codeItems_by_component_searchEngine")
    max_input_tokens = int(max_tokens - max_output_tokens - 1000)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained("openai/" + model_name)
        except:
            # Fallback to a default tokenizer if the specified model fails
            st.warning(f"Failed to load tokenizer for model {model_name}. Using default tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

    G = nx.read_graphml(graph_file) 

    with open(overview_file, "r") as f:
        codebase_overview = f.read()
    
    code_item_documentations = load_documentation_from_file(doc_file)

    code_item_dict = build_function_doc_lookup(code_item_documentations)

    comp_json = load_components_from_json(component_doc_file)

    # Get all component documentation items
    comp_docs = []
    for component in comp_json.get('components', []):
        doc = component.get('documentation')
        if doc:  # Only add if documentation exists
            comp_docs.append(Documentation(
                name=doc.get('name', ''),
                identifier=str(doc.get('identifier', '')),
                documentation=doc.get('documentation', ''),
                summary=doc.get('summary', '')
            ))

    comp_docs = sorted(comp_docs, key=lambda doc: int(doc.identifier))

    codeItem_SE = SemanticSearch.load_vectorstore(codeItem_se_file)
    component_SE = SemanticSearch.load_vectorstore(component_se_file)

    compItems_SE = [[] for _ in range(len(comp_json.get('components', [])))]
    for idx in range(len(comp_json.get('components', []))):
        compItems_SE[idx] = SemanticSearch.load_vectorstore(os.path.join(codeItem_by_comp_dir, f"{idx}"))

load()

LLM = LLMClient(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_output_tokens
)

Q_client = SubQuestionsClient(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_output_tokens
)

# Streamed response emulator
async def generate_response(user_query: str):
    
    show_comps = ""
    for comp in comp_docs:
        show_comps += "Component ID: " + comp.identifier + "\n  Name: " + comp.name + "\n  Summary: " + comp.summary + "\n--------------------\n"

    subQuestions = await Q_client.get_sub_questions(user_query, show_comps)
    st.write(f"answering {len(subQuestions.questions)} sub-questions")

    context = []
    for subQ in subQuestions.questions:
        currentContext = []
        retrieved = compItems_SE[subQ[1]].search(subQ[0], top_k=10)
        for r in retrieved:
            currentContext.append(r[0].page_content)
        context.append(currentContext)
    
    context_str = []

    for idx, f_list in enumerate(context):
        comp_id = subQuestions.questions[idx][1]
        context_str.append(format_context(["Module description: \n" + Documentation_to_string(comp_docs[comp_id], False)] + f_list, max_input_tokens))

    subQ_Answers = []
    for idx, c in enumerate(context_str):
        prompt = (
            "Answer the following question based on the provided context:\n" +
            f"Question: {subQuestions.questions[idx][0]}\n" +
            f"Context: \n\n{c}\n"
        )
        response = await LLM.ask_LLM(prompt)
        subQ_Answers.append(response)
    
    prompt = (
        f"Answer the following question in-depth using the provided context, be as detailed as possible\n" +
        f"Question: {user_query}\n\n" +
        f"Codebase overview:\n{codebase_overview}\n" +
        f"\nFurther information: \n" +
        "\n----------------------\n".join(subQ_Answers) + "\n"
    )

    response = await LLM.ask_LLM(prompt)
    return response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
