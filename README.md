
# RAG for Historical Clerics

## Overview

This repository explores the creation of Retrieval-Augmented Generation (RAG) and other related Natural Language Processing (NLP) models to answer questions about historical clerics. It investigates methods to integrate databases with Large Language Models (LLMs) to enhance database accessibility and uncover interesting connections within the data.

**Target Audience:**  
Researchers interested in leveraging NLP and LLMs for historical data analysis. This repository serves as a source of inspiration and provides reusable classes that can be integrated into other frameworks.

## Features

- **RAG Model Development:** Implements a RAG model similar to GraphRAG for retrieving and generating responses based on historical data.
- **Document Parsing:** Extracts and processes text from the *Repertorium Germanicum* (RG) to facilitate information retrieval.
- **LLM Integration:** Tests the capabilities of various LLMs in extracting entities and creating schemas from historical documents.
- **Alternative Chat API:** Utilizes GWDG's ChatGPT API alternative for enhanced performance with Latin documents.
- **NLP Toolkits Comparison:** Evaluates smaller NLP toolkits against LLMs for entity extraction and schema creation.

## Repository Structure

All files are located in the root of the repository. There are no subdirectories.

### Jupyter Notebooks

- **`langchain.ipynb`**  
  Contains sample code to test the LangChain library.

- **`rag.ipynb`**  
  Implements a RAG model, similar to GraphRAG, for information retrieval and generation.

- **`RG_extraction.ipynb`**  
  Extracts documents from the *Repertorium Germanicum* and uses an LLM-based approach to extract office information from the registries contained within the document.

- **`llm_gen.ipynb`**  
  Tests LLM capabilities for entity extraction and schema creation.

- **`smaller_llms.ipynb`**  
  Evaluates smaller NLP toolkits for tasks similar to those performed in `llm_gen.ipynb`.

### Python Scripts

Python files are available and can be executed directly from the command line:

```bash
python <name_of_file>.py
```

### Data Files

- **`rgx_text_bd1_mn-2.pdf`**  
  The RG document in PDF format.

- **`WIAG-Domherren-DB.json`**  
  Export of the WIAG database.

- **`xml_file.xml`**  
  The RG document in XML format.

- **`rg9sublemma.csv`**  
  *Not currently used in the project.*

**Note:**  
The data files are not fully prepared for general use. Extraction of registries from the *Repertorium Germanicum* has been performed for a single chapter and is not generalizable.

## Installation

### Prerequisites

Ensure you have Python installed (preferably via Conda). The following Python libraries are required:

- `openai`
- `dotenv`
- `pandas`
- `numpy`
- `rdflib`
- `pyshacl`
- `chromadb`
- `langchain`
- `tqdm`
- `matplotlib`
- `nltk`
- `aiohttp`
- `sklearn`
- `spacy`

### Setup Instructions

It is recommended to use Conda for environment management, but `pip` can also be used.

1. **Using Conda:**
   ```bash
   conda create -n rag_env python=3.9
   conda activate rag_env
   conda install openai dotenv pandas numpy rdflib pyshacl chromadb langchain tqdm matplotlib nltk aiohttp sklearn spacy
   ```

2. **Using Pip:**
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   pip install -r requirements.txt
   ```

   *Create a `requirements.txt` file with the listed prerequisites for easier installation.*

   Example `requirements.txt`:
   ```
   openai
   dotenv
   pandas
   numpy
   rdflib
   pyshacl
   chromadb
   langchain
   tqdm
   matplotlib
   nltk
   aiohttp
   sklearn
   spacy
   ```

## Configuration

Before running the scripts or notebooks, create a `.env` file in the root directory with the following variables:

```env
API_KEY=your_api_key_here
ENDPOINT=your_api_endpoint_here
```

Replace `your_api_key_here` and `your_api_endpoint_here` with your actual API credentials.

## Usage

### Running Jupyter Notebooks

The notebooks can be executed cell by cell. Note that they currently lack comprehensive documentation, so familiarity with the code is beneficial.

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run the desired notebook.

### Running Python Scripts

Execute Python files directly from the command line:

```bash
python <name_of_file>.py
```

Ensure that the `.env` file is properly configured before running the scripts.

## Testing the Chat API Service

The included Jupyter notebooks allow you to test the chat API service. Ensure you have the necessary credentials set in the `.env` file. No additional dependencies are required beyond those listed in the installation section.

## License

This project is licensed under the [Apache License](LICENSE).

## Contact

For any questions or support, please contact:

- **Contributor:** [Your Name](mailto:your.email@example.com)
- **Department:** [Your Department Email](mailto:department.email@example.com)

## Acknowledgements

- [OpenAI](https://www.openai.com/) for their API services.
- [GWDG's API Service](https://www.gwdg.de/) for providing an alternative ChatGPT API.
