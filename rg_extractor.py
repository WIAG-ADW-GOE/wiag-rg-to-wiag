import os
import re
import json
import csv
import requests
import pandas as pd
import traceback
from tqdm.notebook import tqdm
import pypdfium2 as pdfium
from datetime import datetime


########################################
# PDF Processing and Parsing
########################################

class PDFExtractor:
    def __init__(self, pdf_path, start_page=100, white_padding=40):
        self.pdf: pdfium.PdfDocument = pdfium.PdfDocument(pdf_path)
        self.start_page = start_page
        self.white_padding = white_padding
        left, bottom, right, top = self.pdf[self.start_page].get_bbox()
        self.bbox = (left, bottom, right, top)
        
    def extract_text_by_page(self):
        data = []
        left, bottom, right, top = self.bbox
        for i in range(self.start_page, len(self.pdf)):
            textpage = self.pdf[i].get_textpage()
            text_all = textpage.get_text_bounded(
                left=left, 
                bottom=bottom + self.white_padding + 10, 
                right=right, 
                top=top - self.white_padding
            )
            data.append(text_all)
        return data


class RegistParser:
    def __init__(self, skip_indices=None):
        if skip_indices is None:
            skip_indices = {2522}
        self.skip_indices = skip_indices
        self.pattern = (
            r'^(?:\d+ [A-Z][a-z]+|'         # a digit followed by a correct grammatical word
            r'\d+ \[[A-Za-z]+\.(?: [A-Za-z]+\.)*\] [A-Za-z]+|'
            r'\d+ \([A-Za-z]+\) [A-Za-z]+)'
        )

    def _clean_segment(self, segment_lines):
        return ''.join(segment_lines).strip().replace('\x02', '').replace('\r\n', '\n')

    def parse_entries(self, pages_data):
        output = []
        current_segment = []
        previous_number = None

        for data_i, text in enumerate(pages_data):
            lines = text.splitlines(keepends=True)
            for line in lines:
                if re.match(self.pattern, line):
                    if current_segment:
                        segment_text = self._clean_segment(current_segment)
                        current_number = int(segment_text.split()[0])
                        if previous_number is None or current_number == previous_number + 1:
                            if current_number + 1 in self.skip_indices:
                                current_number += 1
                            previous_number = current_number
                            output.append(segment_text)
                        else:
                            # Non-sequential number found; handle if needed
                            pass
                    current_segment = [line]
                else:
                    current_segment.append(line)

            if current_segment:
                segment_text = self._clean_segment(current_segment)
                current_number = int(segment_text.split()[0])
                if previous_number is None or current_number == previous_number + 1:
                    output.append(segment_text)

        return output


class HeaderSublemmaParser:
    def __init__(self):
        pass

    def split_header_sublemmas(self, entries):
        def split_outside_brackets(s):
            in_brackets = 0
            for idx, char in enumerate(s):
                if char == '[':
                    in_brackets += 1
                elif char == ']':
                    if in_brackets > 0:
                        in_brackets -= 1
                elif char == ':' and in_brackets == 0:
                    return s[:idx], s[idx+1:]
            return s, ''

        headers = []
        sublemmas_list = []
        split_exceptions = []

        for i, regist in enumerate(entries):
            header, sublemmas = split_outside_brackets(regist)
            headers.append(header)
            sublemmas_list.append(sublemmas)
            if sublemmas == '' and i != 0:
                split_exceptions.append(i)
        return headers, sublemmas_list, split_exceptions

    def extract_dates(self, sublemmas_list, known_exceptions=None):
        if known_exceptions is None:
            known_exceptions = set()

        month = '(?:' + '\.|'.join([
            "ian", "febr", "mart", "apr", "mai", "iun", "iul", "aug", "sept", "oct", "nov", "decb"
        ]) + '\.)'
        date_pattern = f'(?:\d{{1,2}}(?:\.|\sgrossos)\s{month}\s\d{{2,4}}(?:\s\[\d\d\d\d\])?|\d\d/\d\d|\[sine\sdat.\]|\[dat.\sdeest\])'
        optional_secondary_date_pattern = f'(?:\s\(exped\.\s{date_pattern}\))?'
        optional_bracket = f'(?:\([A-Za-z]+\.(?:\s[A-Za-z]+\.?)*\)\s)?'
        ending_sequence = f'{optional_bracket}(?:\w|(?:\w+\.?,?\s)+\w+\.?)\s\d+.*?(?:–|.\s?$)'
        pattern = f'({date_pattern}){optional_secondary_date_pattern}\s({ending_sequence})'

        exceptions = []
        final_list = []

        def split_by_ending_sequence(regist, pattern):
            splits = []
            last_end = 0
            matches_found = False

            for match in re.finditer(pattern, regist, re.DOTALL):
                matches_found = True
                date = match.group(1)
                start, end = match.span()
                text_before = regist[last_end:end].strip()
                if text_before:
                    splits.append({'text': text_before, 'date': date})
                last_end = end
            text_after = regist[last_end:].strip()
            if text_after:
                splits.append({'text': text_after})
            return splits, matches_found

        for i, sublemmas in enumerate(sublemmas_list):
            splits, matches_found = split_by_ending_sequence(sublemmas, pattern)
            if matches_found:
                final_list.append({"header_idx": i, "sublemmas": splits})
            elif i not in known_exceptions:
                exceptions.append(i)
        return final_list, exceptions


########################################
# Data Processing and Export
########################################

class DataExporter:
    @staticmethod
    def export_to_csv(data, output_file):
        # data is expected to be a list of dicts with 'header', 'sublemmas', etc.
        # This is a placeholder implementation. Adjust fields as needed.
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_ALL)
            # example header
            writer.writerow(["id_RG_all", "id_RG", "volume", "nr_RG", "url_RG", 
                             "header_no_tags", "date_sublemma", "date_sublemma_norm", 
                             "nr_suffix", "sublemma_no_tags"])
            # Implement actual logic as per your final_list structure
            # ...


########################################
# Vector Database Integration
########################################

import chromadb
from chromadb.utils import embedding_functions

class VectorDBManager:
    def __init__(self, db_path="rg_vectordb"):
        self.client = chromadb.PersistentClient(path=db_path)

    def create_collection(self, collection_name):
        return self.client.get_or_create_collection(name=collection_name)

    def insert_documents(self, collection, documents, metadatas, ids):
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_documents(self, collection, query_text, n_results=10):
        return collection.query(query_texts=[query_text], n_results=n_results)


########################################
# LLM Query and Integration
########################################

class LLMIntegrator:
    def __init__(self, llm_client, model):
        self.llm_client = llm_client
        self.model = model

    def build_query(self, unknown_doc, known_docs_collection, n_examples=4):
        results = known_docs_collection.query(query_texts=[unknown_doc], n_results=n_examples)
        qa_docs = ""
        empty_json = {
            'persons': [
                {
                    'givenname': '', 'prefix': '', 'familyname': '',
                    'offices': []
                }
            ]
        }
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Reconstruct JSON if needed
        def reconstruct_original_document(flattened_data):
            if not flattened_data:
                return {"persons": []}
            if "offices" in flattened_data:
                flattened_data["offices"] = json.loads(flattened_data["offices"])
            return {"persons": [flattened_data]}

        for doc, md in zip(documents, metadatas):
            qa_docs += f"Q: {doc}\nA: {json.dumps(reconstruct_original_document(md))}\n"

        query_str = f"""
{qa_docs} Q: {unknown_doc} A: {json.dumps(empty_json)}
"""
        return query_str

    def information_extractor(self, unknown_doc, known_docs_collection, system_prompt=''):
        if not system_prompt:
            system_prompt = "You are a Latin text processor. Look at the examples and solve the final question in the json format provided. Only respond with the answer."

        query = self.build_query(unknown_doc, known_docs_collection)
        chat_completion = self.llm_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": query}],
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0
        )
        return chat_completion.choices[0].message.content.strip()


########################################
# Example Workflow Usage
########################################

if __name__ == "__main__":
    # Example usage (paths and endpoints need to be adapted)
    
    # Step 1: Extract PDF text
    pdf_extractor = PDFExtractor('rgx_text_bd1_mn-2.pdf', start_page=100)
    pages_data = pdf_extractor.extract_text_by_page()

    # pdf_extractor.pdf

    # Step 2: Parse entries
    parser = RegistParser(skip_indices={2522})
    entries = parser.parse_entries(pages_data)

    # # Step 3: Split into headers and sublemmas
    header_parser = HeaderSublemmaParser()
    headers, sublemmas_list, split_exceptions = header_parser.split_header_sublemmas(entries)

    # Date extraction (if needed)
    final_list, exceptions = header_parser.extract_dates(sublemmas_list)
    
    # Step 4: Save or process data (CSV)
    # Exporter (implement actual logic inside the exporter)
    # DataExporter.export_to_csv(...)

    # Step 5: Setup VectorDB for unknown and known documents
    db_manager = VectorDBManager(db_path="rg_vectordb")
    unknown_collection = db_manager.create_collection("rg_x_collection")

    # Insert unknown docs (just as example)
    # unknown_collection.upsert(documents=[...], ids=[...])

    # Similarly, create known_docs_collection and insert known docs:
    # known_docs_collection = db_manager.create_collection("known_rg_collection")
    # known_docs_collection.upsert(documents=[...], metadatas=[...], ids=[...])

    # Step 6: Query Building and LLM integration
    # llm_client: provide your LLM client here
    # integrator = LLMIntegrator(llm_client, model='meta-llama-3.1-70b-instruct')

    # unknown_doc_example = entries[500]  # For example
    # extracted_info = integrator.information_extractor(unknown_doc_example, known_docs_collection)
    # print(extracted_info)
