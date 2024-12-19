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
        name_regex = '[A-Z][a-z]+'
        word_regex = '[A-Za-z]+'
        word_dot_regex = f'{word_regex}\.?'
        self.pattern = (
            f'^(?:\d+ {name_regex}|' # a digit followed by a correct grammatical word
            f'\d+ \[{word_dot_regex}(?: {word_dot_regex})*\] {word_regex}|' # a digit followed by an abbreviation
            f'\d+ \({name_regex}\) {word_regex})' # a digit followed by a name in parenthesis and then a word
        )
        # = (
        #     r'^(?:\d+ [A-Z][a-z]+|'         # a digit followed by a correct grammatical word
        #     r'\d+ \[[A-Za-z]+\.(?: [A-Za-z]+\.)*\] [A-Za-z]+|'
        #     r'\d+ \([A-Za-z]+\) [A-Za-z]+)'
        # )

    def _clean_segment(self, segment_lines):
        return ''.join(segment_lines).strip().replace('\x02', '').replace('\r\n', '\n')

    def parse_entries(self, pages_data):
        output = []
        current_segment = []
        previous_number = None
        segment_text =''

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
                        else:
                            current_segment.append(line)  # Append non-sequential line to current segment
                            continue
                    output.append(segment_text)
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

        word_regex = '[A-Za-z]+'
        word_dot_regex = f'{word_regex}\.?'
        optional_bracket = f'(?:\({word_dot_regex}(?:\s{word_dot_regex})*\)\s)?'

        ending_sequence = f'{optional_bracket}(?:\w|(?:\w+\.?,?\s)+\w+\.?)\s\d+.*?(?:–|.\s?$)'
        date_pattern = f'(?:\d{{1,2}}(?:\.|\sgrossos)\s{month}\s\d{{2,4}}(?:\s\[\d\d\d\d\])?|\d\d/\d\d|\[sine\sdat\.\]|\[dat\.\sdeest\])'
        optional_secondary_date_pattern = f'(?:\s\(exped\.\s{date_pattern}\))?'

        date_missing = "\[dat. deest\]|\[sine dat.\]"

        dioc_string = '(?:dioc|commiss)\.\??'
        dioc_pattern = f'(?:\[{dioc_string}\]|{dioc_string})\s(?:vac.\sp.\so.\s)?(\d\d/\d\d)\s({ending_sequence})'

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

        for i, (header, sublemmas) in enumerate(zip(headers, sublemmas_list)):
            if sublemmas == '' and i != 0:
                # print("Empty found!")
                # print("header", header)
                # print("sublemmas", sublemmas)
                # break
                # print("No sublemma found")
                final_list.append({"header": header, "sublemmas": [{'text': f"{i} ", 'date': ""}]})
                # print(final_list[-1])
                continue
            splits, matches_found = split_by_ending_sequence(sublemmas, pattern)
            if matches_found:
                # print(f"Splits found in {i}:")
                # for split in splits:
                #     print(split)
                final_list.append({"header": header, "sublemmas": splits})
            elif i not in known_exceptions:
                exceptions.append(i)
            else:
                print(f"No matches in {i}, but it's a known exception.")
        
        return final_list, exceptions


########################################
# Data Processing and Export
########################################

class DataExporter:
    # Mapping of month abbreviations to their numerical representations
    month_mapping = {
        "ian.": "01", "febr.": "02", "mart.": "03", "apr.": "04", "mai.": "05",
        "iun.": "06", "iul.": "07", "aug.": "08", "sept.": "09", "oct.": "10",
        "nov.": "11", "decb.": "12"
    }

    # Regular expression pattern to match date formats
    date_pattern = r'(?:\d{1,2}(?:\.|\sgrossos)\s(?:ian\.|febr\.|mart\.|apr\.|mai\.|iun\.|iul\.|aug\.|sept\.|oct\.|nov\.|decb\.)\s\d{2,4}(?:\s\[\d\d\d\d\])?|\d\d/\d\d|\[sine\sdat\.\]|\[dat\.\sdeest\])'

    @staticmethod
    def clean_text(text):
        """
        Cleans the input text by removing newlines and trimming whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        return text.replace("\n", " ").strip()

    @staticmethod
    def make_identifier(band, lemma_number, index):
        """
        Creates unique identifiers for entries and subentries.

        Args:
            band (str): The band identifier.
            lemma_number (str): The lemma number as a zero-padded string.
            index (int): The index of the subentry.

        Returns:
            tuple: A tuple containing 'id_RG_all' and 'id_RG'.
        """
        id_RG = f"1{band}{lemma_number}"
        return f"{id_RG}-{index}", id_RG

    @classmethod
    def parse_date(cls, raw_date):
        """
        Parses the raw date string and formats it into 'YYYY-MM-DD' or similar formats.

        Args:
            raw_date (str): The raw date string to parse.

        Returns:
            str: The parsed and formatted date, or "Invalid Date" if parsing fails.
        """
        match = re.search(cls.date_pattern, raw_date)
        if match:
            raw_date = match.group()
            parts = raw_date.split()
            if len(parts) == 3:  # Format: DD. month YY
                day = parts[0].replace(".", "")
                month_abbr = parts[1]
                month = cls.month_mapping.get(month_abbr, "00")
                year = parts[2]
                if len(year) == 2:
                    year = f"14{year}"  # Assume century 1400
                return f"{year}-{month}-{day.zfill(2)}"
            elif "/" in raw_date:  # Handle MM/DD format
                month, day = raw_date.split("/")
                return f"14{month.zfill(2)}/14{day.zfill(2)}"  # Placeholder for unknown year
        return "Invalid Date"  # Return a default value for invalid dates

    @staticmethod
    def export_to_csv(data, output_file, band="10"):
        """
        Exports the processed data to a CSV file with structured columns.

        Args:
            data (list): A list of dictionaries, each containing 'header' and 'sublemmas'.
            output_file (str): The path to the output CSV file.
            band (str, optional): The band identifier. Defaults to "10".
        """
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_ALL)
            # Write the CSV header
            writer.writerow([
                "id_RG_all", "id_RG", "volume", "nr_RG", "url_RG", 
                "header_no_tags", "raw_date", "parsed_date", 
                "nr_suffix", "sublemma_no_tags"
            ])
            
            for entry in data:
                # Clean the header text
                cleaned_header = DataExporter.clean_text(entry.get("header", ""))
                index = 0  # Initialize index for sublemmas

                # Extract and format the lemma number as a 5-digit string
                lemma_number_match = re.match(r"\d+", cleaned_header)
                lemma_int = int(lemma_number_match.group())
                lemma_number = f"{lemma_int:05}"

                # Create identifiers
                id_RG_all, id_RG = DataExporter.make_identifier(band, lemma_number, index)

                # Construct URL
                url_RG = f"http://rg-online.dhi-roma.it/RG/{band}/{lemma_int}"

                # Remove the lemma number from the header
                cleaned_header_without_number = re.sub(r"^\d+\s*", "", cleaned_header)

                # Write the header row with blank sublemma and date
                writer.writerow([
                    id_RG_all, id_RG, band, lemma_int, url_RG, 
                    cleaned_header_without_number, "", "", index, ""
                ])

                # Process each sublemma
                for sublemma in entry.get("sublemmas", []):
                    index += 1  # Increment index for each sublemma
                    cleaned_sublemma_text = DataExporter.clean_text(sublemma.get("text", ""))
                    cleaned_date = DataExporter.clean_text(sublemma.get("date", ""))
                    parsed_date = DataExporter.parse_date(cleaned_date)
                    
                    # Create identifiers for sublemmas
                    id_RG_all_sub, id_RG_sub = DataExporter.make_identifier(band, lemma_number, index)
                    
                    # Write the sublemma row
                    writer.writerow([
                        id_RG_all_sub, id_RG_sub, band, lemma_int, url_RG, 
                        "", cleaned_date, parsed_date, index, cleaned_sublemma_text
                    ])
        
        print(f"CSV file '{output_file}' created successfully.")


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

    print("+"*100)
    print(entries[3:9])

    # # Step 3: Split into headers and sublemmas
    header_parser = HeaderSublemmaParser()
    headers, sublemmas_list, split_exceptions = header_parser.split_header_sublemmas(entries)

    # Date extraction (if needed)
    final_list, exceptions = header_parser.extract_dates(sublemmas_list)
    
    # print("+"*100)
    # print(final_list[3:9])

    # Step 4: Save or process data (CSV)
    # Exporter (implement actual logic inside the exporter)
    DataExporter.export_to_csv(final_list, "band_10_export_same_test.csv")

    # Step 5: Setup VectorDB for unknown and known documents
    # db_manager = VectorDBManager(db_path="rg_vectordb")
    # unknown_collection = db_manager.create_collection("rg_x_collection")

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
