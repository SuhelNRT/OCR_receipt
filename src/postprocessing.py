# src/postprocessing.py
import re

def clean_entity_value(value: str) -> str:
    value = re.sub(r'\[CLS\]|\[SEP\]', '', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip()

def map_synonyms(entities: dict) -> dict:
    synonym_map = {
        'total_amt': ['total', 'amount', 'grand_total'],
        'date': ['invoice_date', 'bill_date'],
        'vendor': ['merchant', 'shop', 'store']
    }
    
    standardized = {}
    for key, value in entities.items():
        for standard_key, aliases in synonym_map.items():
            if key in aliases:
                standardized[standard_key] = value
                break
        else:
            standardized[key] = value
    return standardized

def postprocess_kv_pairs(entities: dict) -> dict:
    cleaned = {k: clean_entity_value(v) for k, v in entities.items()}
    return map_synonyms(cleaned)