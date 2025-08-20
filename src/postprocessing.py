# src/postprocessing.py
import re

def clean_entity_value(value: str) -> str:
    """Clean entity value by removing special tokens and normalizing"""
    # Remove special tokens
    value = re.sub(r'\[PAD\]|\[CLS\]|\[SEP\]', '', value)
    # Normalize whitespace
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def map_synonyms(entities: dict) -> dict:
    mapping = {
        'total': ['total', 'amount', 'amt', 'grand_total'],
        'date': ['date', 'invoice_date', 'bill_date', 'purchased_on'],
        'vendor': ['vendor', 'merchant', 'store', 'shop', 'business'],
        'tax': ['tax', 'gst', 'vat'],
        'item': ['item', 'product', 'description', 'name'],
        'price': ['price', 'cost', 'amount', 'value', 'unit_price'],
        'quantity': ['quantity', 'qty', 'count']
    }
    
    standardized = {}
    used = set()
    
    for key, value in entities.items():
        cleaned_key = re.sub(r'\s+', '', key.lower())
        for std_key, aliases in mapping.items():
            if any(alias.lower() in cleaned_key for alias in aliases):
                standardized[std_key] = value
                used.add(key)
                break

    # Second pass: handle item-price pairs (common in receipts)
    items = {}
    prices = {}
    
    for key, value in entities.items():
        if 'item' in key.lower() or 'product' in key.lower():
            items[key] = value
        elif 'price' in key.lower() or 'amount' in key.lower():
            prices[key] = value

    # Pair items with prices if found
    if items and prices:
        for i, (item_key, item_val) in enumerate(items.items()):
            price_key = list(prices.keys())[min(i, len(prices)-1)]
            price_val = prices[price_key]
            standardized[f"item_{i+1}"] = f"{item_val} @ {price_val}"

    # Add unmapped ones
    for key, value in entities.items():
        if key not in used and clean_entity_value(value):
            standardized[key.lower()] = value
    
    return standardized

def postprocess_kv_pairs(entities: dict) -> dict:
    """Main postprocessing function with comprehensive cleaning"""
    # Filter out empty or special token values
    filtered_entities = {
        k: v for k, v in entities.items() 
        if v.strip() and not re.match(r'^\[\w+\]$', v)
    }
    
    # Clean and standardize
    cleaned = {k: clean_entity_value(v) for k, v in filtered_entities.items()}
    
    # Map to standard keys
    return map_synonyms(cleaned)