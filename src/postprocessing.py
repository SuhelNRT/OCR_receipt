# src/postprocessing.py
import re
from src.utils import clean_currency, clean_date, clean_text

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

# ---------------------------------------  NER Result organisation  -----------------

# Initialize structured output

def organise_ner_result(ner_results):

    structured_entities = {
                    "shop": {},
                    "items": [],
                    "invoice_id": "",
                    "total": "",
                    "tax": "",
                    "date": "",
                    "time": ""
                }


    item_fields = {"product_name", "product_price", "product_qty"}

    current_item = {}

    for entity in ner_results:
        entity_type = entity["entity_group"].lower()
        entity_value = clean_text(entity["word"].strip())
        confidence = entity["score"]
        
        # Skip low confidence or empty values
        if not entity_value or confidence < 0.1:
            continue
            
        # SHOP INFORMATION
        if entity_type == "shop_name":
            structured_entities["shop"]["name"] = entity_value
        elif entity_type == "shop_address":
            # Handle multiple address parts
            if "address" not in structured_entities["shop"]:
                structured_entities["shop"]["address"] = entity_value
            else:
                structured_entities["shop"]["address"] += " " + entity_value
        elif entity_type == "invoice_id":
            structured_entities["invoice_id"] = entity_value
        
        # ITEMS - SMART GROUPING
        elif entity_type in item_fields:
            if entity_type == "product_name":
                # Save current item if exists, then start new one
                if current_item:
                    structured_entities["items"].append(current_item)
                current_item = {"name": entity_value}
            elif entity_type == "product_price":
                current_item["price"] = clean_currency(entity_value)
            elif entity_type == "product_qty":
                current_item["quantity"] = entity_value
        
        # FINANCIALS
        elif entity_type == "total":
            # Might be just "$", so we'll fix later
            if not structured_entities["total"]:
                structured_entities["total"] = clean_currency(entity_value)
        elif entity_type == "tax":
            structured_entities["tax"] = clean_currency(entity_value)
        
        # DATE/TIME
        elif entity_type == "date":
            structured_entities["date"] = clean_date(entity_value)
        elif entity_type == "time":
            structured_entities["time"] = entity_value

    # Don't forget the last item!
    if current_item and current_item not in structured_entities["items"]:
        structured_entities["items"].append(current_item)

    # Final cleanup
    final_entities = {}
    for k, v in structured_entities.items():
        if v:  # Only include non-empty values
            if isinstance(v, list) and len(v) == 0:
                continue
            final_entities[k] = v
    
    return final_entities