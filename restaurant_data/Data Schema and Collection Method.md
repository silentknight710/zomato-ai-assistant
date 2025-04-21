
Project Documentation  
Title: Restaurant Data Scraper – Schema and Collection Methodology  
Purpose: To collect and structure metadata and menus of restaurants from Zomato for building a knowledge base or AI retrieval system.

---

1. Data Schema

The scraper collects and organizes the restaurant data into a well-structured JSON format. Each restaurant object contains the following fields:

Top-Level Fields

- id (string):  
  A URL-safe, slugified identifier derived from the restaurant name. Used to reference files consistently.
  
- name (string):  
  The full name of the restaurant.

- url (string):  
  The original Zomato URL from which the restaurant data was scraped.

- html_file (string):  
  Path to the locally saved HTML file of the restaurant’s webpage.

- menu_file (string):  
  Path to the expected JSON file containing the restaurant’s menu. Format: `menus/<restaurant_id>_menu.json`.

Nested Fields

- address (object):  
  Structured address information extracted from the JSON-LD block.
  - street (string or null): Street address  
  - locality (string or null): Local area or neighborhood  
  - region (string or null): State or region  
  - postalCode (string or null): ZIP or postal code  
  - country (string or null): Country name

- rating (object):  
  Rating data provided by the Zomato page’s structured data.
  - value (string or float): Average rating score  
  - count (int or null): Number of reviews

- cuisine (list of strings):  
  List of cuisine types served by the restaurant.

- price_range (string or null):  
  Indicative cost level (e.g., ₹₹, ₹₹₹, etc.).

- telephone (string or null):  
  Contact number listed for the restaurant.

Example Schema

```json
{
  "id": "lake-district-bar-kitchen",
  "name": "Lake District Bar & Kitchen",
  "url": "https://www.zomato.com/hyderabad/lake-district-bar-kitchen-necklace-road/info",
  "html_file": "restaurant_data/html/zomato.com_hyderabad_lake-district-bar-kitchen.html",
  "menu_file": "restaurant_data/menus/lake-district-bar-kitchen_menu.json",
  "address": {
    "street": "Necklace Road",
    "locality": "Hyderabad",
    "region": null,
    "postalCode": null,
    "country": "India"
  },
  "rating": {
    "value": "4.3",
    "count": 145
  },
  "cuisine": ["North Indian", "Continental"],
  "price_range": "₹₹₹",
  "telephone": "+91 9876543210"
}
```

---

2. Data Collection Methodology

Step 1: Initialization  
- The scraper initializes output folders:  
  - `restaurant_data/html/` for raw HTML  
  - `restaurant_data/menus/` for restaurant menu JSONs

Step 2: HTML Retrieval  
- For each provided Zomato URL:  
  - A GET request is sent with browser-like headers.  
  - Response HTML is saved locally with a unique filename.  
  - Errors in fetching are logged.

Step 3: Metadata Extraction  
- HTML is parsed using BeautifulSoup.  
- `<script type="application/ld+json">` tags are inspected.  
- JSON-LD blocks are parsed to extract structured metadata:  
  - Restaurant details  
  - Address, cuisine, rating, etc.  
- Slugified id is generated from the restaurant name.

Step 4: JSON File Writing  
- For each restaurant:  
  - Individual JSON file is created under `restaurant_data/`  
  - Includes a link to its HTML file and associated menu file.  
- All restaurant entries are also combined into a master file: `restaurant_data/total.json`

Step 5: Menu Files (Assumed)  
- The scraper anticipates the presence of separate menu files.  
- Expected path: `menus/<restaurant_id>_menu.json`  
- These are later merged into a unified structure as needed.

---

3. Assumptions and Limitations

- Assumes Zomato uses consistent structured data in the JSON-LD format.  
- Only the first valid `<script type="application/ld+json">` block with `"@type": "Restaurant"` is parsed.  
- Menus must be separately scraped or generated, then saved under the expected path.  
- Does not handle JavaScript-rendered content.  
- Schema relies on presence of metadata in structured form; missing fields may be null.
