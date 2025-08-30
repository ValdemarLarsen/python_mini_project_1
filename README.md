# Data Ingestion and Wrangling Project

This project demonstrates data loading, cleaning, transformation, and visualization using Python with datasets in multiple formats. This project is mainly focused on getting key data out of the dataset.

## Project Structure

```
project/
├── data/
│   ├── employees.xlsx    # Employee data (25 records)
│   ├── products.json     # Product data (25 records)  
│   └── sales.csv         # Sales data (15,000+ records)
├── src/
│   └── loaders.py       # Data loading functions
├── main.py              # Main analysis script
├── requirements.txt     # Dependencies
└── README.md           
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python main.py
```

## Data Sources

**Note:** All data is AI-generated for educational purposes using realistic Danish business patterns.

### employees.xlsx
- 25 employees with Danish names
- Departments: Sales, Marketing, IT, HR, Finance, Operations
- Includes salary, hire date, contact info

### products.json  
- 25 products including Danish brands (LEGO, Bang & Olufsen, GANNI)
- Categories: Electronics, Home, Fashion, Gaming, Books, Toys
- Price range: 399 - 24,999 DKK

### sales.csv
- 15,000+ sales transactions over 18 months (Jan 2023 - Jun 2024)
- Includes seasonal patterns and realistic payment methods
- Total revenue: ~16 million DKK

## Features

The project includes:
- Data loading from CSV, Excel, and JSON formats
- Data cleaning and validation
- Data merging and transformation
- Statistical analysis
- 4 visualizations (revenue trends, category performance, etc.)
- Data anonymization
- Automated report generation

## Output

Running `python main.py` generates:
- `data_analysis_dashboard.png` - 4-panel visualization dashboard
- `data_analysis_report.txt` - Analysis summary report

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- openpyxl (for Excel files)

See `requirements.txt` for complete list.
