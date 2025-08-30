# Complete Python solution for loading, transforming, and visualizing data with UI so show the data in a simple way

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom loaders the functions we created in the src folder
from src.loaders import load_csv, load_excel, load_json

class DataProcessor:
    """Main class for handling data ingestion and wrangling operations"""
    
    def __init__(self):
        self.employees = None
        self.products = None
        self.sales = None
        self.merged_data = None
        
    def ingest_data(self):
        """Ingest data from all three sources"""
        print("Starting data ingestion process...")
        
        try:
            # Load CSV data (sales)
            print("Loading sales data (CSV)...")
            self.sales = load_csv('data/sales.csv')
            print(f"Loaded {len(self.sales)} sales records")
            
            # Load Excel data (employees)
            print("Loading employee data (Excel)...")
            self.employees = load_excel('data/employees.xlsx')
            print(f"Loaded {len(self.employees)} employee records")
            
            # Load JSON data (products)
            print("Loading product data (JSON)...")
            self.products = load_json('data/products.json')
            print(f"Loaded {len(self.products)} product records")
            
            print("Data ingestion completed successfully!\n")
            
        except Exception as e:
            print(f"Error during data ingestion: {e}")
            return False
            
        return True
    
    def explore_data(self):
        """Explore and display basic information about the datasets"""
        print("Data Exploration Summary")
        print("=" * 50)
        
        datasets = [
            ("Sales Data", self.sales),
            ("Employee Data", self.employees), 
            ("Product Data", self.products)
        ]
        
        for name, df in datasets:
            if df is not None:
                print(f"\n{name}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Data types:\n{df.dtypes}")
                print(f"   Missing values:\n{df.isnull().sum()}")
                print(f"   First few rows:\n{df.head()}")
                print("-" * 40)
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("Starting data cleaning process...")
        
        # Clean sales data
        if self.sales is not None:
            print("Cleaning sales data...")
            # Convert date columns
            if 'date' in self.sales.columns:
                self.sales['date'] = pd.to_datetime(self.sales['date'])
            
            # Handle missing values
            self.sales = self.sales.dropna()
            
            # Ensure numeric columns are proper type
            numeric_cols = ['quantity', 'unit_price', 'total_amount']
            for col in numeric_cols:
                if col in self.sales.columns:
                    self.sales[col] = pd.to_numeric(self.sales[col], errors='coerce')
        
        # Clean employee data
        if self.employees is not None:
            print("   Cleaning employee data...")
            # Handle missing values
            self.employees = self.employees.fillna('Unknown')
            
            # Standardize text data
            text_cols = ['first_name', 'last_name', 'department', 'position']
            for col in text_cols:
                if col in self.employees.columns:
                    self.employees[col] = self.employees[col].str.strip().str.title()
        
        # Clean product data
        if self.products is not None:
            print("   Cleaning product data...")
            # Handle missing values
            if 'price' in self.products.columns:
                self.products['price'] = pd.to_numeric(self.products['price'], errors='coerce')
            
            # Remove duplicates
            self.products = self.products.drop_duplicates()
        
        print("Data cleaning completed!\n")
    
    def transform_data(self):
        """Transform and merge data from different sources"""
        print("Starting data transformation...")
        
        # Create sample data if not available (for demonstration)
        if self.sales is None:
            self.sales = self._create_sample_sales_data()
        if self.employees is None:
            self.employees = self._create_sample_employee_data()
        if self.products is None:
            self.products = self._create_sample_product_data()
        
        # Merge datasets
        print("   Merging datasets...")
        
        # First merge sales with products
        merged = pd.merge(
            self.sales, 
            self.products, 
            left_on='product_id', 
            right_on='id', 
            how='left',
            suffixes=('_sale', '_product')
        )
        
        # Then merge with employees
        self.merged_data = pd.merge(
            merged,
            self.employees,
            left_on='employee_id',
            right_on='id',
            how='left',
            suffixes=('', '_emp')
        )
        
        # Create derived columns
        if 'quantity' in self.merged_data.columns and 'unit_price' in self.merged_data.columns:
            self.merged_data['revenue'] = self.merged_data['quantity'] * self.merged_data['unit_price']
        
        # Add time-based features
        if 'date' in self.merged_data.columns:
            self.merged_data['year'] = self.merged_data['date'].dt.year
            self.merged_data['month'] = self.merged_data['date'].dt.month
            self.merged_data['quarter'] = self.merged_data['date'].dt.quarter
        
        print(f"Data transformation completed! Final dataset shape: {self.merged_data.shape}\n")
    
    def _create_sample_sales_data(self):
        """Create sample sales data for demonstration"""
        np.random.seed(42)
        n_records = 1000
        
        data = {
            'id': range(1, n_records + 1),
            'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
            'product_id': np.random.randint(1, 21, n_records),
            'employee_id': np.random.randint(1, 11, n_records),
            'quantity': np.random.randint(1, 10, n_records),
            'unit_price': np.round(np.random.uniform(10, 500, n_records), 2)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_employee_data(self):
        """Create sample employee data for demonstration"""
        employees = [
            {'id': i, 'first_name': f'Employee{i}', 'last_name': f'Last{i}', 
             'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR']),
             'position': np.random.choice(['Junior', 'Senior', 'Manager', 'Director'])}
            for i in range(1, 11)
        ]
        
        return pd.DataFrame(employees)
    
    def _create_sample_product_data(self):
        """Create sample product data for demonstration"""
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        products = [
            {'id': i, 'name': f'Product {i}', 
             'category': np.random.choice(categories),
             'price': np.round(np.random.uniform(10, 500), 2)}
            for i in range(1, 21)
        ]
        
        return pd.DataFrame(products)
    
    def analyze_data(self):
        """Perform data analysis"""
        print("Data Analysis Results")
        print("=" * 50)
        
        if self.merged_data is not None:
            # Basic statistics
            print("\nRevenue Statistics:")
            if 'revenue' in self.merged_data.columns:
                print(f"   Total Revenue: ${self.merged_data['revenue'].sum():,.2f}")
                print(f"   Average Revenue per Sale: ${self.merged_data['revenue'].mean():.2f}")
                print(f"   Median Revenue per Sale: ${self.merged_data['revenue'].median():.2f}")
            
            # Top performing categories
            print("\nTop Product Categories by Revenue:")
            if 'category' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
                category_revenue = self.merged_data.groupby('category')['revenue'].sum().sort_values(ascending=False)
                for category, revenue in category_revenue.head().items():
                    print(f"   {category}: ${revenue:,.2f}")
            
            # Department performance
            print("\nDepartment Performance:")
            if 'department' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
                dept_performance = self.merged_data.groupby('department')['revenue'].agg(['sum', 'count', 'mean'])
                print(dept_performance)
        
        print("\n" + "=" * 50)
    
    def create_visualizations(self):
        """Create various visualizations"""
        print("Creating visualizations...")
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue over time
        if 'date' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
            monthly_revenue = self.merged_data.groupby(self.merged_data['date'].dt.to_period('M'))['revenue'].sum()
            monthly_revenue.plot(ax=axes[0,0], kind='line', color='blue')
            axes[0,0].set_title('Monthly Revenue Trend')
            axes[0,0].set_xlabel('Month')
            axes[0,0].set_ylabel('Revenue ($)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Revenue by category
        if 'category' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
            category_revenue = self.merged_data.groupby('category')['revenue'].sum().sort_values(ascending=True)
            category_revenue.plot(ax=axes[0,1], kind='barh', color='green')
            axes[0,1].set_title('Revenue by Product Category')
            axes[0,1].set_xlabel('Revenue ($)')
        
        # 3. Department performance
        if 'department' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
            dept_revenue = self.merged_data.groupby('department')['revenue'].sum()
            axes[1,0].pie(dept_revenue.values, labels=dept_revenue.index, autopct='%1.1f%%')
            axes[1,0].set_title('Revenue Distribution by Department')
        
        # 4. Quantity vs Revenue scatter
        if 'quantity' in self.merged_data.columns and 'revenue' in self.merged_data.columns:
            axes[1,1].scatter(self.merged_data['quantity'], self.merged_data['revenue'], alpha=0.6, color='red')
            axes[1,1].set_title('Quantity vs Revenue')
            axes[1,1].set_xlabel('Quantity')
            axes[1,1].set_ylabel('Revenue ($)')
        
        plt.tight_layout()
        plt.savefig('data_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations created and saved as 'data_analysis_dashboard.png'")
    
    def apply_anonymization(self):
        """Apply basic data anonymization techniques"""
        print("Applying data anonymization...")
        
        if self.merged_data is not None:
            # Create anonymized version
            anonymized_data = self.merged_data.copy()
            
            # Hash employee names
            if 'first_name' in anonymized_data.columns:
                anonymized_data['first_name'] = anonymized_data['first_name'].apply(
                    lambda x: f"Employee_{hash(x) % 10000}" if pd.notna(x) else x
                )
            
            if 'last_name' in anonymized_data.columns:
                anonymized_data['last_name'] = anonymized_data['last_name'].apply(
                    lambda x: f"Surname_{hash(x) % 10000}" if pd.notna(x) else x
                )
            
            # Generalize precise values
            if 'revenue' in anonymized_data.columns:
                anonymized_data['revenue_range'] = pd.cut(
                    anonymized_data['revenue'], 
                    bins=[0, 100, 500, 1000, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Very High']
                )
            
            self.anonymized_data = anonymized_data
            print("Data anonymization completed!")
            
            return anonymized_data
        
        return None
    
    def generate_report(self):
        """Generate a comprehensive report"""
        report = []
        report.append("# Data Ingestion and Wrangling Report")
        report.append("=" * 50)
        report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset overview
        report.append("## Dataset Overview")
        if self.merged_data is not None:
            report.append(f"- Total records: {len(self.merged_data):,}")
            report.append(f"- Total columns: {len(self.merged_data.columns)}")
            report.append(f"- Data sources combined: 3 (CSV, Excel, JSON)")
        
        # Key findings
        report.append("\n## Key Findings")
        if self.merged_data is not None and 'revenue' in self.merged_data.columns:
            total_revenue = self.merged_data['revenue'].sum()
            avg_revenue = self.merged_data['revenue'].mean()
            report.append(f"- Total revenue: ${total_revenue:,.2f}")
            report.append(f"- Average revenue per transaction: ${avg_revenue:.2f}")
            
            if 'category' in self.merged_data.columns:
                top_category = self.merged_data.groupby('category')['revenue'].sum().idxmax()
                report.append(f"- Top performing category: {top_category}")
        
        # Data quality
        report.append("\n## Data Quality Assessment")
        if self.merged_data is not None:
            missing_data = self.merged_data.isnull().sum().sum()
            report.append(f"- Total missing values: {missing_data}")
            report.append(f"- Data completeness: {((1 - missing_data / self.merged_data.size) * 100):.1f}%")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('data_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Analysis Report Generated")
        print("=" * 30)
        print(report_text)
        print("\nReport saved as 'data_analysis_report.txt'")

def main():
    """Main function to run the complete data pipeline"""
    print("Starting Data Ingestion and Wrangling Project")
    print("=" * 60)
    
    # Initialize the processor
    processor = DataProcessor()
    
    # Execute the complete pipeline
    try:
        # Step 1: Ingest data
        processor.ingest_data()
        
        # Step 2: Explore data
        processor.explore_data()
        
        # Step 3: Clean data
        processor.clean_data()
        
        # Step 4: Transform and merge data
        processor.transform_data()
        
        # Step 5: Analyze data
        processor.analyze_data()
        
        # Step 6: Create visualizations
        processor.create_visualizations()
        
        # Step 7: Apply anonymization
        processor.apply_anonymization()
        
        # Step 8: Generate report
        processor.generate_report()
        
        print("\nProject completed successfully!")
        print("Output files created:")
        print("   - data_analysis_dashboard.png")
        print("   - data_analysis_report.txt")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()