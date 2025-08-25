# Azure Data Factory (ADF)

## Q1. What is Azure Data Factory, and why is it used in data engineering?

ADF is a cloud tool to move and transform data.  
You can create workflows (pipelines) that:  
- Ingest data (bring it in) from sources like SQL, APIs, files.  
- Transform data (clean, process) using Databricks/other tools.  
- Load data into storage like Azure Data Lake or Synapse.

## Q2. How do you create and schedule a pipeline in ADF?

**Steps:**  
- Open ADF in Azure portal.  
- Create a new pipeline in the visual editor.  
- Add activities (e.g., "Copy Data").  
- Define source (input) and sink (output).  
- Test and publish.  
- Use triggers to schedule runs (daily, hourly, etc.).

## Q3. Mapping Data Flow vs Wrangling Data Flow

- **Mapping Data Flow**: For structured, large-scale ETL (transform data at scale).  
- **Wrangling Data Flow**: For exploration/cleaning using Power Query (ad-hoc tasks).

# Azure Data Lake Storage (ADLS Gen2)

## Q4. What is ADLS Gen2 and why is it important?

It’s Azure’s storage for big data.  
**Features:**  
- Fast hierarchical file system (like folders).  
- Works well with Databricks & Synapse.  
- Secure with access control & encryption.

## Q5. How to secure ADLS Gen2?

- Use RBAC (Role-Based Access Control).  
- Set firewall & VNet rules.  
- Encrypt data at rest & in transit.  
- Monitor access with Azure Monitor/Sentinel.

# Azure Databricks & Spark

## Q6. What is Azure Databricks and how does it connect to Spark?

Databricks = cloud workspace for Spark.  
**Used for:**  
- Big data processing (with Spark clusters).  
- Machine learning.  
- Easy integration with ADLS & Synapse.

## Q7. Benefits of Apache Spark

- **Fast**: In-memory processing.  
- **Scalable**: Handles huge datasets.  
- **Flexible**: Works with Python, Scala, R, Java.  
- **Rich APIs**: MLlib, Spark SQL, Streaming.

# Azure Synapse Analytics

## Q8. Serverless SQL Pool vs Dedicated SQL Pool

- **Serverless**: Pay only per query, directly query data in data lake.  
- **Dedicated**: Pre-provisioned compute, best for heavy, high-performance queries.

## Q9. Can Synapse be a lakehouse?

Yes:  
- Store raw data in ADLS (data lake).  
- Query directly with Serverless SQL.  
- Combine structured + unstructured data → like a lake + warehouse in one.

# General Data Engineering

## Q10. What is Medallion Architecture?

Data is organized in 3 layers:  
- **Bronze**: Raw data.  
- **Silver**: Cleaned data.  
- **Gold**: Business-ready data.  
Ensures quality + scalability.

## Q11. Data Lake vs Data Warehouse

- **Data Lake**: Raw, any format, big data. Good for processing.  
- **Data Warehouse**: Clean, structured, optimized for reporting.

# Scenario-Based

## Q12. ADF pipeline fails at 2 a.m., how to fix?

- Check ADF monitoring → see where it failed.  
- Look at error logs.  
- Verify source/target connections.  
- Fix issue (credentials, mapping, etc.).  
- Rerun pipeline and monitor.

## Q13. How to optimize Spark jobs in Databricks?

- Cache reused data.  
- Optimize partitions.  
- Use Parquet (columnar storage).  
- Avoid heavy ops like groupByKey.  
- Tune cluster settings (memory, cores).
