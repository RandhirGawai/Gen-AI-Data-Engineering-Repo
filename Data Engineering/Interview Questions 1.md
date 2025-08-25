# Azure Synapse Analytics

Azure Synapse Analytics is Microsoft’s integrated platform combining modern data warehousing and big data analytics.

## Overview
- **Traditional Data Warehouses**: Handle structured, clean data (e.g., SQL tables).
- **Synapse Capabilities**: Processes both structured (tables) and unstructured data (JSON, logs, etc.).
- **Modes**:
  - **Provisioned**: Dedicated servers with fixed compute resources.
  - **Serverless**: Pay-per-query, no need to pre-allocate servers.
- **Integration**: Connects with Azure Data Lake, supports Apache Spark for big data, and uses T-SQL for familiar SQL querying.

## Key Features

### OPENROWSET
- **Purpose**: Query external data (e.g., CSV/Parquet files in Blob or Data Lake) directly without importing.
- **Use Case**: Fast data exploration or testing raw datasets.
- **Example**: Treat a file as a table for ad-hoc queries.

### External Tables
- **Definition**: Virtual tables that reference data stored outside Synapse (e.g., in Data Lake).
- **Advantage**: Query with SQL without duplicating data.
- **Process**: Create a pointer to external data, keeping storage efficient.

### Master Keys & Credentials
- **Master Key**: Encrypts sensitive data (e.g., passwords) like a secure locker key.
- **Database-Scoped Credential**: Securely stored login info (username/password or token) for accessing external data sources.

### Optimizing Query Performance
- **Materialized Views**: Precompute results for faster repeated queries.
- **Partitioning**: Split large tables into smaller chunks (e.g., by date).
- **File Format**: Use Parquet (compressed, column-based) instead of CSV for speed.
- **Caching/Indexing**: Store frequently accessed data for quick retrieval.
- **Resource Classes**: Allocate more compute power to heavy queries, less to smaller ones.

## Apache Spark in Synapse

### DataFrame vs RDD
- **RDD**: Raw distributed objects, flexible but requires manual management.
- **DataFrame**: Structured, table-like data structure optimized for SQL-like queries.

### Fault Tolerance
- Spark tracks data lineage (creation steps) to recompute lost data automatically.

### Driver vs Executors
- **Driver**: Plans tasks and monitors progress.
- **Executors**: Perform actual data processing.

### Parquet Files
- **Characteristics**: Column-based, highly compressed, reads only necessary columns.
- **Benefits**: Faster and more storage-efficient than CSV.

### Tuning Spark
- Allocate appropriate memory and cores to executors.
- Reduce shuffle partitions to minimize data movement.
- Use broadcast joins for small tables.
- Cache frequently accessed data for performance.

## Hadoop Ecosystem

### Main Components
- **HDFS**: Distributed file system storing data in chunks across machines.
- **MapReduce**: Batch processing framework (older, disk-heavy).
- **YARN**: Resource manager for assigning compute resources.
- **Hive/Pig**: Tools for querying and scripting on Hadoop.

### Reliability in HDFS
- **Replication**: Each data block is replicated (default: 3 times) for fault tolerance.
- **NameNode**: Stores metadata (file names, block locations).
- **DataNode**: Stores actual data blocks.

### Hadoop vs Spark
- **Hadoop**: Slower, disk-based processing.
- **Spark**: Faster, in-memory processing.

## Data Engineering Workflow

### Data Lake vs Data Warehouse
- **Data Lake**: Stores raw, unprocessed data in any format, ideal for big data.
- **Data Warehouse**: Clean, structured data optimized for reporting.

### Gold Layer
- Final, clean, and curated dataset ready for reporting or visualization (e.g., Power BI).

### Ensuring Data Quality
- Apply validation rules.
- Monitor data pipelines.
- Enforce schema checks to maintain consistency.

### Challenges
- Managing schema evolution (changing data structures).
- Handling pipeline failures.
- Dealing with duplicates or late-arriving data.

### Integrating Spark with Azure Data Lake
- Configure Spark clusters with credentials.
- Use `spark.read.parquet("path")` to read data directly from Data Lake.

## Scenario-Based Solutions

### Joining 1TB + 10MB Dataset
- **Solution**: Use broadcast join to distribute the small dataset to all workers, avoiding expensive shuffles.

### Slow Synapse Query
- **Solution**: Implement partitioning, materialized views, caching, and indexing.

### Cleaning Raw Data in ADLS
- **Process**:
  1. Load raw data with Spark.
  2. Clean (e.g., remove nulls, enrich data).
  3. Save to gold layer in Parquet format.

### Securing Synapse
- Use **RBAC** (role-based access control) for permissions.
- Store credentials securely with master keys.
- Leverage **managed identities** for authentication.

## Machine Learning with Big Data

### Integrating PyTorch with Spark
- **Preprocessing**: Use Spark for large-scale data preparation.
- **Training**: Employ distributed tools like Horovod or Spark MLlib.

### Why Spark MLlib?
- Natively handles big data, unlike scikit-learn, which is limited to single-machine processing.

### Deploying ML in Synapse
- **Steps**:
  1. Prepare data using Synapse pipelines.
  2. Train models in Azure Machine Learning (AML) or Spark.
  3. Deploy models in Synapse for batch or real-time predictions.
