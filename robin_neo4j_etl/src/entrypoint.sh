#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move Cochrane data from CSVs to Neo4j..."

# Run the ETL script
python cochrane_bulk_csv_write.py