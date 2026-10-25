-- Run this in MySQL Workbench. Adjust database name if needed.
CREATE DATABASE IF NOT EXISTS tellco_analytics;
USE tellco_analytics;
CREATE TABLE IF NOT EXISTS customer_scores (
  `MSISDN/Number` DOUBLE,
  `Dur. (ms)` DOUBLE,
  `Total Traffic (Bytes)` DOUBLE,
  `Sessions Frequency` INT,
  `TCP Retransmission (Bytes)` DOUBLE,
  `RTT (ms)` DOUBLE,
  `Throughput (kbps)` DOUBLE,
  `Engagement Cluster` INT,
  `Engagement Score` DOUBLE,
  `Experience Cluster` INT,
  `Experience Score` DOUBLE,
  `Satisfaction Score` DOUBLE,
  `Satisfaction Cluster` INT
);
-- Import outputs/final_customer_scores.csv via MySQL Workbench > Table Data Import Wizard.
-- Then run:
SELECT * FROM customer_scores LIMIT 10;
