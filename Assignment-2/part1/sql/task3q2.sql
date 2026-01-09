-- .open assignment2.db
-- .read task3q2.sql

WITH revenue_by_month AS (
    SELECT year, month, SUM(quantity * list_price * (1 - discount)) AS total_revenue,
        ROW_NUMBER() OVER (PARTITION BY year ORDER BY SUM(quantity * list_price * (1 - discount)) DESC) AS rn
    FROM fact_order_items f
    JOIN dim_time t ON f.order_date = t.order_date
    GROUP BY year, month
)
SELECT year, month, total_revenue
FROM revenue_by_month
WHERE rn = 1;