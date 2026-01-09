-- .open assignment2.db
-- .read task3q1.sql

SELECT year, quarter, month, SUM(quantity * list_price * (1 - discount)) AS total_revenue
FROM fact_order_items f
JOIN dim_time t ON f.order_date = t.order_date
GROUP BY ROLLUP (year, quarter, month)
ORDER BY year, quarter, month;