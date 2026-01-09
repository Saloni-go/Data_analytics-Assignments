-- .open assignment2.db
-- .read task3q3.sql

SELECT c.category_name, SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
FROM fact_order_items f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_category c ON p.category_id = c.category_id
GROUP BY c.category_name
ORDER BY total_revenue DESC;