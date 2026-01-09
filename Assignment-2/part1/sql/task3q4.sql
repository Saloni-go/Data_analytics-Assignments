-- .open assignment2.db
-- .read task3q4.sql

SELECT c.category_name, p.product_name, SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
FROM fact_order_items f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_category c ON p.category_id = c.category_id
GROUP BY GROUPING SETS (
    (c.category_name),           
    (c.category_name, p.product_name)  
)
ORDER BY c.category_name, total_revenue DESC;