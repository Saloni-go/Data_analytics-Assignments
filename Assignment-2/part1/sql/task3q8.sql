-- .open assignment2.db
-- .read task3q8.sql

SELECT c.category_name, s.store_name, t.year,
    SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
FROM fact_order_items f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_category c ON p.category_id = c.category_id
JOIN dim_store s ON f.store_id = s.store_id
JOIN dim_time t ON f.order_date = t.order_date
GROUP BY CUBE (c.category_name, s.store_name, t.year)
ORDER BY c.category_name, s.store_name, t.year;