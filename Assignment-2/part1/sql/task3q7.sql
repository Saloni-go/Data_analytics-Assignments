-- .open assignment2.db
-- .read task3q7.sql

SELECT s.store_id, s.first_name || ' ' || s.last_name AS staff_name,
    SUM(f.quantity * f.list_price * (1 - f.discount)) AS staff_sales
FROM fact_order_items f
JOIN dim_staff s ON f.staff_id = s.staff_id
GROUP BY s.store_id, staff_name
ORDER BY s.store_id, staff_sales DESC;