-- .open assignment2.db
-- .read task3q6.sql

SELECT cu.customer_id, cu.first_name || ' ' || cu.last_name AS customer_name,
    SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_spent
FROM fact_order_items f
JOIN dim_customer cu ON f.customer_id = cu.customer_id
GROUP BY cu.customer_id, customer_name
ORDER BY total_spent DESC
LIMIT 5;