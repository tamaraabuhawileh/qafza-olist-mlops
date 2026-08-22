-- ============================================================
-- Qafza MLOps Training 2026/2027
-- Task 1 - Olist Database Validation
-- ============================================================


-- ------------------------------------------------------------
-- 1. Validate row counts
-- ------------------------------------------------------------

SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'geolocation', COUNT(*) FROM geolocation
UNION ALL
SELECT 'product_category_translation', COUNT(*)
FROM product_category_translation;


-- ------------------------------------------------------------
-- 2. Read sample records
-- ------------------------------------------------------------

SELECT *
FROM orders
LIMIT 10;


-- ------------------------------------------------------------
-- 3. Two-table JOIN: Orders + Customers
-- Relationship: customer_id
-- ------------------------------------------------------------

SELECT
    o.order_id,
    o.order_status,
    c.customer_city,
    c.customer_state
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
LIMIT 20;


-- ------------------------------------------------------------
-- 4. Multi-table JOIN
-- Orders -> Order Items -> Products -> Sellers
-- ------------------------------------------------------------

SELECT
    o.order_id,
    o.order_status,
    p.product_category_name,
    s.seller_city,
    s.seller_state,
    oi.price,
    oi.freight_value
FROM orders AS o
JOIN order_items AS oi
    ON o.order_id = oi.order_id
JOIN products AS p
    ON oi.product_id = p.product_id
JOIN sellers AS s
    ON oi.seller_id = s.seller_id
LIMIT 20;


-- ------------------------------------------------------------
-- 5. Inspect delivery fields related to the future ML problem
-- ------------------------------------------------------------

SELECT
    order_id,
    order_status,
    order_purchase_timestamp,
    order_delivered_customer_date,
    order_estimated_delivery_date
FROM orders
WHERE order_status = 'delivered'
LIMIT 20;